# Autoregressive Transformer with KV-cache for faster generation
# - Implements an autoregressive Transformer using custom CachedTransformerLayer
# - Supports init_kv_cache to populate per-layer key/value caches from a seed sequence
# - Supports generate_step that uses cached keys/values and appends new ones (incremental decoding)

import os
import math
import random
from typing import List, Optional, Tuple, Dict

import pretty_midi
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Defaults
seq_length = 24
vocab_size = 128
checkpoint_path = 'model/model.pth'


# ---------------- Data processing ----------------
def read_midi_notes() -> List[List[float]]:
    midi_inputs = []
    filenames = [os.path.join('datasets', f) for f in os.listdir('datasets') if f.endswith('.midi')]
    for f in filenames:
        pm = pretty_midi.PrettyMIDI(f)
        instruments = pm.instruments
        if len(instruments) == 0:
            continue
        instrument = instruments[0]
        notes = instrument.notes
        if not notes:
            continue
        sorted_notes = sorted(notes, key=lambda note: note.start)
        prev_start = sorted_notes[0].start
        for note in sorted_notes:
            step = note.start - prev_start
            duration = note.end - note.start
            prev_start = note.start
            midi_inputs.append([float(note.pitch), float(step), float(duration)])
    return midi_inputs


class ARMidiSequenceDataset(Dataset):
    def __init__(self, midi_inputs: List[List[float]], normalize: bool = True):
        super().__init__()
        self.midi_inputs = midi_inputs
        self.window = seq_length + 1
        self.sequences = self._make_sequences()
        self.normalize = normalize
        steps = [item[1] for item in midi_inputs]
        durations = [item[2] for item in midi_inputs]
        self.step_mean = float(sum(steps) / len(steps)) if steps else 0.0
        self.duration_mean = float(sum(durations) / len(durations)) if durations else 0.0
        self.step_std = float((sum((s - self.step_mean) ** 2 for s in steps) / len(steps)) ** 0.5) if steps else 1.0
        self.duration_std = float((sum((d - self.duration_mean) ** 2 for d in durations) / len(durations)) ** 0.5) if durations else 1.0
        if self.step_std == 0:
            self.step_std = 1.0
        if self.duration_std == 0:
            self.duration_std = 1.0

    def _make_sequences(self):
        seqs = []
        n = len(self.midi_inputs)
        for i in range(0, n - self.window + 1):
            window = self.midi_inputs[i:i + self.window]
            seqs.append(window)
        return seqs

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        w = self.sequences[idx]
        inputs = w[:-1]
        targets = w[1:]
        pitch_inputs = torch.tensor([int(x[0]) for x in inputs], dtype=torch.long)
        steps_in = torch.tensor([x[1] for x in inputs], dtype=torch.float32)
        durs_in = torch.tensor([x[2] for x in inputs], dtype=torch.float32)
        pitch_targets = torch.tensor([int(x[0]) for x in targets], dtype=torch.long)
        step_targets = torch.tensor([x[1] for x in targets], dtype=torch.float32)
        dur_targets = torch.tensor([x[2] for x in targets], dtype=torch.float32)

        if self.normalize:
            steps_in = (steps_in - self.step_mean) / self.step_std
            durs_in = (durs_in - self.duration_mean) / self.duration_std
            step_targets_norm = (step_targets - self.step_mean) / self.step_std
            dur_targets_norm = (dur_targets - self.duration_mean) / self.duration_std
        else:
            step_targets_norm = step_targets
            dur_targets_norm = dur_targets

        cont_inputs = torch.stack([steps_in, durs_in], dim=-1)

        return (pitch_inputs, cont_inputs), {'pitch': pitch_targets, 'step': step_targets_norm, 'duration': dur_targets_norm}


# ---------------- Cached Transformer Layer ----------------
class CachedTransformerLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, 'd_model must be divisible by nhead'

        # projection layers for q,k,v
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq, d_model) -> (B, nhead, seq, head_dim)
        B, S, D = x.size()
        x = x.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        return x

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, nhead, seq, head_dim) -> (B, seq, d_model)
        x = x.transpose(1, 2).contiguous()
        B, S, _, _ = x.size()
        return x.view(B, S, self.d_model)

    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, seq, d_model)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # split heads
        qh = self._split_heads(q)
        kh = self._split_heads(k)
        vh = self._split_heads(v)
        # scaled dot product
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale  # (B, nhead, seq, seq)
        if causal_mask is not None:
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, vh)  # (B, nhead, seq, head_dim)
        context = self._combine_heads(context)  # (B, seq, d_model)
        out = self.out_proj(context)
        x = self.norm1(x + self.dropout(out))
        # feedforward
        ff = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff))
        return x, kh, vh

    def forward_step(self, x_new: torch.Tensor, cache_k: Optional[torch.Tensor], cache_v: Optional[torch.Tensor], causal_mask_len: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x_new: (B, 1, d_model) single-step input
        # cache_k/cache_v: (B, nhead, seq_cached, head_dim) or None
        q = self.q_proj(x_new)  # (B,1,d_model)
        k = self.k_proj(x_new)
        v = self.v_proj(x_new)
        qh = self._split_heads(q).squeeze(2)  # (B, nhead, head_dim)
        kh = self._split_heads(k).squeeze(2)  # (B, nhead, head_dim)
        vh = self._split_heads(v).squeeze(2)
        # prepare k_all, v_all: (B, nhead, seq_all, head_dim)
        if cache_k is not None:
            k_all = torch.cat([cache_k, kh.unsqueeze(2)], dim=2)
            v_all = torch.cat([cache_v, vh.unsqueeze(2)], dim=2)
        else:
            k_all = kh.unsqueeze(2)
            v_all = vh.unsqueeze(2)
        # compute attention: qh (B,nhead,head_dim) vs k_all (B,nhead,seq_all,head_dim)
        # expand qh to (B, nhead, 1, head_dim)
        qh_exp = qh.unsqueeze(2)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(qh_exp, k_all.transpose(-2, -1)) * scale  # (B,nhead,1,seq_all)
        # causal mask: ensure q only attends to <= current position -> because we build k_all from past+current, it's already causal
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v_all)  # (B,nhead,1,head_dim)
        # combine heads
        context = context.unsqueeze(2)  # make shape (B,nhead,1,head_dim) -> _combine expects (B,nhead,seq,head_dim)
        context = context.transpose(1, 2).contiguous()  # -> (B,1,nhead,head_dim)
        B = context.size(0)
        context = context.view(B, 1, self.d_model)
        out = self.out_proj(context)
        x_out = self.norm1(x_new + self.dropout(out))
        ff = self.linear2(self.dropout(F.relu(self.linear1(x_out))))
        x_out = self.norm2(x_out + self.dropout(ff))
        # return x_out (B,1,d_model) and new k,v in cached head shape
        return x_out, k_all, v_all


# ---------------- Autoregressive Transformer using cached layers ----------------
class AutoregressiveTransformerCached(nn.Module):
    def __init__(self,
                 vocab_size: int = 128,
                 embed_dim: int = 64,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 cont_dim: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.nhead = nhead
        self.pitch_embed = nn.Embedding(vocab_size, embed_dim)
        self.cont_proj = nn.Linear(cont_dim, d_model - embed_dim)
        self.input_proj = nn.Linear(d_model, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            CachedTransformerLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)
        ])
        self.joint_proj = nn.Linear(d_model, d_model)
        self.pitch_head = nn.Linear(d_model, vocab_size)
        self.step_head = nn.Linear(d_model, 1)
        self.duration_head = nn.Linear(d_model, 1)

    def forward(self, pitch_idxs: torch.Tensor, cont_inputs: torch.Tensor):
        # training forward (full sequence) -> returns per-position outputs
        emb = self.pitch_embed(pitch_idxs)  # (B, seq, embed)
        cont = self.cont_proj(cont_inputs)
        x = torch.cat([emb, cont], dim=-1)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        # create causal mask (additive -inf)
        seq_len = x.size(1)
        mask = generate_causal_mask(seq_len, device=x.device)
        per_layer_kv = []
        for layer in self.layers:
            x, k, v = layer(x, causal_mask=mask)
            per_layer_kv.append((k, v))
        joint = torch.relu(self.joint_proj(x))
        pitch_logits = self.pitch_head(joint)
        step_preds = self.step_head(joint).squeeze(-1)
        dur_preds = self.duration_head(joint).squeeze(-1)
        return {'pitch': pitch_logits, 'step': step_preds, 'duration': dur_preds}, per_layer_kv

    def init_kv_cache(self, pitch_idxs: torch.Tensor, cont_inputs: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        # run a forward pass and return per-layer kv caches (in head-shaped form)
        _, per_layer_kv = self.forward(pitch_idxs, cont_inputs)
        # per_layer_kv: list of (k, v) where k shape (B, nhead, seq, head_dim)
        return per_layer_kv

    def generate_step_with_cache(self, last_pitch_idx: torch.Tensor, last_cont: torch.Tensor, pos: int,
                                 kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]) -> Tuple[int, float, float, List[Tuple[torch.Tensor, torch.Tensor]]]:
        # last_pitch_idx: (B,) single token index
        # last_cont: (B, cont_dim) continuous features for last token (already denormed)
        # pos: position index for positional encoding (0-based)
        # kv_cache: list of (k,v) per layer or None
        B = last_pitch_idx.size(0)
        device = last_pitch_idx.device
        # build input vector x_new (B,1,d_model)
        emb = self.pitch_embed(last_pitch_idx).unsqueeze(1)  # (B,1,embed)
        cont_proj = self.cont_proj(last_cont.unsqueeze(1))  # (B,1,d_model-embed)
        x = torch.cat([emb, cont_proj], dim=-1)
        x = self.input_proj(x)
        # add positional encoding for single position: use stored pe buffer
        # pos_enc: pe shape (1, max_len, d_model)
        pe_slice = self.pos_enc.pe[:, pos:pos+1, :]
        x = x + pe_slice.to(device)
        new_kv = []
        # iterate layers and use forward_step to update cache
        for i, layer in enumerate(self.layers):
            cache_k = None
            cache_v = None
            if kv_cache is not None:
                cache_k, cache_v = kv_cache[i]
            x, k_all, v_all = layer.forward_step(x, cache_k, cache_v)
            # store updated cache (exclude q)
            new_kv.append((k_all, v_all))
        joint = torch.relu(self.joint_proj(x))  # (B,1,d_model)
        pitch_logits = self.pitch_head(joint).squeeze(1)  # (B, vocab)
        step_pred = self.step_head(joint).squeeze(1)  # (B,)
        dur_pred = self.duration_head(joint).squeeze(1)  # (B,)
        return pitch_logits, step_pred, dur_pred, new_kv


# Reuse other helpers from previous file
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


def generate_causal_mask(sz: int, device: Optional[torch.device] = None) -> torch.Tensor:
    mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    if device is not None:
        mask = mask.to(device)
    return mask


# ---------------- Sampling helpers ----------------
def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0) -> torch.Tensor:
    top_k = int(top_k)
    if top_k > 0:
        values, indices = torch.topk(logits, top_k)
        min_values = values[-1]
        filtered_logits = torch.where(logits < min_values, torch.tensor(float('-inf'), device=logits.device), logits)
    else:
        filtered_logits = logits
    if top_p > 0.0 and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        cutoff = cumulative_probs > top_p
        if cutoff[0]:
            cutoff[0] = False
        sorted_logits[cutoff] = float('-inf')
        filtered_logits = torch.full_like(filtered_logits, float('-inf'))
        filtered_logits[sorted_indices] = sorted_logits
    return filtered_logits


def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0) -> int:
    if temperature <= 0:
        raise ValueError('Temperature must be > 0')
    logits = logits / float(temperature)
    filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    if torch.isinf(filtered_logits).all():
        filtered_logits = logits
    dist = torch.distributions.Categorical(logits=filtered_logits)
    sample = dist.sample().item()
    return int(sample)


# ---------------- Training & Checkpoint ----------------
def build_model(model_type: str = 'transformer', **kwargs) -> nn.Module:
    if model_type == 'transformer':
        return AutoregressiveTransformerCached(vocab_size=vocab_size,
                                               embed_dim=kwargs.get('embed_dim', 64),
                                               d_model=kwargs.get('d_model', 128),
                                               nhead=kwargs.get('nhead', 8),
                                               num_layers=kwargs.get('num_layers', 4),
                                               dim_feedforward=kwargs.get('dim_feedforward', 256),
                                               dropout=kwargs.get('dropout', 0.1))
    else:
        return MusicLSTMImproved(vocab_size=vocab_size,
                                 embed_dim=kwargs.get('embed_dim', 32),
                                 use_embedding=kwargs.get('use_embedding', True),
                                 hidden_size=kwargs.get('hidden_size', 256),
                                 num_layers=kwargs.get('num_layers', 2))


class MusicLSTMImproved(nn.Module):
    def __init__(self,
                 vocab_size: int = 128,
                 embed_dim: int = 32,
                 use_embedding: bool = True,
                 cont_dim: int = 2,
                 hidden_size: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.use_embedding = use_embedding
        self.embed_dim = embed_dim if use_embedding else 0
        if use_embedding:
            self.pitch_embedding = nn.Embedding(vocab_size, embed_dim)
        lstm_input_size = (self.embed_dim + cont_dim) if use_embedding else (cont_dim + 1)
        self.lstm = nn.LSTM(lstm_input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.pitch_head = nn.Linear(hidden_size, vocab_size)
        self.step_head = nn.Linear(hidden_size, 1)
        self.duration_head = nn.Linear(hidden_size, 1)

    def forward(self, pitch_idxs: Optional[torch.Tensor], cont_inputs: torch.Tensor):
        batch_size, seq_len, _ = cont_inputs.shape
        if self.use_embedding:
            assert pitch_idxs is not None
            emb = self.pitch_embedding(pitch_idxs)
            x = torch.cat([emb, cont_inputs], dim=-1)
        else:
            pitch_float = pitch_idxs.unsqueeze(-1).float() / float(vocab_size)
            x = torch.cat([pitch_float, cont_inputs], dim=-1)
        out, (hn, cn) = self.lstm(x)
        last = out[:, -1, :]
        last = self.layernorm(last)
        last = self.dropout(last)
        pitch_logits = self.pitch_head(last)
        step = self.step_head(last).squeeze(-1)
        duration = self.duration_head(last).squeeze(-1)
        return {'pitch': pitch_logits, 'step': step, 'duration': duration}


class MSEWithPositivePressure(nn.Module):
    def __init__(self, pressure: float = 10.0, mean: float = 0.0, std: float = 1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.pressure = pressure
        self.mean = mean
        self.std = std

    def forward(self, y_pred_norm: torch.Tensor, y_true: torch.Tensor):
        y_true_norm = (y_true - self.mean) / self.std
        mse_loss = self.mse(y_pred_norm, y_true_norm)
        y_pred_denorm = y_pred_norm * self.std + self.mean
        negative_part = torch.relu(-y_pred_denorm)
        positive_pressure = self.pressure * torch.mean(negative_part)
        return mse_loss + positive_pressure


def train(epochs: int = 50,
          batch_size: int = 64,
          lr: float = 0.005,
          device: Optional[str] = None,
          model_type: str = 'transformer',
          **model_kwargs):
    device = torch.device(device) if device else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    midi_inputs = read_midi_notes()
    if len(midi_inputs) < seq_length + 1:
        print('Not enough notes to train.')
        return

    if model_type == 'transformer':
        dataset = ARMidiSequenceDataset(midi_inputs, normalize=True)
    else:
        dataset_all = ARMidiSequenceDataset(midi_inputs, normalize=True)
        class SingleTargetWrapper(Dataset):
            def __init__(self, ar_ds):
                self.ar_ds = ar_ds
            def __len__(self):
                return len(self.ar_ds)
            def __getitem__(self, idx):
                (pitch_inputs, cont_inputs), targets = self.ar_ds[idx]
                pitch_target_last = targets['pitch'][-1]
                step_target_last = targets['step'][-1]
                dur_target_last = targets['duration'][-1]
                return (pitch_inputs, cont_inputs), {'pitch': pitch_target_last, 'step': step_target_last, 'duration': dur_target_last}
        dataset = SingleTargetWrapper(dataset_all)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = build_model(model_type=model_type, **model_kwargs).to(device)

    ce_loss = nn.CrossEntropyLoss()
    if model_type == 'transformer':
        step_mean = dataset.step_mean
        step_std = dataset.step_std
        duration_mean = dataset.duration_mean
        duration_std = dataset.duration_std
    else:
        step_mean = dataset_all.step_mean
        step_std = dataset_all.step_std
        duration_mean = dataset_all.duration_mean
        duration_std = dataset_all.duration_std

    step_loss_fn = MSEWithPositivePressure(pressure=10.0, mean=step_mean, std=step_std)
    duration_loss_fn = MSEWithPositivePressure(pressure=10.0, mean=duration_mean, std=duration_std)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)

    best_loss = float('inf')
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for (pitch_inputs, cont_inputs), batch_labels in loader:
            pitch_inputs = pitch_inputs.to(device)
            cont_inputs = cont_inputs.to(device)
            pitch_labels = batch_labels['pitch'].to(device)
            step_labels = batch_labels['step'].to(device)
            duration_labels = batch_labels['duration'].to(device)

            outputs, _ = model(pitch_inputs, cont_inputs) if model_type == 'transformer' else (model(pitch_inputs, cont_inputs), None)
            pitch_logits = outputs['pitch']
            step_preds = outputs['step']
            dur_preds = outputs['duration']

            if pitch_logits.dim() == 3:
                B, S, C = pitch_logits.shape
                loss_pitch = ce_loss(pitch_logits.view(B * S, C), pitch_labels.view(B * S))
                loss_step = step_loss_fn(step_preds, step_labels)
                loss_duration = duration_loss_fn(dur_preds, duration_labels)
            else:
                loss_pitch = ce_loss(pitch_logits, pitch_labels)
                loss_step = step_loss_fn(step_preds, step_labels)
                loss_duration = duration_loss_fn(dur_preds, duration_labels)

            loss = 0.05 * loss_pitch + 1.0 * loss_step + 1.0 * loss_duration

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        epoch_loss = running_loss / max(1, n_batches)
        print(f'Epoch {epoch}/{epochs} - loss: {epoch_loss:.6f}')
        scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': epoch_loss,
                'dataset_stats': {
                    'step_mean': step_mean,
                    'step_std': step_std,
                    'duration_mean': duration_mean,
                    'duration_std': duration_std,
                },
                'model_meta': {
                    'model_type': model_type,
                    'model_kwargs': model_kwargs,
                }
            }, checkpoint_path)
            print(f'  Saved best checkpoint (loss {epoch_loss:.6f}) to {checkpoint_path}')

    print('Training finished.')


# ---------------- Prediction (with KV cache) ----------------
def predict_midi(num_predictions: int = 600, device: Optional[str] = None, model_type: str = 'transformer', temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0, **model_kwargs):
    device = torch.device(device) if device else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = build_model(model_type=model_type, **model_kwargs).to(device)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        stats = ckpt.get('dataset_stats', None)
        if stats:
            step_mean = stats['step_mean']
            step_std = stats['step_std']
            duration_mean = stats['duration_mean']
            duration_std = stats['duration_std']
        else:
            tmp = ARMidiSequenceDataset(read_midi_notes(), normalize=True)
            step_mean = tmp.step_mean
            step_std = tmp.step_std
            duration_mean = tmp.duration_mean
            duration_std = tmp.duration_std
        print('Loaded checkpoint:', checkpoint_path)
    else:
        print('No checkpoint found, running with randomly initialized model.')
        tmp = ARMidiSequenceDataset(read_midi_notes(), normalize=True)
        step_mean = tmp.step_mean
        step_std = tmp.step_std
        duration_mean = tmp.duration_mean
        duration_std = tmp.duration_std

    model.eval()
    midi_inputs = read_midi_notes()
    if len(midi_inputs) < seq_length:
        print('Not enough notes to predict.')
        return

    # seed with seq_length notes
    seed = random.sample(midi_inputs, seq_length)
    # prepare seed tensors (normalized cont inputs)
    pitch_seed = torch.tensor([[int(x[0]) for x in seed]], dtype=torch.long).to(device)
    steps = torch.tensor([[x[1] for x in seed]], dtype=torch.float32).to(device)
    durs = torch.tensor([[x[2] for x in seed]], dtype=torch.float32).to(device)
    steps_norm = (steps - step_mean) / step_std
    durs_norm = (durs - duration_mean) / duration_std
    cont_seed = torch.stack([steps_norm.squeeze(0), durs_norm.squeeze(0)], dim=-1).unsqueeze(0)  # (1, seq, 2)
    cont_seed = cont_seed.to(device)

    generated = list(seed)

    # build KV cache from seed sequence if transformer
    kv_cache = None
    if model_type == 'transformer':
        kv_cache = model.init_kv_cache(pitch_seed, cont_seed)
    else:
        kv_cache = None

    # generation loop: each step uses generate_step_with_cache which appends caches
    for i in range(num_predictions):
        # last token features
        last = generated[-1]
        last_pitch = torch.tensor([int(last[0])], dtype=torch.long).to(device)
        last_steps = torch.tensor([last[1]], dtype=torch.float32).to(device)
        last_durs = torch.tensor([last[2]], dtype=torch.float32).to(device)
        # last cont normalized
        last_steps_norm = (last_steps - step_mean) / step_std
        last_durs_norm = (last_durs - duration_mean) / duration_std
        last_cont = torch.stack([last_steps_norm, last_durs_norm], dim=-1).squeeze(0)  # (2,)

        if model_type == 'transformer':
            pos = len(generated) - 1  # position index for new token
            pitch_logits, step_pred_norm, dur_pred_norm, kv_cache = model.generate_step_with_cache(last_pitch.unsqueeze(0), last_cont.unsqueeze(0), pos, kv_cache)
            # pitch_logits: (B=1, vocab)
            pitch = sample_from_logits(pitch_logits[0], temperature=temperature, top_k=top_k, top_p=top_p)
            step = step_pred_norm[0].item() * step_std + step_mean
            duration = dur_pred_norm[0].item() * duration_std + duration_mean
        else:
            outputs = model(last_pitch.unsqueeze(0).unsqueeze(0), last_cont.unsqueeze(0).unsqueeze(0))
            pitch_logits = outputs['pitch'][0]
            pitch = sample_from_logits(pitch_logits, temperature=temperature, top_k=top_k, top_p=top_p)
            step = outputs['step'][0].item()
            duration = outputs['duration'][0].item()

        step = max(0.0, float(step))
        duration = max(0.01, float(duration))
        generated.append([pitch, step, duration])

    # reconstruct midi
    prev_start = 0.0
    midi_notes = []
    for m in generated:
        pitch, step, duration = m
        start = prev_start + step
        end = start + duration
        prev_start = start
        midi_notes.append([int(pitch), float(start), float(end)])

    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'))
    for n in midi_notes:
        note = pretty_midi.Note(velocity=100, pitch=int(n[0]), start=float(n[1]), end=float(n[2]))
        instrument.notes.append(note)
    pm.instruments.append(instrument)
    out_path = 'out_pytorch.midi'
    pm.write(out_path)
    print('Wrote generated MIDI to', out_path)


# ---------------- CLI entry ----------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Autoregressive Transformer with KV-cache for faster generation')
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--predict', action='store_true', help='Run prediction (generate MIDI)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--num-predictions', type=int, default=600)
    parser.add_argument('--model-type', type=str, default='transformer', choices=['transformer', 'lstm'])
    # transformer specific
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--d-model', type=int, default=128, help='Transformer d_model')
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--dim-feedforward', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    # lstm specific
    parser.add_argument('--lstm-embed-dim', type=int, default=32)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--lstm-num-layers', type=int, default=2)
    parser.add_argument('--device', type=str, default='', help='torch device string, e.g. cpu or cuda:0')
    # sampling controls
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature (>0).')
    parser.add_argument('--top-k', type=int, default=0, help='Top-k sampling (0 to disable)')
    parser.add_argument('--top-p', type=float, default=0.0, help='Top-p (nucleus) sampling (0.0 to disable)')

    args = parser.parse_args()

    model_type = args.model_type

    if args.train:
        model_kwargs = {}
        if model_type == 'transformer':
            model_kwargs = dict(embed_dim=args.embed_dim, d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
                                dim_feedforward=args.dim_feedforward, dropout=args.dropout)
        else:
            model_kwargs = dict(embed_dim=args.lstm_embed_dim, use_embedding=True, hidden_size=args.hidden_size, num_layers=args.lstm_num_layers)
        train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device or None, model_type=model_type, **model_kwargs)

    if args.predict:
        model_kwargs = {}
        if model_type == 'transformer':
            model_kwargs = dict(embed_dim=args.embed_dim, d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
                                dim_feedforward=args.dim_feedforward, dropout=args.dropout)
        else:
            model_kwargs = dict(embed_dim=args.lstm_embed_dim, use_embedding=True, hidden_size=args.hidden_size, num_layers=args.lstm_num_layers)
        predict_midi(num_predictions=args.num_predictions, device=args.device or None, model_type=model_type, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, **model_kwargs)

    if (not args.train) and (not args.predict):
        predict_midi()
