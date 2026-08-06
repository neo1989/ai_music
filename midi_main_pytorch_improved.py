# Improved Autoregressive Transformer with:
# - Joint pitch-duration embedding
# - Rhythm/Beat information integration
# - Enhanced incremental decoding with better coherence
# - Multi-head attention with causal masking

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
seq_length = 32
vocab_size = 128
checkpoint_path = 'model/model_improved.pth'


# ============== Rhythm/Beat Analysis ==============
class RhythmAnalyzer:
    """分析MIDI中的节奏和节拍信息"""
    def __init__(self, tempo: float = 120.0):
        self.tempo = tempo
        self.beat_duration = 60.0 / tempo  # 一个beat的时长
    
    def analyze_time_signature(self, notes: List) -> Tuple[int, int]:
        """简单推断拍号 (假设4/4)"""
        return (4, 4)
    
    def quantize_to_beat(self, time_val: float, grid: int = 16) -> int:
        """量化到最近的beat grid (1/grid beat)"""
        quantized = round(time_val / (self.beat_duration / grid))
        return max(1, quantized)  # 最少1个grid单位
    
    def extract_beat_position(self, time_val: float, grid: int = 16) -> float:
        """计算在当前beat中的位置 (0.0 ~ 1.0)"""
        beat_unit = self.beat_duration / grid
        beat_pos = (time_val % self.beat_duration) / self.beat_duration
        return beat_pos
    
    def extract_beat_strength(self, quantized_step: int, grid: int = 16) -> float:
        """根据量化位置计算beat强度 (downbeat > on-beat > off-beat)"""
        # 假设4/4拍，grid=16时: pos 0,4,8,12是downbeat，其他是off-beat
        beat_pos = quantized_step % grid
        if beat_pos == 0:
            return 1.0  # downbeat
        elif beat_pos % 4 == 0:
            return 0.7  # on-beat
        else:
            return 0.3  # off-beat


# ============== Data Processing ==============
def read_midi_notes(rhythm_grid: int = 16) -> Tuple[List[List[float]], RhythmAnalyzer]:
    """读取MIDI音符，返回[pitch, step, duration, beat_pos, beat_strength]"""
    midi_inputs = []
    filenames = [os.path.join('datasets', f) for f in os.listdir('datasets') if f.endswith('.midi')]
    
    rhythm_analyzer = RhythmAnalyzer(tempo=120.0)
    
    for f in filenames:
        pm = pretty_midi.PrettyMIDI(f)
        if hasattr(pm, 'get_end_time'):
            estimated_tempo = pm.estimate_tempo()
            if estimated_tempo > 0:
                rhythm_analyzer.tempo = estimated_tempo
        
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
            
            # 计算节奏特征
            beat_pos = rhythm_analyzer.extract_beat_position(note.start, grid=rhythm_grid)
            quantized_step = rhythm_analyzer.quantize_to_beat(step, grid=rhythm_grid)
            beat_strength = rhythm_analyzer.extract_beat_strength(quantized_step, grid=rhythm_grid)
            
            midi_inputs.append([
                float(note.pitch),        # 0: pitch
                float(step),              # 1: step
                float(duration),          # 2: duration
                float(beat_pos),          # 3: beat position (0.0-1.0)
                float(beat_strength)      # 4: beat strength
            ])
    
    return midi_inputs, rhythm_analyzer


# ============== Joint Pitch-Duration Embedding ==============
class JointPitchDurationEmbedding(nn.Module):
    """联合pitch和duration的embedding层"""
    def __init__(self, vocab_size: int, embed_dim: int, max_duration: float = 4.0, duration_buckets: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.duration_buckets = duration_buckets
        self.max_duration = max_duration
        
        # Pitch embedding
        self.pitch_embed = nn.Embedding(vocab_size, embed_dim // 2)
        
        # Duration embedding (bucket-based)
        self.duration_embed = nn.Embedding(duration_buckets, embed_dim // 2)
        
        # Joint projection
        self.joint_proj = nn.Linear(embed_dim, embed_dim)
    
    def _duration_to_bucket(self, duration: torch.Tensor) -> torch.Tensor:
        """将duration连续值映射到bucket索引"""
        normalized = torch.clamp(duration / self.max_duration, 0.0, 1.0)
        bucket_idx = (normalized * (self.duration_buckets - 1)).long()
        return bucket_idx
    
    def forward(self, pitch_idxs: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pitch_idxs: (B, seq) long tensor
            durations: (B, seq) float tensor
        Returns:
            joint_embed: (B, seq, embed_dim)
        """
        # Pitch embedding
        pitch_emb = self.pitch_embed(pitch_idxs)  # (B, seq, embed_dim//2)
        
        # Duration to bucket and embed
        duration_buckets = self._duration_to_bucket(durations)
        duration_emb = self.duration_embed(duration_buckets)  # (B, seq, embed_dim//2)
        
        # Concatenate and project
        joint = torch.cat([pitch_emb, duration_emb], dim=-1)  # (B, seq, embed_dim)
        joint = self.joint_proj(joint)
        
        return joint


# ============== Rhythm-Aware Attention ==============
class RhythmAwareCachedTransformerLayer(nn.Module):
    """Transformer layer with rhythm-aware attention"""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, 'd_model must be divisible by nhead'
        
        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Rhythm bias (learnable)
        self.rhythm_bias = nn.Parameter(torch.zeros(nhead, 1, 1))
        
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.size()
        x = x.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        return x
    
    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).contiguous()
        B, S, _, _ = x.size()
        return x.view(B, S, self.d_model)
    
    def forward(self, x: torch.Tensor, rhythm_weights: Optional[torch.Tensor] = None,
                causal_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, seq, d_model)
            rhythm_weights: (B, seq) beat strength weights
            causal_mask: causal attention mask
        Returns:
            x, k, v
        """
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        qh = self._split_heads(q)
        kh = self._split_heads(k)
        vh = self._split_heads(v)
        
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale  # (B, nhead, seq, seq)
        
        # Apply rhythm weights as attention bias
        if rhythm_weights is not None:
            rhythm_bias = rhythm_weights.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, seq)
            scores = scores + (1.0 - rhythm_bias) * self.rhythm_bias
        
        if causal_mask is not None:
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, vh)
        context = self._combine_heads(context)
        out = self.out_proj(context)
        x = self.norm1(x + self.dropout(out))
        
        # Feedforward
        ff = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff))
        
        return x, kh, vh
    
    def forward_step(self, x_new: torch.Tensor, rhythm_weight: Optional[torch.Tensor],
                     cache_k: Optional[torch.Tensor], cache_v: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Incremental decoding step"""
        q = self.q_proj(x_new)
        k = self.k_proj(x_new)
        v = self.v_proj(x_new)
        
        qh = self._split_heads(q).squeeze(2)  # (B, nhead, head_dim)
        kh = self._split_heads(k).squeeze(2)
        vh = self._split_heads(v).squeeze(2)
        
        if cache_k is not None:
            k_all = torch.cat([cache_k, kh.unsqueeze(2)], dim=2)
            v_all = torch.cat([cache_v, vh.unsqueeze(2)], dim=2)
        else:
            k_all = kh.unsqueeze(2)
            v_all = vh.unsqueeze(2)
        
        qh_exp = qh.unsqueeze(2)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(qh_exp, k_all.transpose(-2, -1)) * scale
        
        # Apply rhythm bias
        if rhythm_weight is not None:
            rhythm_bias = (1.0 - rhythm_weight).unsqueeze(1).unsqueeze(1)  # (B, 1, 1)
            scores = scores + rhythm_bias * self.rhythm_bias
        
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v_all)
        context = context.unsqueeze(2)
        context = context.transpose(1, 2).contiguous()
        B = context.size(0)
        context = context.view(B, 1, self.d_model)
        
        out = self.out_proj(context)
        x_out = self.norm1(x_new + self.dropout(out))
        
        ff = self.linear2(self.dropout(F.relu(self.linear1(x_out))))
        x_out = self.norm2(x_out + self.dropout(ff))
        
        return x_out, k_all, v_all


# ============== Improved Autoregressive Transformer ==============
class ImprovedAutoregressiveTransformer(nn.Module):
    def __init__(self,
                 vocab_size: int = 128,
                 embed_dim: int = 64,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 duration_buckets: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.nhead = nhead
        
        # Joint pitch-duration embedding
        self.joint_pitch_duration_embed = JointPitchDurationEmbedding(
            vocab_size, embed_dim, duration_buckets=duration_buckets
        )
        
        # Rhythm encoding (beat_pos + beat_strength)
        self.rhythm_embed = nn.Linear(2, d_model - embed_dim)
        
        # Input projection
        self.input_proj = nn.Linear(d_model, d_model)
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model)
        
        # Transformer layers with rhythm awareness
        self.layers = nn.ModuleList([
            RhythmAwareCachedTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Output heads
        self.joint_proj = nn.Linear(d_model, d_model)
        self.pitch_head = nn.Linear(d_model, vocab_size)
        self.step_head = nn.Linear(d_model, 1)
        self.duration_head = nn.Linear(d_model, 1)
    
    def forward(self, pitch_idxs: torch.Tensor, durations_in: torch.Tensor,
                rhythm_features: torch.Tensor):
        """
        Args:
            pitch_idxs: (B, seq) pitch indices
            durations_in: (B, seq) duration values
            rhythm_features: (B, seq, 2) [beat_pos, beat_strength]
        Returns:
            outputs: dict of logits
        """
        # Joint embedding
        joint_emb = self.joint_pitch_duration_embed(pitch_idxs, durations_in)  # (B, seq, embed_dim)
        
        # Rhythm embedding
        rhythm_emb = self.rhythm_embed(rhythm_features)  # (B, seq, d_model-embed_dim)
        
        # Concatenate
        x = torch.cat([joint_emb, rhythm_emb], dim=-1)  # (B, seq, d_model)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        
        # Causal mask
        seq_len = x.size(1)
        mask = generate_causal_mask(seq_len, device=x.device)
        
        # Extract beat strength for attention weighting
        beat_strength = rhythm_features[..., 1]  # (B, seq)
        
        # Transformer layers
        per_layer_kv = []
        for layer in self.layers:
            x, k, v = layer(x, rhythm_weights=beat_strength, causal_mask=mask)
            per_layer_kv.append((k, v))
        
        # Output
        joint = torch.relu(self.joint_proj(x))
        pitch_logits = self.pitch_head(joint)
        step_preds = self.step_head(joint).squeeze(-1)
        dur_preds = self.duration_head(joint).squeeze(-1)
        
        return {'pitch': pitch_logits, 'step': step_preds, 'duration': dur_preds}, per_layer_kv
    
    def init_kv_cache(self, pitch_idxs: torch.Tensor, durations_in: torch.Tensor,
                      rhythm_features: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        _, per_layer_kv = self.forward(pitch_idxs, durations_in, rhythm_features)
        return per_layer_kv
    
    def generate_step_with_cache(self, last_pitch: torch.Tensor, last_duration: torch.Tensor,
                                 last_rhythm: torch.Tensor, pos: int,
                                 kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]) -> Tuple:
        """
        Incremental generation step
        Args:
            last_pitch: (B,) pitch index
            last_duration: (B,) duration value
            last_rhythm: (B, 2) [beat_pos, beat_strength]
            pos: position index
            kv_cache: cached k,v from previous steps
        """
        B = last_pitch.size(0)
        device = last_pitch.device
        
        # Joint embedding
        joint_emb = self.joint_pitch_duration_embed(last_pitch.unsqueeze(1), last_duration.unsqueeze(1))
        joint_emb = joint_emb.squeeze(1)  # (B, embed_dim)
        
        # Rhythm embedding
        rhythm_emb = self.rhythm_embed(last_rhythm)  # (B, d_model-embed_dim)
        
        # Concatenate
        x = torch.cat([joint_emb, rhythm_emb], dim=-1)  # (B, d_model)
        x = self.input_proj(x).unsqueeze(1)  # (B, 1, d_model)
        
        # Positional encoding
        pe_slice = self.pos_enc.pe[:, pos:pos+1, :]
        x = x + pe_slice.to(device)
        
        # Extract beat strength
        beat_strength = last_rhythm[:, 1]  # (B,)
        
        new_kv = []
        for i, layer in enumerate(self.layers):
            cache_k = None
            cache_v = None
            if kv_cache is not None:
                cache_k, cache_v = kv_cache[i]
            
            x, k_all, v_all = layer.forward_step(x, beat_strength.unsqueeze(1), cache_k, cache_v)
            new_kv.append((k_all, v_all))
        
        joint = torch.relu(self.joint_proj(x))
        pitch_logits = self.pitch_head(joint).squeeze(1)  # (B, vocab)
        step_pred = self.step_head(joint).squeeze(-1).squeeze(-1)  # (B,)
        dur_pred = self.duration_head(joint).squeeze(-1).squeeze(-1)  # (B,)
        
        return pitch_logits, step_pred, dur_pred, new_kv


# ============== Dataset ==============
class ImprovedMidiSequenceDataset(Dataset):
    def __init__(self, midi_inputs: List[List[float]], normalize: bool = True):
        super().__init__()
        self.midi_inputs = midi_inputs
        self.window = seq_length + 1
        self.sequences = self._make_sequences()
        self.normalize = normalize
        
        # Compute statistics
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
        beat_pos_in = torch.tensor([x[3] for x in inputs], dtype=torch.float32)
        beat_strength_in = torch.tensor([x[4] for x in inputs], dtype=torch.float32)
        
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
        
        rhythm_features = torch.stack([beat_pos_in, beat_strength_in], dim=-1)  # (seq, 2)
        
        return (pitch_inputs, durs_in, rhythm_features), {
            'pitch': pitch_targets,
            'step': step_targets_norm,
            'duration': dur_targets_norm
        }


# ============== Helper Functions ==============
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


def generate_causal_mask(sz: int, device: Optional[torch.device] = None) -> torch.Tensor:
    mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    if device is not None:
        mask = mask.to(device)
    return mask


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


# ============== Training ==============
def train(epochs: int = 50,
          batch_size: int = 32,
          lr: float = 0.001,
          device: Optional[str] = None,
          **model_kwargs):
    device = torch.device(device) if device else (
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    )
    
    print("Reading MIDI files with rhythm analysis...")
    midi_inputs, rhythm_analyzer = read_midi_notes(rhythm_grid=16)
    
    if len(midi_inputs) < seq_length + 1:
        print(f'Not enough notes to train. Got {len(midi_inputs)}, need {seq_length + 1}.')
        return
    
    print(f"Loaded {len(midi_inputs)} notes")
    
    dataset = ImprovedMidiSequenceDataset(midi_inputs, normalize=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    model = ImprovedAutoregressiveTransformer(
        vocab_size=vocab_size,
        embed_dim=model_kwargs.get('embed_dim', 64),
        d_model=model_kwargs.get('d_model', 128),
        nhead=model_kwargs.get('nhead', 8),
        num_layers=model_kwargs.get('num_layers', 4),
        dim_feedforward=model_kwargs.get('dim_feedforward', 256),
        dropout=model_kwargs.get('dropout', 0.1),
        duration_buckets=model_kwargs.get('duration_buckets', 16)
    ).to(device)
    
    ce_loss = nn.CrossEntropyLoss()
    step_mean, step_std = dataset.step_mean, dataset.step_std
    duration_mean, duration_std = dataset.duration_mean, dataset.duration_std
    
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
        
        for (pitch_inputs, durs_in, rhythm_features), batch_labels in loader:
            pitch_inputs = pitch_inputs.to(device)
            durs_in = durs_in.to(device)
            rhythm_features = rhythm_features.to(device)
            pitch_labels = batch_labels['pitch'].to(device)
            step_labels = batch_labels['step'].to(device)
            duration_labels = batch_labels['duration'].to(device)
            
            outputs, _ = model(pitch_inputs, durs_in, rhythm_features)
            pitch_logits = outputs['pitch']
            step_preds = outputs['step']
            dur_preds = outputs['duration']
            
            B, S, C = pitch_logits.shape
            loss_pitch = ce_loss(pitch_logits.view(B * S, C), pitch_labels.view(B * S))
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
                    'model_kwargs': model_kwargs,
                }
            }, checkpoint_path)
            print(f'  Saved best checkpoint (loss {epoch_loss:.6f})')
    
    print('Training finished.')


# ============== Prediction/Generation ==============
def predict_midi(num_predictions: int = 600, device: Optional[str] = None,
                 temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0, **model_kwargs):
    device = torch.device(device) if device else (
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    )
    
    model = ImprovedAutoregressiveTransformer(
        vocab_size=vocab_size,
        embed_dim=model_kwargs.get('embed_dim', 64),
        d_model=model_kwargs.get('d_model', 128),
        nhead=model_kwargs.get('nhead', 8),
        num_layers=model_kwargs.get('num_layers', 4),
        dim_feedforward=model_kwargs.get('dim_feedforward', 256),
        dropout=model_kwargs.get('dropout', 0.1),
        duration_buckets=model_kwargs.get('duration_buckets', 16)
    ).to(device)
    
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        stats = ckpt.get('dataset_stats', None)
        if stats:
            step_mean, step_std = stats['step_mean'], stats['step_std']
            duration_mean, duration_std = stats['duration_mean'], stats['duration_std']
        else:
            midi_inputs, _ = read_midi_notes()
            tmp = ImprovedMidiSequenceDataset(midi_inputs, normalize=True)
            step_mean, step_std = tmp.step_mean, tmp.step_std
            duration_mean, duration_std = tmp.duration_mean, tmp.duration_std
        print('Loaded checkpoint:', checkpoint_path)
    else:
        print('No checkpoint found, using random initialization.')
        midi_inputs, _ = read_midi_notes()
        tmp = ImprovedMidiSequenceDataset(midi_inputs, normalize=True)
        step_mean, step_std = tmp.step_mean, tmp.step_std
        duration_mean, duration_std = tmp.duration_mean, tmp.duration_std
    
    model.eval()
    midi_inputs, rhythm_analyzer = read_midi_notes()
    
    if len(midi_inputs) < seq_length:
        print(f'Not enough notes. Got {len(midi_inputs)}, need {seq_length}.')
        return
    
    # Seed sequence
    seed = random.sample(midi_inputs, seq_length)
    
    pitch_seed = torch.tensor([[int(x[0]) for x in seed]], dtype=torch.long).to(device)
    durations_seed = torch.tensor([[x[2] for x in seed]], dtype=torch.float32).to(device)
    rhythm_seed = torch.tensor([[x[3:5] for x in seed]], dtype=torch.float32).to(device)
    
    generated = list(seed)
    
    # Initialize cache
    with torch.no_grad():
        kv_cache = model.init_kv_cache(pitch_seed, durations_seed, rhythm_seed)
        
        # Generation loop
        for i in range(num_predictions):
            last = generated[-1]
            last_pitch = torch.tensor([int(last[0])], dtype=torch.long).to(device)
            last_dur = torch.tensor([last[2]], dtype=torch.float32).to(device)
            last_rhythm = torch.tensor([[last[3], last[4]]], dtype=torch.float32).to(device)
            
            pos = len(generated) - 1
            pitch_logits, step_pred, dur_pred, kv_cache = model.generate_step_with_cache(
                last_pitch, last_dur, last_rhythm, pos, kv_cache
            )
            
            pitch = sample_from_logits(pitch_logits[0], temperature=temperature, top_k=top_k, top_p=top_p)
            step = float(step_pred[0].item() * step_std + step_mean)
            duration = float(dur_pred[0].item() * duration_std + duration_mean)
            
            step = max(0.0, step)
            duration = max(0.01, duration)
            
            # Compute new rhythm features (fix: use grid parameter name, not rhythm_grid)
            beat_pos = rhythm_analyzer.extract_beat_position(step, grid=16)
            beat_strength = rhythm_analyzer.extract_beat_strength(
                rhythm_analyzer.quantize_to_beat(step, grid=16), grid=16
            )
            
            generated.append([pitch, step, duration, beat_pos, beat_strength])
    
    # Reconstruct MIDI
    prev_start = 0.0
    midi_notes = []
    for m in generated:
        pitch, step, duration = m[0], m[1], m[2]
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
    
    out_path = 'out_improved.midi'
    pm.write(out_path)
    print(f'Generated MIDI written to {out_path}')


# ============== CLI ==============
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Improved Autoregressive Transformer for Music Generation')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Generate MIDI')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num-predictions', type=int, default=600)
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--dim-feedforward', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--duration-buckets', type=int, default=16)
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-k', type=int, default=0)
    parser.add_argument('--top-p', type=float, default=0.0)
    
    args = parser.parse_args()
    
    model_kwargs = dict(
        embed_dim=args.embed_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        duration_buckets=args.duration_buckets
    )
    
    if args.train:
        train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
              device=args.device or None, **model_kwargs)
    
    if args.predict:
        predict_midi(num_predictions=args.num_predictions, device=args.device or None,
                     temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, **model_kwargs)
    
    if not args.train and not args.predict:
        predict_midi(**model_kwargs)
