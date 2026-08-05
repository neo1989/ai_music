# Transformer-based PyTorch version of midi_main.py
# Upgrades:
# - Pitch embedding + positional encoding
# - Transformer encoder stack (multi-head self-attention)
# - Joint pitch-duration modeling with shared representation and separate heads (multi-head outputs)
# - CLI to choose model type (transformer or lstm)
# - Keeps normalization, checkpointing, and generation logic

import os
import math
import random
from typing import List, Optional, Tuple

import pretty_midi
import torch
import torch.nn as nn
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


class MidiSequenceDataset(Dataset):
    def __init__(self, midi_inputs: List[List[float]], normalize: bool = True):
        super().__init__()
        self.midi_inputs = midi_inputs
        self.cut_seq_length = seq_length + 1
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
        for i in range(0, n - self.cut_seq_length + 1):
            window = self.midi_inputs[i:i + self.cut_seq_length]
            seqs.append(window)
        return seqs

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        inputs = seq[:-1]
        target = seq[-1]
        pitch_idxs = torch.tensor([int(item[0]) for item in inputs], dtype=torch.long)
        steps = torch.tensor([item[1] for item in inputs], dtype=torch.float32)
        durations = torch.tensor([item[2] for item in inputs], dtype=torch.float32)
        if self.normalize:
            steps = (steps - self.step_mean) / self.step_std
            durations = (durations - self.duration_mean) / self.duration_std
        cont_inputs = torch.stack([steps, durations], dim=-1)
        label_pitch = torch.tensor(int(target[0]), dtype=torch.long)
        label_step = torch.tensor(float(target[1]), dtype=torch.float32)
        label_duration = torch.tensor(float(target[2]), dtype=torch.float32)
        return (pitch_idxs, cont_inputs), {'pitch': label_pitch, 'step': label_step, 'duration': label_duration}


# ---------------- Model components ----------------
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
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TransformerMusicModel(nn.Module):
    def __init__(self,
                 vocab_size: int = 128,
                 embed_dim: int = 64,
                 nhead: int = 8,
                 d_model: int = 128,
                 num_encoder_layers: int = 4,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 cont_dim: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.pitch_embed = nn.Embedding(vocab_size, embed_dim)
        # project continuous features to d_model-sized vectors
        self.cont_proj = nn.Linear(cont_dim, d_model - embed_dim)
        self.input_proj = nn.Linear(d_model, d_model)  # optional projection
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # Shared joint representation to predict both pitch and duration/step
        self.joint_proj = nn.Linear(d_model, d_model)
        # Multi-head outputs: pitch classification + step & duration regression heads
        self.pitch_head = nn.Linear(d_model, vocab_size)
        self.step_head = nn.Linear(d_model, 1)
        self.duration_head = nn.Linear(d_model, 1)

    def forward(self, pitch_idxs: torch.Tensor, cont_inputs: torch.Tensor):
        # pitch_idxs: (B, seq), cont_inputs: (B, seq, 2)
        emb = self.pitch_embed(pitch_idxs)  # (B, seq, embed_dim)
        cont = self.cont_proj(cont_inputs)  # (B, seq, d_model - embed_dim)
        x = torch.cat([emb, cont], dim=-1)  # (B, seq, d_model)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)  # (B, seq, d_model)
        last = x[:, -1, :]
        last = self.layernorm(last)
        last = self.dropout(last)
        joint = torch.relu(self.joint_proj(last))
        pitch_logits = self.pitch_head(joint)
        step = self.step_head(joint).squeeze(-1)
        duration = self.duration_head(joint).squeeze(-1)
        return {'pitch': pitch_logits, 'step': step, 'duration': duration}


# Keep the previous LSTM model as fallback
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


# ---------------- Losses ----------------
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


# ---------------- Training & Checkpoint ----------------

def build_model(model_type: str = 'transformer', **kwargs) -> nn.Module:
    if model_type == 'transformer':
        return TransformerMusicModel(vocab_size=vocab_size,
                                     embed_dim=kwargs.get('embed_dim', 64),
                                     nhead=kwargs.get('nhead', 8),
                                     d_model=kwargs.get('d_model', 128),
                                     num_encoder_layers=kwargs.get('num_layers', 4),
                                     dim_feedforward=kwargs.get('dim_feedforward', 256),
                                     dropout=kwargs.get('dropout', 0.1))
    else:
        return MusicLSTMImproved(vocab_size=vocab_size,
                                 embed_dim=kwargs.get('embed_dim', 32),
                                 use_embedding=kwargs.get('use_embedding', True),
                                 hidden_size=kwargs.get('hidden_size', 256),
                                 num_layers=kwargs.get('num_layers', 2))


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
    dataset = MidiSequenceDataset(midi_inputs, normalize=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = build_model(model_type=model_type, **model_kwargs).to(device)
    ce_loss = nn.CrossEntropyLoss()
    step_loss_fn = MSEWithPositivePressure(pressure=10.0, mean=dataset.step_mean, std=dataset.step_std)
    duration_loss_fn = MSEWithPositivePressure(pressure=10.0, mean=dataset.duration_mean, std=dataset.duration_std)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)

    best_loss = float('inf')
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for (pitch_idxs, cont_inputs), batch_labels in loader:
            pitch_idxs = pitch_idxs.to(device)
            cont_inputs = cont_inputs.to(device)
            pitch_labels = batch_labels['pitch'].to(device)
            step_labels = batch_labels['step'].to(device)
            duration_labels = batch_labels['duration'].to(device)

            outputs = model(pitch_idxs, cont_inputs)
            pitch_logits = outputs['pitch']
            step_preds_norm = outputs['step']
            duration_preds_norm = outputs['duration']

            loss_pitch = ce_loss(pitch_logits, pitch_labels)
            loss_step = step_loss_fn(step_preds_norm, step_labels)
            loss_duration = duration_loss_fn(duration_preds_norm, duration_labels)

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
                    'step_mean': dataset.step_mean,
                    'step_std': dataset.step_std,
                    'duration_mean': dataset.duration_mean,
                    'duration_std': dataset.duration_std,
                },
                'model_meta': {
                    'model_type': model_type,
                    'model_kwargs': model_kwargs,
                }
            }, checkpoint_path)
            print(f'  Saved best checkpoint (loss {epoch_loss:.6f}) to {checkpoint_path}')

    print('Training finished.')


# ---------------- Prediction ----------------

def predict_midi(num_predictions: int = 600, device: Optional[str] = None, model_type: str = 'transformer', **model_kwargs):
    device = torch.device(device) if device else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = build_model(model_type=model_type, **model_kwargs).to(device)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        stats = ckpt.get('dataset_stats', None)
        meta = ckpt.get('model_meta', None)
        if stats:
            step_mean = stats['step_mean']
            step_std = stats['step_std']
            duration_mean = stats['duration_mean']
            duration_std = stats['duration_std']
        else:
            midi_inputs = read_midi_notes()
            tmp = MidiSequenceDataset(midi_inputs, normalize=True)
            step_mean = tmp.step_mean
            step_std = tmp.step_std
            duration_mean = tmp.duration_mean
            duration_std = tmp.duration_std
        print('Loaded checkpoint:', checkpoint_path)
    else:
        print('No checkpoint found, running with randomly initialized model.')
        midi_inputs = read_midi_notes()
        tmp = MidiSequenceDataset(midi_inputs, normalize=True)
        step_mean = tmp.step_mean
        step_std = tmp.step_std
        duration_mean = tmp.duration_mean
        duration_std = tmp.duration_std

    model.eval()
    midi_inputs = read_midi_notes()
    if len(midi_inputs) < seq_length:
        print('Not enough notes to predict.')
        return

    sample_notes = random.sample(midi_inputs, seq_length)
    generated = list(sample_notes)

    for i in range(num_predictions):
        n_notes = generated[-seq_length:]
        pitch_idxs = torch.tensor([[int(item[0]) for item in n_notes]], dtype=torch.long).to(device)
        steps = torch.tensor([[item[1] for item in n_notes]], dtype=torch.float32)
        durations = torch.tensor([[item[2] for item in n_notes]], dtype=torch.float32)
        steps_norm = (steps - step_mean) / step_std
        durations_norm = (durations - duration_mean) / duration_std
        cont_in = torch.stack([steps_norm.squeeze(0), durations_norm.squeeze(0)], dim=-1).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(pitch_idxs, cont_in)
            pitch_logits = outputs['pitch'][0]
            pitch_dist = torch.distributions.Categorical(logits=pitch_logits)
            pitch = int(pitch_dist.sample().item())
            step_pred_norm = outputs['step'][0].item()
            dur_pred_norm = outputs['duration'][0].item()
            step = step_pred_norm * step_std + step_mean
            duration = dur_pred_norm * duration_std + duration_mean
            step = max(0.0, float(step))
            duration = max(0.01, float(duration))
        generated.append([pitch, step, duration])

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

    parser = argparse.ArgumentParser(description='Transformer-enhanced PyTorch AI music example')
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

    args = parser.parse_args()

    model_type = args.model_type

    if args.train:
        model_kwargs = {}
        if model_type == 'transformer':
            model_kwargs = dict(embed_dim=args.embed_dim, nhead=args.nhead, d_model=args.d_model, num_layers=args.num_layers,
                                dim_feedforward=args.dim_feedforward, dropout=args.dropout)
        else:
            model_kwargs = dict(embed_dim=args.lstm_embed_dim, use_embedding=True, hidden_size=args.hidden_size, num_layers=args.lstm_num_layers)
        train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device or None, model_type=model_type, **model_kwargs)

    if args.predict:
        model_kwargs = {}
        if model_type == 'transformer':
            model_kwargs = dict(embed_dim=args.embed_dim, nhead=args.nhead, d_model=args.d_model, num_layers=args.num_layers,
                                dim_feedforward=args.dim_feedforward, dropout=args.dropout)
        else:
            model_kwargs = dict(embed_dim=args.lstm_embed_dim, use_embedding=True, hidden_size=args.hidden_size, num_layers=args.lstm_num_layers)
        predict_midi(num_predictions=args.num_predictions, device=args.device or None, model_type=model_type, **model_kwargs)

    if (not args.train) and (not args.predict):
        # default: run prediction
        predict_midi()
