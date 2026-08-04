# Improved PyTorch version of midi_main.py
# Improvements added:
# - pitch embedding instead of pitch/vocab normalization (configurable)
# - standardization (mean/std) for step and duration with proper un-normalization at generation
# - multi-layer LSTM, dropout and layer normalization
# - gradient clipping and LR scheduler (ReduceLROnPlateau)
# - --device argument, other hyperparameters configurable via CLI

import os
import random
from typing import List, Tuple, Optional

import pretty_midi
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Default hyperparameters
seq_length = 24
vocab_size = 128
checkpoint_path = 'model/model.pth'


# ---------------- Data processing ----------------
def read_midi_notes() -> List[List[float]]:
    """Read MIDI files from datasets/ and return list of [pitch, step, duration].
    """
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
    """Creates sliding window sequences of length seq_length+1 and returns
    inputs: (pitch_idxs (seq,), cont_inputs (seq,2)) and labels: pitch, step, duration
    Also computes step/duration mean/std for normalization.
    """

    def __init__(self, midi_inputs: List[List[float]], normalize: bool = True):
        super().__init__()
        self.midi_inputs = midi_inputs
        self.cut_seq_length = seq_length + 1
        self.sequences = self._make_sequences()
        self.normalize = normalize

        # compute stats for step and duration
        steps = [item[1] for item in midi_inputs]
        durations = [item[2] for item in midi_inputs]
        # fallback to small epsilon for std
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

        # inputs: list of [pitch, step, duration]
        pitch_idxs = torch.tensor([int(item[0]) for item in inputs], dtype=torch.long)  # (seq_length,)
        steps = torch.tensor([item[1] for item in inputs], dtype=torch.float32)
        durations = torch.tensor([item[2] for item in inputs], dtype=torch.float32)

        if self.normalize:
            steps = (steps - self.step_mean) / self.step_std
            durations = (durations - self.duration_mean) / self.duration_std

        cont_inputs = torch.stack([steps, durations], dim=-1)  # (seq_length, 2)

        # labels
        label_pitch = torch.tensor(int(target[0]), dtype=torch.long)
        label_step = torch.tensor(float(target[1]), dtype=torch.float32)
        label_duration = torch.tensor(float(target[2]), dtype=torch.float32)

        return (pitch_idxs, cont_inputs), {'pitch': label_pitch, 'step': label_step, 'duration': label_duration}


# ---------------- Model ----------------
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
        # pitch_idxs: (B, seq) or None if not using embedding
        # cont_inputs: (B, seq, 2)
        batch_size, seq_len, _ = cont_inputs.shape
        if self.use_embedding:
            assert pitch_idxs is not None
            emb = self.pitch_embedding(pitch_idxs)  # (B, seq, embed_dim)
            x = torch.cat([emb, cont_inputs], dim=-1)  # (B, seq, embed+2)
        else:
            # use raw pitch as a float feature concatenated
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
        # y_pred_norm is in normalized space (if normalization used)
        # y_true is raw (denormalized) time in seconds
        # compute mse in normalized space for stable gradients
        # but compute pressure on denormalized prediction
        # MSE: compare y_pred_norm to normalized y_true
        y_true_norm = (y_true - self.mean) / self.std
        mse_loss = self.mse(y_pred_norm, y_true_norm)
        # denormalize prediction
        y_pred_denorm = y_pred_norm * self.std + self.mean
        negative_part = torch.relu(-y_pred_denorm)
        positive_pressure = self.pressure * torch.mean(negative_part)
        return mse_loss + positive_pressure


# ---------------- Training & Checkpoint ----------------

def train(epochs: int = 50,
          batch_size: int = 64,
          lr: float = 0.005,
          device: Optional[str] = None,
          embed_dim: int = 32,
          use_embedding: bool = True,
          hidden_size: int = 256,
          num_layers: int = 2):
    device = torch.device(device) if device else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    midi_inputs = read_midi_notes()
    if len(midi_inputs) < seq_length + 1:
        print('Not enough notes to train.')
        return
    dataset = MidiSequenceDataset(midi_inputs, normalize=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = MusicLSTMImproved(vocab_size=vocab_size, embed_dim=embed_dim, use_embedding=use_embedding,
                              hidden_size=hidden_size, num_layers=num_layers).to(device)
    ce_loss = nn.CrossEntropyLoss()
    # For step and duration losses, provide dataset stats
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
            pitch_idxs = pitch_idxs.to(device)  # (B, seq)
            cont_inputs = cont_inputs.to(device)  # (B, seq, 2)
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
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        epoch_loss = running_loss / max(1, n_batches)
        print(f'Epoch {epoch}/{epochs} - loss: {epoch_loss:.6f}')
        scheduler.step(epoch_loss)

        # save best
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
                }
            }, checkpoint_path)
            print(f'  Saved best checkpoint (loss {epoch_loss:.6f}) to {checkpoint_path}')

    print('Training finished.')


# ---------------- Prediction ----------------

def predict_midi(num_predictions: int = 600, device: Optional[str] = None, use_embedding: bool = True, embed_dim: int = 32):
    device = torch.device(device) if device else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = MusicLSTMImproved(vocab_size=vocab_size, embed_dim=embed_dim, use_embedding=use_embedding).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        # load dataset stats if available
        stats = ckpt.get('dataset_stats', None)
        if stats:
            step_mean = stats['step_mean']
            step_std = stats['step_std']
            duration_mean = stats['duration_mean']
            duration_std = stats['duration_std']
        else:
            # fallback compute from data
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
        pitch_idxs = torch.tensor([[int(item[0]) for item in n_notes]], dtype=torch.long).to(device)  # (1, seq)
        steps = torch.tensor([[item[1] for item in n_notes]], dtype=torch.float32)
        durations = torch.tensor([[item[2] for item in n_notes]], dtype=torch.float32)
        # normalize
        steps_norm = (steps - step_mean) / step_std
        durations_norm = (durations - duration_mean) / duration_std
        cont_in = torch.stack([steps_norm.squeeze(0), durations_norm.squeeze(0)], dim=-1).unsqueeze(0)  # (1, seq, 2)
        cont_in = cont_in.to(device)
        with torch.no_grad():
            outputs = model(pitch_idxs, cont_in)
            pitch_logits = outputs['pitch'][0]
            pitch_dist = torch.distributions.Categorical(logits=pitch_logits)
            pitch = int(pitch_dist.sample().item())
            step_pred_norm = outputs['step'][0].item()
            dur_pred_norm = outputs['duration'][0].item()
            # denormalize
            step = step_pred_norm * step_std + step_mean
            duration = dur_pred_norm * duration_std + duration_mean
            # enforce non-negative durations/steps
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

    parser = argparse.ArgumentParser(description='Improved PyTorch version of AI music example')
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--predict', action='store_true', help='Run prediction (generate MIDI)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--num-predictions', type=int, default=600)
    parser.add_argument('--embed-dim', type=int, default=32)
    parser.add_argument('--no-embed', action='store_true', help='Do not use pitch embedding')
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--device', type=str, default='', help='torch device string, e.g. cpu or cuda:0')
    args = parser.parse_args()

    use_embedding = not args.no_embed

    if args.train:
        train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device or None,
              embed_dim=args.embed_dim, use_embedding=use_embedding, hidden_size=args.hidden_size, num_layers=args.num_layers)
    if args.predict:
        predict_midi(num_predictions=args.num_predictions, device=args.device or None, use_embedding=use_embedding, embed_dim=args.embed_dim)
    if (not args.train) and (not args.predict):
        # default: run prediction
        predict_midi()
