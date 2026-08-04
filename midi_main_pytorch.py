# PyTorch version of midi_main.py
# Converted from TensorFlow/Keras implementation in the repository

import os
import random
from typing import List, Tuple

import pretty_midi
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Hyperparameters kept from original
seq_length = 24
vocab_size = 128
checkpoint_path = 'model/model.pth'


# ---------------- Data processing ----------------
def read_midi_notes() -> List[List[float]]:
    """Read MIDI files from datasets/ and return list of [pitch, step, duration].
    Mirrors the original behavior in midi_main.py.
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
    (inputs, labels) where inputs shape = (seq_length, 3) and labels is a dict.
    """

    def __init__(self, midi_inputs: List[List[float]]):
        super().__init__()
        self.midi_inputs = midi_inputs
        self.cut_seq_length = seq_length + 1
        self.sequences = self._make_sequences()

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
        # normalize pitch by vocab_size as original
        inputs_x = [[item[0] / vocab_size, item[1], item[2]] for item in inputs]
        target = seq[-1]
        # target pitch should be int
        label_pitch = int(target[0])
        label_step = float(target[1])
        label_duration = float(target[2])
        # return tensors
        inputs_x = torch.tensor(inputs_x, dtype=torch.float32)  # (seq_length, 3)
        labels = {
            'pitch': torch.tensor(label_pitch, dtype=torch.long),
            'step': torch.tensor(label_step, dtype=torch.float32),
            'duration': torch.tensor(label_duration, dtype=torch.float32),
        }
        return inputs_x, labels


# ---------------- Model ----------------
class MusicLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=1, vocab_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.pitch_head = nn.Linear(hidden_size, vocab_size)
        self.step_head = nn.Linear(hidden_size, 1)
        self.duration_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (hn, cn) = self.lstm(x)
        # take the last time-step output
        last = out[:, -1, :]
        pitch_logits = self.pitch_head(last)
        step = self.step_head(last).squeeze(-1)
        duration = self.duration_head(last).squeeze(-1)
        return {
            'pitch': pitch_logits,  # logits
            'step': step,
            'duration': duration,
        }


# ---------------- Losses ----------------
class MSEWithPositivePressure(nn.Module):
    def __init__(self, pressure=10.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.pressure = pressure

    def forward(self, y_pred, y_true):
        # y_pred, y_true: (batch,)
        mse_loss = self.mse(y_pred, y_true)
        # positive pressure for negative predictions: pressure * max(-y_pred, 0)
        negative_part = torch.relu(-y_pred)
        positive_pressure = self.pressure * torch.mean(negative_part)
        return mse_loss + positive_pressure


# ---------------- Training & Checkpoint ----------------

def train(epochs=50, batch_size=64, lr=0.01, device=None):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    midi_inputs = read_midi_notes()
    if len(midi_inputs) < seq_length + 1:
        print('Not enough notes to train.')
        return
    dataset = MidiSequenceDataset(midi_inputs)
    # shuffle similar to TF shuffle buffer by shuffling indices in DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = MusicLSTM(vocab_size=vocab_size).to(device)
    ce_loss = nn.CrossEntropyLoss()
    mse_pressure = MSEWithPositivePressure()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for batch_x, batch_labels in loader:
            batch_x = batch_x.to(device)  # (B, seq_length, 3)
            # labels
            pitch_labels = batch_labels['pitch'].to(device)  # (B,)
            step_labels = batch_labels['step'].to(device)
            duration_labels = batch_labels['duration'].to(device)

            outputs = model(batch_x)
            pitch_logits = outputs['pitch']  # (B, vocab_size)
            step_preds = outputs['step']  # (B,)
            duration_preds = outputs['duration']  # (B,)

            loss_pitch = ce_loss(pitch_logits, pitch_labels)
            loss_step = mse_pressure(step_preds, step_labels)
            loss_duration = mse_pressure(duration_preds, duration_labels)

            loss = 0.05 * loss_pitch + 1.0 * loss_step + 1.0 * loss_duration

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        epoch_loss = running_loss / max(1, n_batches)
        print(f'Epoch {epoch}/{epochs} - loss: {epoch_loss:.6f}')

        # save best
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': epoch_loss,
            }, checkpoint_path)
            print(f'  Saved best checkpoint (loss {epoch_loss:.6f}) to {checkpoint_path}')

    print('Training finished.')


# ---------------- Prediction ----------------

def predict_midi(num_predictions=600, device=None):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = MusicLSTM(vocab_size=vocab_size).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print('Loaded checkpoint:', checkpoint_path)
    else:
        print('No checkpoint found, running with randomly initialized model.')

    model.eval()
    midi_inputs = read_midi_notes()
    if len(midi_inputs) < seq_length:
        print('Not enough notes to predict.')
        return

    sample_notes = random.sample(midi_inputs, seq_length)

    generated = list(sample_notes)  # list of [pitch, step, duration]

    for i in range(num_predictions):
        n_notes = generated[-seq_length:]
        notes_in = [[item[0] / vocab_size, item[1], item[2]] for item in n_notes]
        x = torch.tensor([notes_in], dtype=torch.float32).to(device)  # (1, seq_length, 3)
        with torch.no_grad():
            outputs = model(x)
            pitch_logits = outputs['pitch'][0]  # (vocab_size,)
            # sample pitch from logits
            pitch_dist = torch.distributions.Categorical(logits=pitch_logits)
            pitch = int(pitch_dist.sample().item())
            step = float(outputs['step'][0].item())
            duration = float(outputs['duration'][0].item())
        generated.append([pitch, step, duration])

    print('Generated sequence length:', len(generated))

    # reconstruct midi notes (start/end times)
    prev_start = 0.0
    midi_notes = []
    for m in generated:
        pitch, step, duration = m
        start = prev_start + step
        end = start + duration
        prev_start = start
        midi_notes.append([int(pitch), float(start), float(end)])

    # write to out.midi
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

    parser = argparse.ArgumentParser(description='PyTorch version of AI music example')
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--predict', action='store_true', help='Run prediction (generate MIDI)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num-predictions', type=int, default=600)
    args = parser.parse_args()

    if args.train:
        train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    if args.predict:
        predict_midi(num_predictions=args.num_predictions)
    if (not args.train) and (not args.predict):
        # default: run prediction to match original script behavior
        predict_midi()
