from google.colab import drive
drive.mount('/content/drive')
%cd "/content/drive/MyDrive/2025 deepmir/hw1"
import os
import json
import random
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils import data
SINGER_GENRES=['aerosmith', 'beatles', 'creedence_clearwater_revival', 'cure', 'dave_matthews_band', 'depeche_mode',
            'fleetwood_mac', 'garth_brooks', 'green_day', 'led_zeppelin', 'madonna', 'metallica', 'prince', 'queen', 'radiohead',
            'roxette', 'steely_dan', 'suzanne_vega', 'tori_amos', 'u2']
DATA_ROOT = Path('artist20')
SR = 16000
def augmentations(wav):
        #if torch.rand(1).item() < 0.5:
           # wav = -wav # reverse the wav
        gain_db = float(torch.empty(1).uniform_(-6,6))
        wav = wav * (10 ** (gain_db / 20))

        if torch.rand(1).item() < 0.5:
          shift = int(torch.empty(1).uniform_(-0.2, 0.2).item() * SR)  # ±0.2s
          wav = torch.roll(wav, shifts=shift, dims=0)

        if torch.rand(1).item() < 0.5:
            rms = wav.pow(2).mean().sqrt().clamp_min(1e-8)
            snr_db = float(torch.empty(1).uniform_(15, 30))
            noise_rms = rms / (10 ** (snr_db / 20))
            noise = torch.randn_like(wav) * noise_rms
            wav = wav + noise
        return  wav.clamp(-1, 1)

class Artist20Dataset(data.Dataset):
    def __init__(self, json_file: str, split, num_samples, num_chunks,  is_augmentation:bool=True):
        with open(json_file, "r", encoding="utf-8") as f:
            rel_path = json.load(f)
        self.song_list = [DATA_ROOT / p for p in rel_path]

        self.genres = SINGER_GENRES
        self.genre_to_list = {}
        for i, c in enumerate(self.genres):
            self.genre_to_list[c] = i

        ## self.sr = sr
        ## self.train = train
        self.split = split
        self.num_samples = num_samples
        self.num_chunks = num_chunks
        self.is_augmentation = is_augmentation
        ## self.mel = T.MelSpectrogram(sample_rate=self.sr, n_mels=n_mels,  hop_length=hop, center=True)
        ## self.to_db = T.AmplitudeToDB()

    def load_wave(self, path:str):
        wav, _ = torchaudio.load(path/train_val)
        return wav.squeeze(0)

    def __len__(self):
        return len(self.song_list)

    def crop(self, y:torch.Tensor):
        N = y.numel()
        if N < self.num_samples:
            y = torch.nn.functional.pad(y, (0, self.num_samples - N))
            return y
        if self.split=="train":
            start = random.randint(0, N - self.num_samples)
        else:
            start = (N - self.num_samples) // 2
        return y[start:start+self.num_samples]

    def find_artist(self, p:str):
        for part in Path(p).parts:
            if part in SINGER_GENRES:
                return part
        raise ValueError(f"Cannot find artist in path: {p}")

    def __getitem__(self, idx):
        path = self.song_list[idx]
        # get genre label
        genre_name = self.find_artist(path)
        genre_index = self.genre_to_list[genre_name]

        # adjust audio length
        wav = self.load_wave(path)
        wav = self.crop(wav)

        # data augmentation
        if self.is_augmentation and self.split == "train":
            wav = augmentations(wav)

        return wav, genre_index

def get_loader(data_path=None,
               split='train',
               num_samples=16000*5,
               num_chunks=1,
               batch_size=16,
               num_workers=0,
               is_augmentation=False):
    is_shuffle = True if split=='train' else False
    dataset = Artist20Dataset(json_file=data_path,
                              split=split,
                              num_samples=num_samples,
                              num_chunks=num_chunks,
                              is_augmentation=is_augmentation)

    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,                  # move data to gpu
    )
    return data_loader
import torch
import torch.nn as nn
import torchaudio
import numpy as np
class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, shape, stride, padding=shape//2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        if isinstance(pooling, int):
          pooling = (pooling, pooling)
        self.mp = nn.MaxPool2d(pooling)
    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out

class  ShortChunkCNN(nn.Module):
    def __init__(self, n_channels=128, sr=16000, n_fft=512, hop_length=512, f_min=0.0, f_max=8000.0, n_mels=128, n_class=20):
        super().__init__()
        # sepctrogram
        self.spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            center=True,
            power=2.0,)
        self.to_db   = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=16)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=32)
        self.apply_specaug = True

        # convolutional layers
        self.layer1 = Conv_2d(1,            n_channels,   pooling=(2,1))
        self.layer2 = Conv_2d(n_channels,   n_channels,   pooling=(2,1))
        self.layer3 = Conv_2d(n_channels,   n_channels*2, pooling=(2,1))
        self.layer4 = Conv_2d(n_channels*2, n_channels*2, pooling=(2,1))
        self.layer5 = Conv_2d(n_channels*2, n_channels*2, pooling=(2,1))
        self.layer6 = Conv_2d(n_channels*2, n_channels*2, pooling=(2,1))
        self.layer7 = Conv_2d(n_channels*2, n_channels*4, pooling=(2,1))

        # Dense
        self.dense1 = nn.Linear(n_channels*4, n_channels*4)
        self.bn1d   = nn.BatchNorm1d(n_channels*4)
        self.relu   = nn.ReLU(inplace=True)
        self.dropout= nn.Dropout(0.5)
        self.dense2 = nn.Linear(n_channels*4, n_class)

    def forward(self, x):
        x = self.spec(x)
        x = self.to_db(x)
        if self.training and self.apply_specaug:
            x = self.freq_mask(x)
            x = self.time_mask(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)

        # Global Max Pooling over time
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze()

        # Dense
        x = self.dense1(x)
        x = self.bn1d(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
from sklearn.metrics import confusion_matrix
BASE = Path("/content/drive/MyDrive/2025 deepmir/hw1/artist20")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_loader = get_loader(str(BASE / "train.json"), split='train',
                                 num_samples=SR*8, batch_size=16,
                                 num_workers=2, is_augmentation=True)

val_loader   = get_loader(str(BASE / "val.json"), split='val',
                                 num_samples=SR*8, batch_size=16,
                                 num_workers=2, is_augmentation=False)

model = ShortChunkCNN().to(device)
use_amp = (device.type == 'cuda')
loss_function = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scaler = GradScaler()
num_epochs = 10
from tqdm import tqdm
from torch.cuda.amp import autocast

for epoch in range(num_epochs):
    model.train()
    running = 0.0
  
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
    
    for wav, genre_index in progress_bar:
        wav = wav.to(device)
        genre_index = genre_index.to(device)
        optimizer.zero_grad()
        with autocast():
            out = model(wav)
            loss = loss_function(out, genre_index)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running += loss.item() * wav.size(0)

        progress_bar.set_postfix(loss=loss.item())
    
    train_loss = running / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}")

    train_loss = running / len(train_loader.dataset)

import contextlib
from sklearn.metrics import confusion_matrix
### validation
model.eval()
all_pred1 = []
all_true = []
top_1_num = 0
top_3_num = 0
total = 0
has_cuda = torch.cuda.is_available()
device_type = 'cuda' if has_cuda else 'cpu'
use_autocast = (use_amp and has_cuda)  

amp_ctx = torch.amp.autocast(device_type=device_type) if use_autocast else contextlib.nullcontext()

with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=use_autocast):
    for wav, genre_index in val_loader:
        wav = wav.to(device)
        genre_index = genre_index.to(device)

        out = model(wav)

        # top_1
        pred1 = out.argmax(dim=1)
        top_1_num += (pred1 == genre_index).sum().item()
        # top_3
        top_3 = out.topk(3, dim=1).indices
        top_3_num += (top_3 == genre_index.unsqueeze(1)).any(dim=1).sum().item()
        total += wav.size(0)

        all_pred1.append(pred1.cpu())
        all_true.append(genre_index.cpu())

val_acc1 = top_1_num / total
val_acc3 = top_3_num / total

y_true   = torch.cat(all_true).numpy()
y_pred1  = torch.cat(all_pred1).numpy()
cm       = confusion_matrix(y_true, y_pred1, labels=list(range(len(SINGER_GENRES))))

print(f"Epoch {epoch+1:02d} | train loss: {train_loss:.4f} | "
    f"val top1: {val_acc1:.4f} | val top3: {val_acc3:.4f}")
print(cm)
