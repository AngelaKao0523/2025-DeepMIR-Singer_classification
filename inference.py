import json
from pathlib import Path
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
##from google.colab import files
from torch import amp
import torch.nn as nn
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
    def __init__(self, n_channels=128, sr=16000, n_fft=1024, hop_length=320, f_min=20.0, f_max=7900.0, n_mels=96, n_class=20):
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
        self.layer7 = Conv_2d(n_channels*2, n_channels*4, pooling=(1,2))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)

        # Dense
        self.dense1 = nn.Linear(n_channels*8, n_channels*4)
        self.bn1d   = nn.BatchNorm1d(n_channels*4)
        self.relu   = nn.ReLU(inplace=True)
        self.dropout= nn.Dropout(0.5)
        self.dense2 = nn.Linear(n_channels*4, n_class)

    def forward(self, x):
        x = self.spec(x)
        x = self.to_db(x)
        m = x.mean(dim=(1,2), keepdim=True)
        s = x.std(dim=(1,2), keepdim=True).clamp_min(1e-5)
        x = (x - m) / s
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
        gap = self.gap(x).squeeze(-1)
        gmp = self.gmp(x).squeeze(-1)
        x = torch.cat([gap, gmp], dim=1)

        # Dense
        x = self.dense1(x)
        x = self.bn1d(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x

TEST_DIR   = Path("/content/drive/MyDrive/2025 deepmir/hw1/artist20/test")
WIN_SEC    = 8.0
HOP_SEC    = 4.0
CHUNK_BATCH= 16
SR = 16000
STUDENT_ID = "B11505063"
SINGER_GENRES=['aerosmith', 'beatles', 'creedence_clearwater_revival', 'cure', 'dave_matthews_band', 'depeche_mode',
            'fleetwood_mac', 'garth_brooks', 'green_day', 'led_zeppelin', 'madonna', 'metallica', 'prince', 'queen', 'radiohead',
            'roxette', 'steely_dan', 'suzanne_vega', 'tori_amos', 'u2']

CHECKPOINT_PATH = "/content/drive/MyDrive/2025 deepmir/hw1/model_checkpoint.pth"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = ShortChunkCNN().to(device)
state = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(state)
model.eval()
print(f"[OK] Loaded checkpoint from: {CHECKPOINT_PATH}")

def load_wave_simple(path):
    wav, _ = torchaudio.load(path)  # (1, N)
    return wav.squeeze(0)
def make_crops(wav, sr=16000, win_sec=8.0, hop_sec=4.0):
  win = int(win_sec * sr)
  hop = int(hop_sec * sr)
  N   = wav.numel() ## total number
  if N <= win:
        out = torch.zeros(win, dtype=wav.dtype)
        out[:N] = wav
        return [out]
  crops, start = [], 0
  while start + win <= N:
        crops.append(wav[start:start+win])
        start += hop
  if start < N:
        crops.append(wav[-win:])
  return crops

@torch.no_grad()
def predict(model, path, genres, sr=16000, win_sec=8.0, hop_sec=4.0, k=3, device=device):
  model.eval()
  wav = load_wave_simple(path)
  crops =  make_crops(wav, sr=sr, win_sec=win_sec, hop_sec=hop_sec) ## crops is a list, each element is a window size

  use_amp   = (device.type == 'cuda')
  prob_sum = None

  with amp.autocast(device_type='cuda', enabled=use_amp):
    for i in range(0, len(crops), CHUNK_BATCH):
      batch = torch.stack(crops[i:i+CHUNK_BATCH], dim=0).to(device)
      logit = model(batch)
      probs = F.softmax(logit, dim=1)
      s = probs.sum(dim=0)
      prob_sum = s if prob_sum is None else prob_sum + s
    prob_avg = prob_sum / len(crops)
    top_k_indice = torch.topk(prob_avg, k=k, dim=0).indices.tolist()
    names = [genres[i] for i in top_k_indice]
    return names, prob_avg.cpu()
  
test_files = sorted(TEST_DIR.glob("*.mp3"), key=lambda x: int(x.stem))
result = {}
for p in test_files:
    names, _ = predict(model, p, SINGER_GENRES, sr=SR, win_sec=WIN_SEC, hop_sec=HOP_SEC, k=3, device=device)
    result[p.stem] = names

## output
out_json = f"{STUDENT_ID}.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print(f"[OK] Saved predictions to: {out_json}  ({len(result)} items)")
print({k: result[k] for k in list(result.keys())[:5]})
files.download(out_json)