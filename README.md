# Artist20 Inference Guide
**Student ID:** B11505063
**Task:** Music Artist Classification (Inference Only)
---
## 1. Overview
This project performs inference for a pretrained **ShortChunkCNN** model that classifies audio
clips into one of 20 artists from the Artist20 dataset.
Each input `.mp3` file is split into 8-second chunks, passed through the model, and the top-3 most
probable artists are saved to a JSON file.
- **Input:** `.mp3` files in the test folder
- **Output:** `B11505063.json` — top-3 artist predictions per test file
---
## 2. Environment Setup
**Python version:** 3.9 or above (recommended 3.10)
### Install Dependencies
To set up the environment, run:
```
pip install -r requirements.txt
```
### requirements.txt
```
---
## 3. Directory Structure
```
/
■■■ inference.py
■■■ README.md
■■■ requirements.txt
■■■ models/
■ ■■■ model_checkpoint.pth
```
---
## 4. Model Checkpoint
- **Download link:**
- **Expected path:** `models/model_checkpoint.pth`
■■ **Important:**
Before running inference, please check and **manually update the `CHECKPOINT_PATH`** in
your `inference.py` file to the correct model location on your environment.
For example:
```
CHECKPOINT_PATH = "models/model_checkpoint.pth" # local
```
Make sure the file name and path match exactly.
## 5. Quick Start (Guide: How to Perform Inference on Your Model)
To perform inference on the pretrained model, run the following command:
---
```
```
python inference.py
This will:
1. Load the pretrained ShortChunkCNN model from `models/model_checkpoint.pth`
2. Read all `.mp3` files under `data/test/`
3. Generate predictions for each file
4. Save results to `B11505063.json` in the current directory
5. (If in Colab) Automatically download the JSON file
Example console output:
```
[OK] Loaded checkpoint from: models/model_checkpoint.pth
[OK] Saved predictions to: B11505063.json (20 items)
{'1': ['queen', 'u2', 'radiohead'], '2': ['madonna', 'prince', 'aerosmith'], ...}
```
---
## 6. Parameters (Defined in Script)
| Variable | Description | Default |
|-----------|--------------|----------|
| TEST_DIR | Path to test audio folder | `/content/drive/MyDrive/2025 deepmir/hw1/artist20/test` |
| CHECKPOINT_PATH | Path to model checkpoint (must be updated manually) |
`/content/drive/MyDrive/2025 deepmir/hw1/model_checkpoint.pth` |
| WIN_SEC | Window size (seconds) | 8.0 |
| HOP_SEC | Hop length (seconds) | 4.0 |
| CHUNK_BATCH | Batch size | 16 |
| SR | Sampling rate | 16000 |
| STUDENT_ID | Output file name ID | "B11505063" |
---
## 7. Input and Output
### Input
All `.mp3` files inside the test folder (e.g., `1.mp3`, `2.mp3`, ...)
### Output
A JSON file named `B11505063.json`, e.g.:
```
{
"1": ["queen", "u2", "radiohead"],
"2": ["madonna", "prince", "aerosmith"],
"3": ["metallica", "led_zeppelin", "green_day"]
}
```
---
## 8. Inference Flow
1. Load model checkpoint to CUDA or CPU
2. Read `.mp3` file and resample to 16 kHz
3. Split into 8-second overlapping segments (4-second hop)
4. Perform batched inference (mixed precision if CUDA available)
5. Average probabilities across crops
6. Select top-3 artists
7. Save all predictions to `B11505063.json`
8. Auto-download the JSON in Colab
