import json
from pathlib import Path
import librosa
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score, confusion_matrix, classification_report
import joblib


SR = 16000 ## Sample Rate
DATA_ROOT = Path('artist20')

def find_singer(p: Path):
    if p.parent and p.parent.parent:
        return p.parent.parent.name

def load_mp3(p : Path):
    y,_ = librosa.load(p, sr = SR, mono = True)
    return y

def pool(M):
    return np.concatenate([M.mean(axis=1), M.std(axis=1), M.max(axis=1)], axis=0)

def extract_features(y, sr=SR, n_mfcc=20):
    mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=n_mfcc)
    dmfcc = librosa.feature.delta(mfcc)
    ddmfcc = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(y, sr=sr)
    centriod = librosa.feature.spectral_centroid(y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    tempo, _ = librosa.beat.beat_track(y, sr=sr)

    feats =[
        pool(mfcc), 
        pool(dmfcc), 
        pool(ddmfcc), 
        pool(chroma), 
        pool(centriod), 
        pool(bandwidth), 
        pool(rolloff), 
        pool(zcr),
        np.array([tempo], dtype=np.float32)   
    ]

    result = np.concatenate(feats).astype(np.float32)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result

def get_score(pipe, X):
    clf = pipe.named_steps['clf']
    if hasattr(clf, 'predict_proba'):
        return pipe.predict_proba(X)
    elif hasattr(clf, 'decision_function'):
        s = pipe.decision_function(X)
        if s.ndim == 1:
            s = np.vstack([-s, s]).T
        return s

def evaluate_model(X_tr, y_tr, X_val, y_val):
    le = LabelEncoder()
    y_tr = le.fit_transform(y_tr)
    y_val = le.transform(y_val)

    pipe=Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(C = 2.0, gamma='scale', kernel='rbf', probability=True))
    ])

    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_val)
    top_1 = accuracy_score(y_val, y_pred)
    scores = get_score(pipe, X_val)
    k3 = min(3, scores.shape[1])
    top_3 = top_k_accuracy_score(y_val, scores, k=k3)

    print(f"Top-1 accuracy: {top_1:.4f}")
    print(f"Top-{k3} accuracy: {top_3:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_val, y_pred))

    joblib.dump(pipe, 'model.pkl')
    joblib.dump(le, "label_encoder.joblib")
    print("Model saved to model.pkl")


def main():     
    with open('artist20/train.json', 'r', encoding = 'utf-8') as f:
        train_list = json.load(f)
        train_path = [DATA_ROOT / p for p in train_list]
        train_labels = [find_singer(p) for p in train_path]

    with open('artist20/val.json', 'r', encoding = 'utf-8') as f:
        val_list = json.load(f)
        val_path = [DATA_ROOT /p for p in val_list]
        val_labels = [find_singer(p) for p in val_path]

    X_tr = []
    for i, p in enumerate(train_path):
        y = load_mp3(p)
        feat = extract_features(y)
        X_tr.append(feat)
        if (i + 1) % 20 == 0:
            print(f"[train] {i+1}/{len(train_path)}: {p.name}")

    X_val = []
    for i, p in enumerate(val_path):
        y = load_mp3(p)
        feat = extract_features(y)
        X_val.append(feat)
        if (i + 1) % 20 == 0:
            print(f"[val] {i+1}/{len(val_path)}: {p.name}")

    X_tr = np.vstack(X_tr).astype(np.float32)
    X_val = np.vstack(X_val).astype(np.float32)
    
    y_tr = np.array(train_labels, dtype=object)   
    y_val = np.array(val_labels,   dtype=object)

    evaluate_model(X_tr, y_tr, X_val, y_val)


        
    