# Multi-Modal Bangla Sign Language (BdSL) Recognition System

This repository provides a complete scaffold for building a real-time offline BdSL recognition pipeline. The system captures RGB webcam videos, extracts MediaPipe Holistic landmarks, trains multi-stream neural encoders, and deploys a real-time demo with simultaneous sign (60 classes) and facial grammar (3 classes) predictions.

## Repository Structure
```
project/
├── capture/record_videos.py         # 1080p video capture tool with metadata logging
├── preprocess/                      # Landmark extraction + normalization utilities
│   ├── extract_landmarks.py
│   ├── normalize.py
│   └── build_manifest.py
├── data/                            # Placeholder for raw videos, landmarks, manifest.csv
├── models/                          # PyTorch encoders, fusion module, multi-task head
├── train/                           # Training pipelines for baselines and fusion
├── eval/                            # Evaluation, confusion matrices, ablations
└── demo/                            # Real-time webcam inference demo
```

## Installation
1. Create a Python 3.10+ virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Install GPU-enabled PyTorch following instructions from [pytorch.org](https://pytorch.org/).

## Dataset Capture Workflow
1. **Record videos**
   ```bash
   python capture/record_videos.py data/raw <word> S01 1 1 --metadata data/manifest.csv --grammar neutral
   ```
   - Resolution: 1080p @ 30 FPS
   - Filename format: `<word>__S<id>__sess<id>__rep<id>__<grammar>.mp4` (grammar in {neutral, question, negation})

2. **Build or update manifest** (if recording without `--metadata`):
   ```bash
   python preprocess/build_manifest.py data/raw data/manifest.csv
   ```

3. **Extract landmarks + normalize**
   ```bash
   python preprocess/extract_landmarks.py data/raw data/landmarks --manifest data/manifest.csv --sequence-length 48
   ```
   Each video becomes a `.npz` file with left/right hands, face, and pose arrays.

## Training
1. **Baseline (single-modality) models**
   ```bash
   python train/train_baselines.py data/manifest.csv data/landmarks hands --train-signers S01 S02 S03 --val-signers S04 --test-signers S05
   ```
2. **Full fusion model**
   ```bash
   python train/train_fusion.py data/manifest.csv data/landmarks --train-signers S01 S02 S03 --val-signers S04 --test-signers S05 --epochs 40 --batch-size 96
   ```
   - Loss: `CE(sign) + 0.5 * CE(grammar)`
   - Optimizer: AdamW (lr=3e-4) + cosine scheduler

## Evaluation & Analysis
1. **Quantitative evaluation**
   ```bash
   python eval/evaluate.py data/manifest.csv data/landmarks fusion_model.pt --train-signers S01 S02 S03 --val-signers S04 --test-signers S05
   ```
2. **Confusion matrices**
   ```bash
   python eval/confusion_matrix.py data/manifest.csv data/landmarks fusion_model.pt --train-signers S01 S02 S03 --val-signers S04 --test-signers S05 --output cm.png
   ```
3. **Ablation studies**
   ```bash
   python eval/ablations.py data/manifest.csv data/landmarks --train-signers S01 S02 S03 --val-signers S04 --test-signers S05
   ```

## Real-Time Demo
Phase 7 adds the AI tutor HUD and Brain executor into the webcam loop.

1) Place a Bangla font (e.g., `kalpurush.ttf` or `SolaimanLipi.ttf`) in `demo/`.
2) (Optional) Export a Gemini key if you want live model responses:
   ```bash
   export GEMINI_API_KEY="<your key>"
   ```
3) Run the demo (stub mode by default):
   ```bash
   python demo/realtime_demo.py fusion_model.pt --device cpu --buffer 48
   ```

HUD controls:
- **q / ESC**: quit
- **c**: clear buffered words
- **g**: toggle Gemini on/off (restarts executor safely)
- **p**: toggle prompt preview in HUD

Notes:
- The sentence buffer only accepts a word after it stays stable for several frames and above the confidence threshold.
- Tokens submitted to the Brain module exclude punctuation; question intent is inferred from the grammar tag.
- If no manifest is available, labels fall back to `#<idx>` numbering.

## Notes
- `data/manifest.csv` defines signer-independent train/val/test splits through explicit signer lists.
- Add augmentation, checkpointing, and logger integrations as needed.
- Ensure MediaPipe has webcam permissions on your platform.
