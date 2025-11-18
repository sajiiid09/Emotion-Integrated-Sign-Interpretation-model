
# Project Context Overview — Multi-Modal BdSL Recognition System
### (Detailed Technical Specification for Codebase Generation)

## 1. Project Title
**Multi-Modal Bangla Sign Language (BdSL) Recognition Using Hand, Face, and Pose Landmarks with Facial Grammar Classification**

---

## 2. High-Level Description

The goal is to build a **real-time, offline BdSL recognition system** that uses **RGB webcam video** to classify:

1. **The signed word** (from a 60-word BdSL vocabulary)  
2. **Facial grammar / non-manual markers**, specifically:
   - Neutral (statement)
   - Question (raised eyebrows, widened eyes)
   - Negation (head shake, eyebrow frown)

The system uses **MediaPipe Holistic** to extract:
- Hand landmarks (21 points × 2 hands)
- Face landmarks (468 points)
- Body pose landmarks (33 points)

And processes them through a **multi-stream temporal neural model**:
- **Hand encoder** (TCN or Transformer)
- **Face encoder**
- **Pose encoder**
- **Fusion module**
- **Multi-task heads** (sign classification + grammar classification)

The final system must run **in real-time** with a webcam (≈ 15–25 FPS), including live landmark extraction and classification.

---

## 3. Technical Objectives

### Required Outputs
1. **Sign Classifier Output**  
   - 60 categorical classes  
   - Top-1 prediction

2. **Facial Grammar Classifier Output**  
   - 3 categorical classes  
   - Neutral / Question / Negation

3. **Real-Time Demo UI**  
   - Webcam feed  
   - Skeleton overlay  
   - Live predictions  
   - Prediction smoothing  

4. **Training + Evaluation Scripts**
   - Signer-independent split
   - Accuracy, F1-score, confusion matrices
   - Ablation studies

5. **Dataset Tools**
   - Video capture script
   - Landmark extraction pipeline
   - Data manifest generation

---

## 4. Dataset Requirements

### Vocabulary
- **60 isolated BdSL signs** (single word signs)

### Signers
- Target: **6–8 signers**
- At minimum: **4 signers**

### Sessions
- **2 sessions per signer**
- Different clothing / lighting / day

### Repetitions
- **4–8 repetitions per word per session**
- Each repetition recorded as **one 2-second clip**

### Camera / Resolution
- Preferred: **Logitech C922** or **Intel RealSense RGB stream**
- **1080p @ 30FPS**  
- Upper body + face in frame

### File Naming Convention
```

<word>__S<signer_id>__sess<session_id>__rep<rep_id>.mp4

```

### Manifest CSV Example
```

filepath,word,signer_id,session,rep,grammar_label,fps,width,height,notes
খাওয়া__S01__sess1__rep03.mp4, খাওয়া, S01, 1, 3, neutral, 30, 1920, 1080, none

````

### Grammar Labels
- `neutral`
- `question`
- `negation`

---

## 5. Landmark Extraction Pipeline

### Tool: MediaPipe Holistic
Extract the following per-frame:

**Hands (left/right)**
- 21 points × (x, y, z, visibility)

**Face (mesh)**
- 468 points × (x, y, z)

**Pose**
- 33 points × (x, y, z, visibility)

---

### Processing Steps
1. **Load video → Extract frames**
2. **Run MediaPipe Holistic**
3. **Extract + normalize:**
   - Center coordinates on *neck* (pose landmark)
   - Scale by *shoulder width*
   - Flip horizontally if needed for canonicalization
4. **Time normalization:**
   - Target length: **48–64 frames**
   - Use cropping + zero-padding
5. **Save compressed landmark file**
   - `.npz` with arrays:
     ```
     hands_left:  (T, 21, 3)
     hands_right: (T, 21, 3)
     face:        (T, 468, 3)
     pose:        (T, 33, 3)
     ```
6. **Store metadata** in parallel `.json` or manifest CSV

---

## 6. Model Architecture

### Overview
A **multi-stream temporal encoder architecture**:
````

```
      Hands landmarks  → Hand Encoder (TCN/Transformer)
      Face landmarks   → Face Encoder
      Pose landmarks   → Pose Encoder
                          ↓
                  Temporal pooled features
                          ↓
                     Fusion MLP
                          ↓
                    Multi-task Heads
               → Sign classifier (60 classes)
               → Grammar classifier (3 classes)
```

```

---

### 6.1 Input Shapes
Assume:
- Sequence length `T = 48`
- Coordinates = normalized (x, y, z)

#### Hands Input
```

(T, 42, 3) → (T, 126)

```

#### Face Input
Large number of landmarks → reduce with PCA to 128 dims:
```

(T, 468, 3) → (T, 1404) → PCA → (T, 128)

```

#### Pose Input
```

(T, 33, 3) → (T, 99)

```

---

### 6.2 Encoders (per-stream)

You may choose either:

#### Option A: TCN-based Encoder
- 3 temporal blocks
- Kernel sizes: 5 → 5 → 3  
- Hidden dims: 128 → 128 → 256  
- Dropout: 0.1

#### Option B: Transformer Encoder
- 2 layers  
- Embedding dim: 128  
- FF dim: 256  
- Heads: 4  
- Dropout: 0.1  

---

### 6.3 Fusion Module
After temporal pooling:

```

hand_vec  = HandEncoder(x_hands)  → (256)
face_vec  = FaceEncoder(x_face)   → (256)
pose_vec  = PoseEncoder(x_pose)   → (256)

concat = [hand_vec | face_vec | pose_vec]   → (768)

fusion = MLP(768 → 256 → 128)

```

---

### 6.4 Heads (Multi-task)
#### Sign classification:
- Linear → Softmax(60)

#### Grammar classification:
- Linear → Softmax(3)

#### Loss:
```

Total_Loss = CE(sign) + 0.5 * CE(grammar)

```

---

## 7. Training Specification

### Splits
- **Signer-independent**
- Train: 70%
- Val: 10%
- Test: 20% (signers not seen in training)

### Hyperparameters
- Optimizer: AdamW  
- LR: 3e-4  
- Batch size: 64–128  
- Epochs: 30–50  
- Weight decay: 0.01  
- LR scheduler: cosine annealing  

### Data Augmentations
- Temporal jitter  
- Random crop of starting frame  
- Time stretching (0.95× – 1.05×)  
- Gaussian coordinate noise  

---

## 8. Evaluation Requirements

### Metrics
- **Sign accuracy (top-1)**  
- **Grammar F1-score**  
- **Macro accuracy across signers**  
- **Confusion matrices**  
- **Ablations**:
  - hands-only
  - face-only
  - pose-only
  - hands+face
  - full fusion

### Expected results
- Sign accuracy: **70–80%**  
- Grammar F1: **0.80–0.90**  

---

## 9. Real-Time Demo Requirements

### Tasks
- Live webcam stream (OpenCV)
- Run MediaPipe Holistic in real-time
- Maintain rolling buffer of **48 frames**
- Convert landmarks to normalized tensors
- Run fused model inference
- Smooth predictions (EMA or majority vote)
- Overlay:
  - Landmarks
  - Predicted sign
  - Predicted grammar

### Output UI
- Simple CV2 or Streamlit interface
- Framerate target: **15–25 FPS**

---

## 10. Codebase Structure (Recommended)

```

project/
│
├── capture/
│   └── record_videos.py
│
├── preprocess/
│   ├── extract_landmarks.py
│   ├── normalize.py
│   └── build_dataset.py
│
├── data/
│   ├── raw/
│   ├── landmarks/
│   └── manifest.csv
│
├── models/
│   ├── encoders.py
│   ├── fusion.py
│   └── classifier.py
│
├── train/
│   ├── train_baselines.py
│   ├── train_fusion.py
│   └── utils.py
│
├── eval/
│   ├── evaluate.py
│   ├── confusion_matrix.py
│   └── ablations.py
│
└── demo/
├── realtime_demo.py
└── ui_helpers.py

```

---

## 11. Deliverables Needed by Codex

Codex should generate:

1. Data capture script  
2. Landmark extraction + saving pipeline  
3. PCA reduction for face mesh  
4. Multi-stream temporal encoders  
5. Fusion module + heads  
6. Training loop (with metrics + checkpoints)  
7. Evaluation logic (signer-independent)  
8. Real-time inference UI code  
9. Utility scripts (manifest creation, dataset split)

---

## 12. Summary

This document defines the **full and detailed context** needed for generating a complete codebase for a **multi-modal BdSL recognition system**.  
All architectural, data, pipeline, and engineering specifications are included so that an LLM (or Codex-like model) can automatically scaffold and implement the system.


