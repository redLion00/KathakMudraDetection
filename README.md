# Kathak Mudra Detection

A deep learning pipeline for recognizing **Kathak hand mudras (gestures)** from video, using full-body/hand pose landmarks and a Temporal Convolutional Network (TCN) with self-attention. The project covers the full workflow  from raw video preprocessing and landmark extraction, to model training, to real-time mudra classification via webcam.

## Overview

Kathak is a classical Indian dance form built around precise hand gestures (mudras). This project automates mudra recognition by:

1. Converting and filtering raw dance videos.
2. Extracting whole-body pose landmarks (body + both hands) using [`rtmlib`](https://github.com/Tau-J/rtmlib) (RTMPose-based whole-body keypoint detector).
3. Normalizing and preprocessing the landmark sequences.
4. Training a Residual TCN + temporal self-attention classifier on the landmark sequences.
5. Running real-time mudra classification on a live webcam feed.

## Project Structure

```
KathakMudraDetection/
├── video_conversion.ipynb        # Converts/standardizes raw video files (format, fps, etc.)
├── video_filtering.py            # Filters/trims videos before landmark extraction
├── landmark_extraction.ipynb     # Extracts body + hand pose landmarks from videos (rtmlib Wholebody)
├── landmark_normalization.ipynb  # Normalizes and preprocesses extracted landmark sequences
├── model.ipynb                   # Defines and trains the TCN-based mudra classifier
├── live_check.py                 # Real-time mudra detection using a webcam feed
├── VideoReport.csv               # Metadata/report of processed videos
├── VideoReport_updated.csv       # Updated video metadata (includes mudra labels)
├── landmarks/                    # Raw extracted landmark data
├── landmarks_processed/          # Normalized/processed landmark data used for training
└── .gitignore
```
## Dataset

The dataset used for this project (raw Kathak mudra videos / extracted landmarks) is available at the following Google Drive link:

**Dataset link:** *https://drive.google.com/drive/folders/15j-ZHvfAdJG0H7Y4YNAgGhoggMioTxE3?usp=drive_link*

Download and place the data according to the paths expected by the notebooks (`video_conversion.ipynb`, `landmark_extraction.ipynb`) before running the pipeline.

## Pipeline

**1. Video Preprocessing**
- `video_conversion.ipynb` — standardizes input videos (e.g., format/frame rate conversion).
- `video_filtering.py` — filters and prepares the video clips used for landmark extraction.

**2. Landmark Extraction**
- `landmark_extraction.ipynb` — runs a whole-body pose estimator (`rtmlib.Wholebody`) on each video frame to extract 59 keypoints per frame: 17 body keypoints and 21 keypoints per hand (left + right), each with (x, y, confidence).

**3. Landmark Normalization**
- `landmark_normalization.ipynb` — normalizes hand landmarks relative to the wrist and scale, resamples sequences to a fixed number of frames, and prepares tensors for training.

**4. Model Training**
- `model.ipynb` — trains a `TCNActionRecognizer` model:
  - Input projection layer (236-dim features → 128-dim embedding)
  - 4 stacked residual dilated TCN blocks (dilations 1, 2, 4, 8)
  - Temporal multi-head self-attention layer
  - Global average pooling + classification head
  - Trained on fixed-length (172-frame) landmark sequences derived from `VideoReport_updated.csv` mudra labels.

**5. Real-Time Inference**
- `live_check.py` — loads the trained TCN checkpoint and runs live mudra classification from a webcam:
  - Captures frames, extracts pose/hand landmarks in real time.
  - Buffers a rolling window of frames (172 frames at 15 FPS).
  - Runs inference and displays the predicted mudra with confidence and top-3 predictions as an on-screen HUD.

## Requirements

- Python 3.x
- PyTorch
- OpenCV (`opencv-python`)
- NumPy, Pandas, SciPy
- [`rtmlib`](https://github.com/Tau-J/rtmlib) (whole-body pose estimation)
- ONNX Runtime (for `rtmlib` inference backend; CUDA-enabled build recommended for GPU inference)

Install dependencies:

```bash
pip install torch opencv-python numpy pandas scipy rtmlib onnxruntime
```

> For GPU acceleration, install `onnxruntime-gpu` instead of `onnxruntime`, and a CUDA-enabled build of PyTorch.


## Usage

1. **Prepare videos:** Run `video_conversion.ipynb` and `video_filtering.py` on your raw dataset.
2. **Extract landmarks:** Run `landmark_extraction.ipynb` to generate landmark data into `landmarks/`.
3. **Normalize landmarks:** Run `landmark_normalization.ipynb` to produce processed data in `landmarks_processed/`.
4. **Train the model:** Run `model.ipynb` to train the TCN classifier and save a checkpoint (e.g., `models/saved/best_tcn.pt`).
5. **Run live detection:**
   ```bash
   python live_check.py
   ```
   Press **R** to reset the frame buffer, and **Q** to quit.

> Note: `live_check.py` currently references local paths (e.g., the CSV path and checkpoint path)  update `CHECKPOINT_PATH` and the CSV path in the script to match your local setup before running.

## Model Details

| Component | Description |
|---|---|
| Input | (172 frames × 59 landmarks × 4 features) — normalized position + velocity |
| Backbone | 4× Residual TCN blocks, dilations (1, 2, 4, 8) |
| Attention | Temporal multi-head self-attention (4 heads) |
| Pooling | Global average pooling over time |
| Output | Softmax over mudra classes |


