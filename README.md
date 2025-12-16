# Traffic Sign Detection Project

A computer vision project for detecting stop signs in road images using color-based detection, shape filtering, and machine learning classifiers (HOG-SVM and CNN).

## Quick Start

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

### 2. Download Datasets
1. **Road Sign Detection Dataset**: https://www.kaggle.com/datasets/andrewmvd/road-sign-detection
   - Place in `data/raw/` and rename to `kaggle_roadsign`
   
2. **Caltech101 Stop Sign Dataset**: https://www.kaggle.com/datasets/maricinnamon/caltech101-stop-sign-images-annotations
   - Place in `data/raw/` and rename to `CALTECH101_STOP_SIGN`

**Final structure:**
```
data/raw/
  ├── kaggle_roadsign/
  │   ├── annotations/
  │   └── images/
  └── CALTECH101_STOP_SIGN/
      ├── stop_sign/
      └── stop_sign_annotations_converted.txt
```

### 3. Prepare Training Data & Generate Color Signatures

```bash
# Step 1: Extract chips from Kaggle dataset
python3 src/data_prep.py

# Step 2: Integrate Caltech dataset
python3 src/process_caltech_dataset.py

# Step 3: Augment data for better performance 
# This increases training data from ~400 to ~4000+ samples
python3 src/augment_training_data.py --augmentations 15 --balance-ratio 0.8

# Check your dataset size
python3 -c "from pathlib import Path; print(f'Stop signs: {len(list(Path(\"data/processed/chips_augmented/train/stop\").glob(\"*\")))}'); print(f'Background: {len(list(Path(\"data/processed/chips_augmented/train/bg\").glob(\"*\")))}')"

# Step 4: Generate color signatures (required before detection)
python3 src/color_shape_prep.py
```

### 4. Train Classifiers

```bash
# Train HOG-SVM classifier
python3 src/hog_svm_baseline.py

# Train CNN classifier with augmented data
python3 src/cnn_baseline.py \
    --train-dir data/processed/chips_augmented/train \
    --val-dir data/processed/chips_augmented/val \
    --test-dir data/processed/chips_augmented/test \
    --epochs 50 --batch-size 32
```

### 5. Run Detection Pipeline

```bash
# CNN
python3 src/detect_color_shape.py \
    --classifier cnn \
    --cnn-model computations/cnn_checkpoints/best_model.pth \
    --threshold 0.4

# HOG-SVM
python3 src/detect_color_shape.py --classifier hog

# Ensemble (both must agree - highest precision, lowest recall)
python3 src/detect_color_shape.py \
    --classifier ensemble \
    --cnn-model computations/cnn_checkpoints/best_model.pth \
    --threshold 0.4
```


### 6. Evaluate Results

```bash
# Evaluate detection performance
python3 src/evaluate_detections.py

# Compare different classifiers (optional)
python3 src/compare_classifiers.py
```

### 7. Hard Negative Mining

Mining "hard negatives" (i.e. false positives) from our detections and add them to the training set.

1. Run detections using the ensemble classifier:
   ```bash
   python3 src/detect_color_shape.py --classifier ensemble
   ```

2. Run the mining script to collect false positives from images that don't contain stop signs:
   ```bash
   python3 src/mine_hard_negatives.py
   ```
   This will copy false positive chips to `data/processed/chips/train/bg`.

3. Re-train CNN classifier

### 8. Video Processing

```bash
python src/process_video.py 
    --input data/raw/videos/DrivingClip1.mp4 
    --output data/processed/videos 
    --classifier hog 
    --width 800 
    --skip 3

python src/process_video.py 
    --input data/raw/videos/DrivingClip1.mp4 
    --output data/processed/videos 
    --classifier cnn 
    --width 800 
    --skip 3

python src/process_video.py 
    --input data/raw/videos/DrivingClip1.mp4 
    --output data/processed/videos 
    --classifier ensemble 
    --width 800 
    --skip 3
```

### 9. Web Interface (Real-time)

To start the real-time streaming interface:

```bash
python src/web/app.py
```

Navigate to `http://127.0.0.1:5001` in your browser.

## Project Structure

```
computer-vision-proj/
├── src/
│   ├── data_prep.py                  # Extract chips from Kaggle dataset
│   ├── process_caltech_dataset.py    # Integrate Caltech101 stop signs
│   ├── augment_training_data.py      # Data augmentation pipeline
│   ├── color_shape_prep.py           # Color hist logic, color signatures
│   ├── hog_svm_baseline.py           # Train HOG-SVM classifier
│   ├── cnn_baseline.py               # Train CNN classifier
│   ├── cnn_model.py                  # CNN wrapper for detection
│   ├── detect_color_shape.py         # Main detection pipeline
│   ├── utils.py                      # Utility functions (I/O, WB, overlays)
│   ├── evaluate_detections.py        # Evaluate against ground truth
│   ├── sample_pixels.py              # Interactive tool for mask exploration
│   ├── analyze_false_negatives.py    # Analyze missed detections
│   ├── compare_classifiers.py        # Compare classifier results
│   └── CVproject_cnn.ipynb           # Original notebook (reference)
│
├── data/
│   ├── raw/
│   │   ├── kaggle_roadsign/          # Road sign detection dataset
│   │   │   ├── annotations/
│   │   │   └── images/
│   │   └── CALTECH101_STOP_SIGN/     # Caltech101 stop signs
│   │       ├── stop_sign/
│   │       └── stop_sign_annotations_converted.txt
│   │
│   ├── graphs/                      # Color signature visualization outputs (pie + RGB scatter)
│   └── processed/
│       ├── chips/                     # Original training chips
│       │   ├── train/{stop,bg}/
│       │   ├── val/{stop,bg}/
│       │   └── test/{stop,bg}/
│       ├── chips_augmented/           # Augmented dataset (4000+)
│       │   ├── train/{stop,bg}/
│       │   ├── val/{stop,bg}/
│       │   └── test/{stop,bg}/
│       ├── found_chips_hog/           # HOG-SVM detections
│       ├── found_chips_cnn/           # CNN detections
│       ├── found_chips_ensemble/      # Ensemble detections
│       ├── evaluation_results.json    # Evaluation metrics
│       ├── color_signatures.json      # Learned color composition
│       └── classifier_comparison/     # Comparison visualizations
│
├── computations/
│   ├── color_signatures.json         # Color histogram signatures for stop sign detection
│   ├── cnn_checkpoints/              # CNN model checkpoints
│   │   ├── best_model.pth            # Best model (highest val accuracy)
│   │   ├── last_model.pth            # Last epoch
│   │   └── history.json              # Training history
│   ├── hog_svm_stop_and_bg.pkl       # Trained HOG-SVM model
│
├── FALSE_NEGATIVE_IMPROVEMENTS.md     # Detailed improvement documentation
├── IMPROVEMENTS_APPLIED.md            # Verification of applied changes
├── README.md                          # This file
└── requirements.txt                   # Python dependencies
```