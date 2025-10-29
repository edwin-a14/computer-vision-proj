# Traffic Sign Detection Project

## Setup
1. ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip3 install -r requirements.txt
    ```
2. Download the following Kaggle: https://www.kaggle.com/datasets/andrewmvd/road-sign-detection
3. Place it into `data/raw/` and rename it to `kaggle_roadsign`
4. Folder structure should be: `data/raw/kaggle_roadsign/{annotations, images}`

## Run
```bash
source .venv/bin/activate # only if terminal isn't in venv environment
python3 src/data_prep.py
python3 src/hog_svm_baseline.py
python3 src/detect_color_shape.py #create potential chips to test from full dataset
```