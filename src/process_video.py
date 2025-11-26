import cv2
import numpy as np
import argparse
import logging
import os
import joblib
import time
from tqdm import tqdm

from detect_color_shape import (
    detect_and_classify_frame,
    init_sift_verifier
)
from cnn_model import CNNClassifier
from skimage.feature import hog

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_classifiers(classifier_type='ensemble', cnn_path='computations/cnn_checkpoints/last_model.pth', hog_path='computations/hog_svm_stop_and_bg.pkl'):
    clf = None
    
    if classifier_type.lower() == 'ensemble':
        logging.info("Loading Ensemble classifiers...")
        try:
            hog_clf = joblib.load(hog_path)

            if not os.path.exists(cnn_path):
                logging.warning(f"Model {cnn_path} not found, trying best_model.pth")
                cnn_path = cnn_path.replace("last_model.pth", "best_model.pth")
            
            cnn_clf = CNNClassifier(model_path=cnn_path, input_size=224)
            clf = (hog_clf, cnn_clf)
        except Exception as e:
            logging.error(f"Failed to load classifiers: {e}")
            return None
            
    elif classifier_type.lower() == 'cnn':
        logging.info("Loading CNN classifier...")
        try:
            if not os.path.exists(cnn_path):
                logging.warning(f"Model {cnn_path} not found, trying best_model.pth")
                cnn_path = cnn_path.replace("last_model.pth", "best_model.pth")
            clf = CNNClassifier(model_path=cnn_path, input_size=224)
        except Exception as e:
            logging.error(f"Failed to load CNN: {e}")
            return None
            
    elif classifier_type.lower() == 'hog':
        logging.info("Loading HOG classifier...")
        try:
            clf = joblib.load(hog_path)
        except Exception as e:
            logging.error(f"Failed to load HOG: {e}")
            return None

    return clf

def process_video(input_path, output_path, classifier_type='ensemble', frame_skip=2, resize_width=None):
    if not os.path.exists(input_path):
        logging.error(f"Input video not found: {input_path}")
        return

    clf = load_classifiers(classifier_type)
    if clf is None: return
    
    init_sift_verifier()
    
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if resize_width and resize_width < width:
        scale_factor = resize_width / width
        new_width = resize_width
        new_height = int(height * scale_factor)
        logging.info(f"Resizing video from {width}x{height} to {new_width}x{new_height}")
    else:
        scale_factor = 1.0
        new_width = width
        new_height = height
    
    logging.info(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    pbar = tqdm(total=total_frames)

    video_scales = [0.4, 0.7, 1.0, 1.3]
    
    frame_count = 0
    last_processed_frame = None
    total_detections = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if scale_factor != 1.0:
            frame = cv2.resize(frame, (new_width, new_height))
        
        if frame_count % frame_skip == 0:
            processed_frame, num_dets = detect_and_classify_frame(frame, clf, classifier_type, cnn_threshold=0.50, scales=video_scales)
            last_processed_frame = processed_frame
            total_detections += num_dets
        else:
            if last_processed_frame is not None:
                processed_frame = last_processed_frame
            else:
                processed_frame = frame
                
        out.write(processed_frame)
        pbar.update(1)
        frame_count += 1
        
    cap.release()
    out.release()
    pbar.close()
    logging.info(f"Video saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Process videos to detect stop signs')
    parser.add_argument('--input', default='data/raw/videos', help='Path to input video directory')
    parser.add_argument('--output', default='data/processed/videos', help='Path to output video directory')
    parser.add_argument('--classifier', default='ensemble', choices=['ensemble', 'cnn', 'hog'])
    parser.add_argument('--skip', type=int, default=2, help='Process every Nth frame (default: 2)')
    parser.add_argument('--width', type=int, default=None, help='Resize video to this width (e.g. 800)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    video_extensions = ('.mp4')
    files = [f for f in os.listdir(args.input) if f.lower().endswith(video_extensions)]

    for f in files:
        in_path = os.path.join(args.input, f)
        out_path = os.path.join(args.output, f"processed_{args.classifier}_{f}")
            
        process_video(in_path, out_path, args.classifier, args.skip, args.width)
            
if __name__ == "__main__":
    main()
