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
from action_inference import determine_driver_action
from cnn_model import CNNClassifier
from speed_estimation import SpeedEstimator
from lane_detection import LaneDetector
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
    
    speed_est = SpeedEstimator(new_width, new_height, fps)
    lane_det = LaneDetector(new_width, new_height)
    
    pbar = tqdm(total=total_frames)

    video_scales = [0.4, 0.7, 1.0, 1.3]
    
    frame_count = 0
    trackers = [] 
    last_detected_boxes = []
    last_action = "Unknown"
    last_action_color = (0, 255, 0)
    total_detections = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if scale_factor != 1.0:
            frame = cv2.resize(frame, (new_width, new_height))
            
        speed_est.estimate_speed(frame)

        processed_frame = lane_det.detect_lanes(frame)
        
        if frame_count % frame_skip == 0:
            processed_frame, num_dets, detections = detect_and_classify_frame(processed_frame, clf, classifier_type, cnn_threshold=0.50, scales=video_scales)
            
            trackers = []
            for (x, y, w, h) in detections:
                tracker = cv2.TrackerMIL_create()
                
                tracker.init(frame, (x, y, w, h))
                trackers.append(tracker)

            action, color = determine_driver_action(detections, new_width, new_height)
            
            cv2.putText(processed_frame, f"ACTION: {action}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            if detections:
                # Find largest detection (closest stop sign)
                largest_det = max(detections, key=lambda b: b[2] * b[3])
                speed_est.process_stop_sign(largest_det, frame_count)
            
            last_detected_boxes = detections
            last_action = action
            last_action_color = color
            total_detections += num_dets
        else:

            updated_boxes = []
            for tracker in trackers:
                success, box = tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in box]
                    updated_boxes.append((x, y, w, h))
            
            if not updated_boxes and last_detected_boxes:
                 updated_boxes = last_detected_boxes

            for (x, y, w, h) in updated_boxes:
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(processed_frame, "Stop", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            
            # Draw last known action
            cv2.putText(processed_frame, f"ACTION: {last_action}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, last_action_color, 3)
        
        speed_est.draw_speed(processed_frame)
                
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
    
    if os.path.isfile(args.input):
        files = [os.path.basename(args.input)]
        input_dir = os.path.dirname(args.input)
    else:
        files = [f for f in os.listdir(args.input) if f.lower().endswith(video_extensions)]
        input_dir = args.input

    for f in files:
        in_path = os.path.join(input_dir, f)
        out_path = os.path.join(args.output, f"processed_{args.classifier}_{f}")
            
        process_video(in_path, out_path, args.classifier, args.skip, args.width)
            
if __name__ == "__main__":
    main()
