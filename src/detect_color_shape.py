import cv2
import numpy as np
import os
import shutil
import logging
import argparse
import json
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from skimage.feature import hog
from sklearn.svm import LinearSVC
import pickle
import joblib
import matplotlib.colors as clr
from typing import Optional, Union
from cnn_model import CNNClassifier
import glob

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

_sift = None
_sift_ref_descriptors = [] # List of descriptors for multiple reference images
_sift_matcher = None

def init_sift_verifier(ref_img_paths=None):
    global _sift, _sift_ref_descriptors, _sift_matcher
    
    if ref_img_paths is None:
        # Theres prob a better way to select the reference images, but for now use top 5 stop sign chips
        files = glob.glob('data/processed/chips/train/stop/*.png')
        files.sort()
        ref_img_paths = files[:5] if files else []

    try:
        _sift = cv2.SIFT_create()
        _sift_ref_descriptors = []
        
        for path in ref_img_paths:
            ref_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
            kp, des = _sift.detectAndCompute(ref_img, None)
            if des is not None and len(des) > 0:
                _sift_ref_descriptors.append(des)
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        _sift_matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
    except Exception as e:
        logging.warning(f"Failed to initialize SIFT verifier: {e}")

def verify_with_sift(chip_img, min_matches=2):
    global _sift, _sift_ref_descriptors, _sift_matcher
    
    if _sift is None or not _sift_ref_descriptors:
        return True
        
    try:
        gray = cv2.cvtColor(chip_img, cv2.COLOR_BGR2GRAY)
        kp, des = _sift.detectAndCompute(gray, None)
        
        if des is None or len(des) < 2:
            return False
            
        # Check against all reference images, if any reference image has enough matches, we accept it
        for ref_des in _sift_ref_descriptors:
            matches = _sift_matcher.knnMatch(ref_des, des, k=2)
            
            good_matches = 0
            for m_n in matches:
                if len(m_n) != 2: continue
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good_matches += 1
            
            if good_matches >= min_matches:
                return True
                
        return False
    except Exception as e:
        return False

def non_max_suppression(boxes, scores, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    areas = boxes[:, 2] * boxes[:, 3]
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        
        inds = np.where(iou <= overlap_thresh)[0]
        order = order[inds + 1]
    
    return keep

def detect_single_scale(img, min_area, max_area, shape_threshold=0.25):
    candidates = []
    
    # Detect red regions
    red_mask = detect_red_regions(img)
    
    # Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and score contours
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area < min_area or area > max_area:
            continue
        
        shape_score = calculate_shape_score(contour)
        
        if shape_score < shape_threshold:  # Lowered threshold for better recall
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        candidates.append(((x, y, w, h), shape_score))
    
    return candidates

def detect_multiscale(orig_img, scales=[0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 1.8]):
    #Detect traffic signs at multiple scales for better small/large sign detection.
    
    img_height, img_width = orig_img.shape[:2]
    img_area = img_height * img_width
    
    # Lowered min area to 100 to detect smaller/further signs
    base_min_area = max(100, int(img_area * 0.0001))  
    base_max_area = int(img_area * 0.4)
    
    all_candidates = []
    
    for scale in scales:
        # Resize image
        scaled_h = int(img_height * scale)
        scaled_w = int(img_width * scale)
        
        if scaled_h < 40 or scaled_w < 40:
            continue
        
        scaled_img = cv2.resize(orig_img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        
        # Adjust area thresholds for this scale
        scaled_min_area = max(80, int(base_min_area * (scale ** 2))) 
        scaled_max_area = int(base_max_area * (scale ** 2))
        
        # Detect at this scale
        candidates = detect_single_scale(scaled_img, scaled_min_area, scaled_max_area)
        
        # Scale bounding boxes back to original image coordinates
        for (x, y, w, h), shape_score in candidates:
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            orig_w = int(w / scale)
            orig_h = int(h / scale)
            
            # Store with scale information for scoring
            all_candidates.append(((orig_x, orig_y, orig_w, orig_h), shape_score, scale))
    
    # If no candidates found, return empty list
    if not all_candidates:
        return []
    
    # Prepare for NMS - we need boxes and scores
    boxes = [bbox for bbox, _, _ in all_candidates]
    scores = [shape_score for _, shape_score, _ in all_candidates]
    
    # Apply NMS across all scales with more lenient threshold
    keep_indices = non_max_suppression(boxes, scores, overlap_thresh=0.4)
    
    # Return kept detections (without scale info)
    final_detections = [(boxes[i], scores[i]) for i in keep_indices]
    
    return final_detections

def detect_red_regions(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Strategy 1: HSV with more lenient adaptive saturation threshold
    h, s, v = cv2.split(hsv)
    sat_mean = np.mean(s)
    # Loosened saturation threshold back to handle desaturated signs
    sat_threshold = max(30, int(sat_mean * 0.5))
    val_threshold = 25
    
    lower_red1 = np.array([0, sat_threshold, val_threshold])
    upper_red1 = np.array([12, 255, 255]) # Loosened Hue range
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    
    lower_red2 = np.array([165, sat_threshold, val_threshold]) # Loosened Hue range
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    hsv_mask = cv2.bitwise_or(mask1, mask2)
    
    # Strategy 2: LAB color space (L*a*b*)
    # Red has high 'a' channel value (green-red axis)
    l, a, b_channel = cv2.split(lab)
    
    a_threshold = np.percentile(a, 75) 
    a_threshold = max(a_threshold, 115) # Loosened threshold
    
    l_threshold = 20  
    
    lab_mask = np.zeros_like(a, dtype=np.uint8)
    lab_mask[(a > a_threshold) & (l > l_threshold)] = 255
    
    # Combine masks from both color spaces
    combined_mask = cv2.bitwise_or(hsv_mask, lab_mask)
    
    # Reduced kernel size slightly to preserve smaller signs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    return combined_mask

def is_octagon(contour, epsilon_factor=0.02):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon_factor * perimeter, True)
    
    num_vertices = len(approx)
    is_convex = cv2.isContourConvex(approx)
    
    return 5 <= num_vertices <= 10 and is_convex

def calculate_shape_score(contour):
    # Calculate a score for how likely a contour is a stop sign.
    # Considers area, aspect ratio, and octagonal shape.

    area = cv2.contourArea(contour)
    if area < 100:
        return 0.0
    
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    
    # Stop signs are roughly square (aspect ratio near 1)
    aspect_diff = abs(1.0 - aspect_ratio)
    if aspect_diff < 0.4:  # Very square
        aspect_score = 1.0
    elif aspect_diff < 0.8:  # Somewhat square
        aspect_score = 0.7
    else:
        aspect_score = 0.5
    
    if is_octagon(contour):
        octagon_score = 1.0
    else:
        # Check if it's at least convex
        if cv2.isContourConvex(contour):
            octagon_score = 0.6  
        else:
            octagon_score = 0.4 
    
    # Circularity/compactness
    perimeter = cv2.arcLength(contour, True)
    if perimeter > 0:
        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        circularity_score = min(circularity, 1.0)
    else:
        circularity_score = 0.0
    
    # Adjusted weights: balanced scoring to avoid over-filtering
    score = (aspect_score * 0.35 + octagon_score * 0.35 + circularity_score * 0.30)
    
    return score

def extract_chip_with_padding(img, x, y, w, h, target_size=128, padding_ratio=0.1, keep_aspect_ratio=True):
    # Extract a chip from an image with padding and proper resizing.
    # Maintains aspect ratio and adds context around the detection.

    height, width = img.shape[:2]
    
    if w <= 0 or h <= 0 or x < 0 or y < 0:
        return None
    
    pad_w = int(w * padding_ratio)
    pad_h = int(h * padding_ratio)
    
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(width, x + w + pad_w)
    y2 = min(height, y + h + pad_h)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    cropped = img[y1:y2, x1:x2]
    
    if cropped.size == 0 or cropped.shape[0] < 2 or cropped.shape[1] < 2:
        return None
    
    # If we don't care about aspect ratio (e.g. for CNN trained on distorted crops),
    # just resize directly.
    if not keep_aspect_ratio:
        try:
            return cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        except Exception:
            return None

    # Resize to target size maintaining aspect ratio with padding
    h_crop, w_crop = cropped.shape[:2]
    
    # Ensure minimum dimensions for valid HOG extraction
    if h_crop < 8 or w_crop < 8:
        return None
    
    scale = target_size / max(h_crop, w_crop)
    
    new_w = max(1, int(w_crop * scale))
    new_h = max(1, int(h_crop * scale))
    
    if new_w < 8 or new_h < 8:
        return None
    
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # Center the resized image on the canvas
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

def main(classifier_type: str = 'hog', cnn_model_path: Optional[str] = None, 
         cnn_threshold: float = 0.5):
    
    logging.info(f"Starting traffic sign detection pipeline")
    logging.info(f"Classifier: {classifier_type.upper()}")
    
    directory_path = "data/raw/kaggle_roadsign/images"
    
    # Create classifier-specific output directory
    classifier_suffix = classifier_type.lower()
    results_path = f"data/processed/found_chips_{classifier_suffix}"
    
    os.makedirs(results_path, exist_ok=True)
    logging.info(f"Output directory: {results_path}")

    images = []
    try:
        entries = os.listdir(directory_path)
        for entry in entries:
            full_path = os.path.join(directory_path, entry)
            if os.path.isfile(full_path): 
                images.append(entry)
                path = os.path.join(results_path, os.path.splitext(entry)[0])
                os.makedirs(path, exist_ok=True)
    except FileNotFoundError:
        logging.error(f"Directory not found at {directory_path}")
        return
    
    # Load classifier based on type
    if classifier_type.lower() == 'sift':
        logging.info("Using SIFT-only detection mode")
        clf = None # No ML classifier needed
        init_sift_verifier()
    elif classifier_type.lower() == 'ensemble':
        # Load both classifiers for ensemble mode
        logging.info("Loading both HOG-SVM and CNN for ensemble mode...")
        
        # Load HOG-SVM
        hog_path = os.path.join("computations", "hog_svm_stop_and_bg.pkl")
        try:
            hog_clf = joblib.load(hog_path)
            logging.info(f"  HOG-SVM loaded from {hog_path}")
        except Exception as e:
            logging.error(f"Failed to load HOG-SVM classifier: {e}")
            return
        
        # Load CNN
        if cnn_model_path is None:
            cnn_model_path = os.path.join("computations", "cnn_stop_classifier.pth")
        
        if not os.path.exists(cnn_model_path):
            logging.error(f"CNN model not found at {cnn_model_path}")
            logging.error("Please train a CNN model first or provide the correct path")
            return
        
        try:
            cnn_clf = CNNClassifier(model_path=cnn_model_path, input_size=224)
            cnn_clf.threshold = cnn_threshold
            logging.info(f"  CNN loaded from {cnn_model_path}")
        except Exception as e:
            logging.error(f"Failed to load CNN classifier: {e}")
            return
        
        # Store both as tuple
        clf = (hog_clf, cnn_clf)
        logging.info("Ensemble mode: Both classifiers must agree for detection")
        
    elif classifier_type.lower() == 'cnn':
        
        if cnn_model_path is None:
            cnn_model_path = os.path.join("computations", "cnn_stop_classifier.pth")
        
        if not os.path.exists(cnn_model_path):
            logging.error(f"CNN model not found at {cnn_model_path}")
            logging.error("Please train a CNN model first or provide the correct path")
            return
        
        try:
            clf = CNNClassifier(model_path=cnn_model_path, input_size=224)
            clf.threshold = cnn_threshold
            logging.info(f"Loaded CNN classifier from {cnn_model_path}")
            logging.info(f"Classification threshold: {cnn_threshold}")
        except Exception as e:
            logging.error(f"Failed to load CNN classifier: {e}")
            return
    else:
        # Default to HOG-SVM
        computations_path = os.path.join("computations", "hog_svm_stop_and_bg.pkl")
        try:
            clf = joblib.load(computations_path)
            logging.info(f"Loaded HOG-SVM classifier from {computations_path}")
        except Exception as e:
            logging.error(f"Failed to load HOG-SVM classifier: {e}")
            return

    if classifier_type.lower() in ['cnn', 'ensemble', 'hog']:
        init_sift_verifier()

    total_images = len(images)
    detection_stats = {
        'total_images': total_images,
        'images_with_detections': 0,
        'total_detections': 0,
        'images_processed': []
    }
    
    for idx, road_sign_image in enumerate(images):
        if idx % 50 == 0:
            logging.info(f"Processing image {idx+1}/{total_images}: {road_sign_image}")
        
        try:
            num_detections = process_single_image(road_sign_image, directory_path, results_path, clf, 
                                                 classifier_type=classifier_type, cnn_threshold=cnn_threshold)
            if num_detections > 0:
                detection_stats['images_with_detections'] += 1
                detection_stats['total_detections'] += num_detections
            detection_stats['images_processed'].append({
                'name': road_sign_image,
                'detections': num_detections
            })
        except Exception as e:
            logging.warning(f"Error processing {road_sign_image}: {e}")
            continue
    
    # Save statistics
    stats_file = os.path.join(results_path, f'detection_stats_{classifier_type}.json')
    with open(stats_file, 'w') as f:
        json.dump(detection_stats, f, indent=2)
    
    logging.info("Detection pipeline complete!")
    logging.info(f"Total detections: {detection_stats['total_detections']}")
    logging.info(f"Images with detections: {detection_stats['images_with_detections']}/{total_images}")
    logging.info(f"Statistics saved to: {stats_file}")
    
    return detection_stats

def detect_sift_only(orig_img, min_matches=4):
    """
    Detect stop signs using only SIFT feature matching against reference images.
    This avoids the sliding window approach entirely.
    """
    global _sift, _sift_ref_descriptors, _sift_matcher
    
    if _sift is None:
        init_sift_verifier()
        
    if not _sift_ref_descriptors:
        return orig_img, 0, []

    detected_boxes = []
    num_detections = 0
    
    try:
        gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        kp, des = _sift.detectAndCompute(gray, None)
        
        if des is None or len(des) < min_matches:
            return orig_img, 0, []
            
        # Match against all reference images
        all_good_matches = []
        
        for ref_des in _sift_ref_descriptors:
            matches = _sift_matcher.knnMatch(ref_des, des, k=2)
            
            for m_n in matches:
                if len(m_n) != 2: continue
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    # m.trainIdx is the index of the keypoint in the scene (frame)
                    all_good_matches.append(kp[m.trainIdx].pt)
        
        if len(all_good_matches) < min_matches:
            return orig_img, 0, []
            
        # Cluster the matched keypoints to find potential objects
        points = np.array(all_good_matches)
        
        # Use DBSCAN to cluster points that are close together
        # eps=50 pixels, min_samples=3 points to form a cluster
        clustering = DBSCAN(eps=50, min_samples=3).fit(points)
        labels = clustering.labels_
        
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1: continue # Noise
            
            cluster_points = points[labels == label]
            
            if len(cluster_points) < min_matches:
                continue
                
            # Find bounding box of the cluster
            x_min, y_min = np.min(cluster_points, axis=0)
            x_max, y_max = np.max(cluster_points, axis=0)
            
            w = int(x_max - x_min)
            h = int(y_max - y_min)
            x = int(x_min)
            y = int(y_min)
            
            # Add some padding
            pad_w = int(w * 0.2)
            pad_h = int(h * 0.2)
            x = max(0, x - pad_w)
            y = max(0, y - pad_h)
            w = w + 2*pad_w
            h = h + 2*pad_h
            
            # Filter by aspect ratio (stop signs are roughly square)
            aspect_ratio = float(w) / h if h > 0 else 0
            if 0.5 < aspect_ratio < 2.0 and w > 20 and h > 20:
                num_detections += 1
                detected_boxes.append((x, y, w, h))
                
                # # Draw detection
                # cv2.rectangle(orig_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.putText(orig_img, f'SIFT ({len(cluster_points)})', (x, y - 10), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
    except Exception as e:
        logging.error(f"Error during SIFT-only detection: {e}")
        
    return orig_img, num_detections, detected_boxes

def detect_color_candidates(img, min_area=100):
    """
    Detect red regions to use as candidates for HOG classification.
    This is much faster than sliding window and has better recall than SIFT.
    """
    candidates = []
    # Detect red regions once
    red_mask = detect_red_regions(img)
    
    # Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        # Use the existing shape score function to filter obvious non-signs
        shape_score = calculate_shape_score(contour)
        
        # Lower threshold to be inclusive at this stage (we just want candidates)
        if shape_score > 0.05: # Lowered from 0.15
            x, y, w, h = cv2.boundingRect(contour)
            candidates.append((x, y, w, h))
            
    return candidates

def detect_and_classify_frame(orig_img, clf, classifier_type='hog', cnn_threshold=0.85, output_path=None, file_name=None, scales=None):
    if classifier_type.lower() == 'sift':
        return detect_sift_only(orig_img)

    results = []
    bounding_boxes = []
    detection_scores = []
    
    if scales is None:
        scales = [0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 1.8]
    
    # Step 1: Candidate Generation
    if classifier_type.lower() == 'hog':
        # Use Color to find candidate regions (Fast & Accurate for red signs)
        # This replaces the slow multiscale sliding window and the inaccurate SIFT-only proposal
        color_boxes = detect_color_candidates(orig_img)
        
        candidates = []
        for (x, y, w, h) in color_boxes:
            # Add the detected box
            candidates.append(((x, y, w, h), 1.0)) # 1.0 is dummy score
            
            # Add scale variations to ensure HOG gets a good crop
            cx, cy = x + w//2, y + h//2
            
            # 1.2x scale (more context)
            w2, h2 = int(w*1.2), int(h*1.2)
            x2, y2 = max(0, cx - w2//2), max(0, cy - h2//2)
            candidates.append(((x2, y2, w2, h2), 0.9))
            
            # 1.4x scale (even more context, sometimes needed)
            w3, h3 = int(w*1.4), int(h*1.4)
            x3, y3 = max(0, cx - w3//2), max(0, cy - h3//2)
            candidates.append(((x3, y3, w3, h3), 0.8))
            
    else:
        # Standard multiscale detection for other modes
        candidates = detect_multiscale(orig_img, scales=scales)
    
    # Step 2: Extract chips
    for (x, y, w, h), shape_score in candidates:
        chip = extract_chip_with_padding(orig_img, x, y, w, h, target_size=128, padding_ratio=0.0, keep_aspect_ratio=False)
        if chip is not None:
            results.append(chip)
            bounding_boxes.append((x, y, w, h))
            detection_scores.append(shape_score)
    
    # Step 3: Apply Non-Maximum Suppression
    final_scores = []
    if len(bounding_boxes) > 0:
        keep_indices = non_max_suppression(bounding_boxes, detection_scores, overlap_thresh=0.3)
        results = [results[i] for i in keep_indices]
        bounding_boxes = [bounding_boxes[i] for i in keep_indices]
        final_scores = [detection_scores[i] for i in keep_indices]
    
    # Step 4: Classify and draw final detections
    final_img, num_detections, detected_boxes = test(results, bounding_boxes, final_scores, orig_img.copy(), clf, classifier_type, cnn_threshold, output_path, file_name)
    
    return final_img, num_detections, detected_boxes

def process_single_image(road_sign_image, directory_path, results_path, clf, 
                        classifier_type: str = 'hog', cnn_threshold: float = 0.85):

    path = os.path.join(directory_path, road_sign_image)
    orig_img = cv2.imread(path)
    
    if orig_img is None:
        logging.warning(f"Failed to read image: {road_sign_image}")
        return 0
    
    output_path = os.path.join(results_path, os.path.splitext(road_sign_image)[0])
    
    # Clean up previous results
    remove_previous_chips(output_path, 0, road_sign_image)
    
    final_img, num_detections, _ = detect_and_classify_frame(orig_img, clf, classifier_type, cnn_threshold, output_path, road_sign_image)
    cv2.imwrite(os.path.join(output_path, "result.png"), final_img)
    
    return num_detections


def test(results: list, bounding_boxes: list, scores: list, orig_img, clf, classifier_type: str = 'hog', cnn_threshold: float = 0.85, output_path=None, file_name=None):
    """
    Test detected chips using either HOG-SVM or CNN classifier, or ensemble.
    
    Args:
        results: List of cropped image chips
        bounding_boxes: List of bounding boxes corresponding to chips
        scores: List of shape scores corresponding to chips
        orig_img: Original image to draw on
        clf: Classifier (HOG-SVM, CNN, or tuple of both for ensemble)
        classifier_type: 'hog', 'cnn', or 'ensemble'
        cnn_threshold: Confidence threshold for CNN
        output_path: Path to save detected chips
        file_name: Original image filename
        
    Returns:
        orig_img: Annotated image
        num_detections: Number of detections
        detected_boxes: List of detected bounding boxes [(x, y, w, h), ...]
    """
    detected_boxes = []
    if len(results) == 0:
        return orig_img, 0, []
    
    num_detections = 0
    
    # Ensemble mode: both classifiers must agree
    if classifier_type.lower() == 'ensemble':
        try:
            hog_clf, cnn_clf = clf
            
            # Get HOG-SVM predictions
            hog_predictions = []
            features = []
            valid_indices = []
            
            for i, result in enumerate(results):
                if result is None or result.size == 0:
                    continue
                if len(result.shape) != 3 or result.shape[2] != 3:
                    continue
                if result.shape[0] < 8 or result.shape[1] < 8:
                    continue
                
                try:
                    # Extract features
                    hog_features = hog(result, orientations=9, pixels_per_cell=(8, 8),
                                      cells_per_block=(2, 2), block_norm='L2-Hys',
                                      channel_axis=2, feature_vector=True)
                    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
                    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
                    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
                    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
                    hist_h = hist_h.flatten() / (hist_h.sum() + 1e-7)
                    hist_s = hist_s.flatten() / (hist_s.sum() + 1e-7)
                    hist_v = hist_v.flatten() / (hist_v.sum() + 1e-7)
                    color_features = np.concatenate([hist_h, hist_s, hist_v])
                    combined_features = np.concatenate([hog_features, color_features])
                    
                    if np.isfinite(combined_features).all():
                        features.append(combined_features)
                        valid_indices.append(i)
                except:
                    continue
            
            # Get HOG predictions
            if len(features) > 0:
                features_array = np.array(features)
                if hasattr(hog_clf, 'decision_function'):
                    decision_scores = hog_clf.decision_function(features_array)
                    hog_predictions = ['stop' if score > -0.1 else 'bg' for score in decision_scores]
                else:
                    hog_predictions = hog_clf.predict(features_array)
            
            # Get CNN predictions
            valid_results = [results[i] for i in valid_indices]
            if len(valid_results) > 0:
                cnn_preds, cnn_confs = cnn_clf.predict_with_confidence(np.array(valid_results))
            else:
                cnn_preds, cnn_confs = [], []
            
            # Smart Ensemble Logic: Combine classifiers + SIFT verification
            for j, idx in enumerate(valid_indices):
                if j < len(hog_predictions) and j < len(cnn_preds):
                    
                    is_stop = False
                    confidence = cnn_confs[j]
                    #shape_score = scores[idx] if idx < len(scores) else 0
                    required_matches = 2
                    
                    # If strong CNN confidence, trust it
                    if cnn_preds[j] == 'stop' and confidence > 0.50:
                        is_stop = True
                        required_matches = 0
                        
                    # Moderate CNN + Minimal Verification
                    # If CNN is > 0.50, we just need one SIFT match or HOG agreement
                    elif cnn_preds[j] == 'stop' and confidence > 0.50:
                        if hog_predictions[j] == 'stop':
                            is_stop = True
                            required_matches = 0 # HOG confirms it
                        else:
                            is_stop = True
                            required_matches = 1 # Just 1 SIFT match needed
                    
                    if is_stop:
                        # Verify with SIFT if required
                        if required_matches == 0 or verify_with_sift(results[idx], min_matches=required_matches):
                            num_detections += 1
                            x, y, w, h = bounding_boxes[idx]
                            detected_boxes.append((x, y, w, h))
                            orig_img = cv2.rectangle(orig_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            label = 'STOP'
                            cv2.putText(orig_img, label, (x, y - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                            if output_path and file_name:
                                chip_path_str = chip_path(output_path, num_detections, file_name)
                                cv2.imwrite(chip_path_str, results[idx])
        except Exception as e:
            logging.error(f"Error during ensemble classification: {e}")
        
        return orig_img, num_detections, detected_boxes
    
    # Use CNN classifier
    if classifier_type.lower() == 'cnn':
        try:
            # Stack results into numpy array
            chips_array = np.array(results)
            
            # Get predictions with confidence scores
            predictions, confidences = clf.predict_with_confidence(chips_array)
            
            # DEBUG: Uncomment to save chips
            # debug_dir = "debug_chips"
            # os.makedirs(debug_dir, exist_ok=True)
            # for i, chip in enumerate(results):
            #     cv2.imwrite(os.path.join(debug_dir, f"chip_{i}.png"), chip)
            
            confidence_threshold = cnn_threshold if cnn_threshold is not None else 0.85
            
            # Draw detections for predictions meeting threshold
            for i, (pred, conf) in enumerate(zip(predictions, confidences)):
                if pred == 'stop' and conf >= confidence_threshold:
                    num_detections += 1
                    x, y, w, h = bounding_boxes[i]
                    detected_boxes.append((x, y, w, h))
                    orig_img = cv2.rectangle(orig_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # Add label with confidence
                    label = f'STOP {conf:.2f}'
                    cv2.putText(orig_img, label, (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    if output_path and file_name:
                        chip_path_str = chip_path(output_path, num_detections, file_name)
                        cv2.imwrite(chip_path_str, results[i])
        except Exception as e:
            logging.error(f"Error during CNN classification: {e}")
        
        return orig_img, num_detections, detected_boxes
    
    # Use HOG-SVM classifier
    features = []
    valid_indices = []
    
    for i, result in enumerate(results):
        # Validate chip before feature extraction
        if result is None or result.size == 0:
            continue
        
        if len(result.shape) != 3 or result.shape[2] != 3:
            continue
        
        if result.shape[0] < 8 or result.shape[1] < 8:
            continue
        
        try:
            # Extract improved features (HOG + color histograms)
            # Match the feature extraction from hog_svm_baseline.py
            hog_features = hog(result, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), block_norm='L2-Hys',
                              channel_axis=2, feature_vector=True)
            
            # Extract color histogram features
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
            
            hist_h = hist_h.flatten() / (hist_h.sum() + 1e-7)
            hist_s = hist_s.flatten() / (hist_s.sum() + 1e-7)
            hist_v = hist_v.flatten() / (hist_v.sum() + 1e-7)
            
            color_features = np.concatenate([hist_h, hist_s, hist_v])
            
            # Combine HOG and color features
            combined_features = np.concatenate([hog_features, color_features])
            
            # Validate features
            if np.isfinite(combined_features).all():
                features.append(combined_features)
                valid_indices.append(i)
        except Exception as e:
            logging.debug(f"Skipping invalid chip at index {i}: {e}")
            continue
    
    if len(features) == 0:
        return orig_img, num_detections
    
    try:
        features_array = np.array(features)
        
        # Use decision_function for more control over threshold
        # For SVC with RBF kernel, decision_function gives distance from hyperplane
        decision_scores = clf.decision_function(features_array)
        
        logging.debug(f"Decision scores range: [{decision_scores.min():.2f}, {decision_scores.max():.2f}]")
        
        # Collect candidates that pass the threshold
        candidates = []
        for j, score in enumerate(decision_scores):
            if score > -0.3:
                candidates.append((score, valid_indices[j]))
        
        # Sort by score descending (highest confidence first)
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Only verify the top 5 candidates with SIFT to save time
        # (Usually there are only 1-2 stop signs per frame)
        for score, pred_idx in candidates[:5]:
            # Verify with SIFT (require at least 1 match to confirm)
            if verify_with_sift(results[pred_idx], min_matches=1):
                num_detections += 1
                x, y, w, h = bounding_boxes[pred_idx]
                detected_boxes.append((x, y, w, h))
                # Draw green rectangle for stop sign detections
                orig_img = cv2.rectangle(orig_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Add label
                cv2.putText(orig_img, 'STOP', (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                if output_path and file_name:
                    chip_path_str = chip_path(output_path, num_detections, file_name)
                    cv2.imwrite(chip_path_str, results[pred_idx])
    except Exception as e:
        logging.error(f"Error during classification: {e}")
    
    return orig_img, num_detections, detected_boxes


def chip_path(directory: str, index: int, file_name: str):
    parts = os.path.splitext(file_name)
    return os.path.join(directory, parts[0]+'_'+ str(index)+parts[-1])

def remove_previous_chips(directory: str, index: int, file_name: str):
    while True:
        path = chip_path(directory, index, file_name)
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                logging.warning(f"Could not remove {path}: {e}")
            index+=1
        else:
            break


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Traffic sign detection using color, shape, and ML classifiers')
    parser.add_argument('--classifier', type=str, default='hog', choices=['hog', 'cnn', 'ensemble', 'sift'],
                        help='Type of classifier to use: hog (HOG-SVM), cnn (CNN), ensemble (both must agree), or sift (SIFT-only)')
    parser.add_argument('--cnn-model', type=str, default=None,
                        help='Path to CNN model checkpoint (.pth file) - required if using CNN or ensemble classifier')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for CNN predictions (default: 0.5)')
    
    args = parser.parse_args()
    
    # Validate CNN arguments
    if args.classifier in ['cnn', 'ensemble']:
        if args.cnn_model is None:
            print(f"Error: --cnn-model is required when using {args.classifier} classifier")
            exit(1)
        if not os.path.exists(args.cnn_model):
            print(f"Error: CNN model file not found: {args.cnn_model}")
            exit(1)
    
    main(classifier_type=args.classifier, cnn_model_path=args.cnn_model, cnn_threshold=args.threshold)
