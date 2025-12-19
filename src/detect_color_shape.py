import cv2
import numpy as np
import os
import logging
import argparse
import json
from skimage.feature import hog
import joblib
from typing import Optional
from cnn_model import CNNClassifier
import glob
from color_shape_prep import validate_histogram_against_signature, extract_color_histogram
from hog_svm_baseline import extract_improved_features
from utils import draw_histogram_overlay, remove_previous_outputs, apply_gray_world

# Global debug flags
debug_shape = False
draw_hist = False
debug_mask = False

# --- CONSTANTS ---
# Must match the chip size used for HOG-SVM training in hog_svm_baseline.py
HOG_CHIP_SIZE = 128

# Valid labels for detection/classification (from HOG/CNN prep)
# VALID_LABELS: list of valid detection/classification labels (currently only 'stop').
# If you add more classes, update this list. [Currently: valid label = 'stop']
VALID_LABELS = ["stop"]

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Global color signatures (for histogram checks)
_color_signatures = None

_sift = None
_sift_ref_descriptors = []  # List of descriptors for multiple reference images
_sift_matcher = None


def init_sift_verifier(ref_img_paths=None):
    global _sift, _sift_ref_descriptors, _sift_matcher

    if ref_img_paths is None:
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

        for ref_des in _sift_ref_descriptors:
            matches = _sift_matcher.knnMatch(ref_des, des, k=2)

            good_matches = 0
            for m_n in matches:
                if len(m_n) != 2:
                    continue
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good_matches += 1

            if good_matches >= min_matches:
                return True

        return False
    except Exception:
        return False


def load_color_signatures(json_path):
    """Load color signatures JSON into global cache."""
    global _color_signatures
    try:
        with open(json_path, 'r') as f:
            _color_signatures = json.load(f)
        logging.info(f"Loaded color signatures from {json_path}")
        return _color_signatures
    except Exception as e:
        logging.warning(f"Could not load color signatures: {e}")
        _color_signatures = None
        return None


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

def build_red_masks(img, debug_ctx: Optional[dict] = None, skip_wb: bool = False):
    """Create HSV, LAB, and combined red masks; optionally persist for debugging. Honors skip_wb flag."""
    img_proc = img if skip_wb else apply_gray_world(img)
    hsv = cv2.cvtColor(img_proc, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_proc, cv2.COLOR_BGR2LAB)

    h, s, v = cv2.split(hsv)
    sat_mean = np.mean(s)
    sat_threshold = max(45, int(sat_mean * 0.5))
    val_threshold = 25

    lower_red1 = np.array([0, sat_threshold, val_threshold])
    upper_red1 = np.array([12, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    lower_red2 = np.array([168, sat_threshold, val_threshold])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    hsv_mask = cv2.bitwise_or(mask1, mask2)

    l, a, b_channel = cv2.split(lab)
    a_threshold = np.percentile(a, 75)
    a_threshold = max(a_threshold, 115)
    l_threshold = 20
    lab_mask = np.zeros_like(a, dtype=np.uint8)
    lab_mask[(a > a_threshold) & (l > l_threshold)] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel)
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel)
    lab_mask = cv2.morphologyEx(lab_mask, cv2.MORPH_CLOSE, kernel)
    lab_mask = cv2.morphologyEx(lab_mask, cv2.MORPH_OPEN, kernel)

    combined_mask = cv2.bitwise_or(hsv_mask, lab_mask)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    if isinstance(debug_ctx, dict):
        out_dir = debug_ctx.get('output_dir')
        prefix = debug_ctx.get('prefix')
        save_masks = debug_ctx.get('save_masks', False)
        if save_masks and out_dir and prefix:
            try:
                os.makedirs(out_dir, exist_ok=True)
                from utils import save_chip
                save_chip(out_dir, prefix + '.png', 1, hsv_mask, prefix=f"{prefix}_hsv_mask")
                save_chip(out_dir, prefix + '.png', 1, lab_mask, prefix=f"{prefix}_lab_mask")
                save_chip(out_dir, prefix + '.png', 1, combined_mask, prefix=f"{prefix}_combined_mask")
            except Exception:
                pass

    return hsv_mask, lab_mask, combined_mask


def detect_single_scale(img, min_area, max_area, shape_threshold=0.25, debug_ctx: Optional[dict] = None, combined_mask_only=False, skip_wb: bool = False):
    candidates = []
    if debug_ctx and isinstance(debug_ctx, dict) and 'skip_wb' in debug_ctx:
        skip_wb = debug_ctx['skip_wb']

    masks_to_check = []
    if not skip_wb:
        # Generate both WB and non-WB masks
        hsv_mask_wb, lab_mask_wb, combined_mask_wb = build_red_masks(img, debug_ctx=debug_ctx, skip_wb=False)
        hsv_mask, lab_mask, combined_mask = build_red_masks(img, debug_ctx=debug_ctx, skip_wb=True)
        if combined_mask_only:
            masks_to_check.extend([
                (combined_mask_wb, 'combined_wb'),
                (combined_mask, 'combined')
            ])
        else:
            masks_to_check.extend([
                (hsv_mask_wb, 'hsv_wb'),
                (lab_mask_wb, 'lab_wb'),
                (combined_mask_wb, 'combined_wb'),
                (hsv_mask, 'hsv'),
                (lab_mask, 'lab'),
                (combined_mask, 'combined')
            ])
    else:
        # Only non-WB masks, no postfix
        hsv_mask, lab_mask, combined_mask = build_red_masks(img, debug_ctx=debug_ctx, skip_wb=True)
        if combined_mask_only:
            masks_to_check.append((combined_mask, 'combined'))
        else:
            masks_to_check.extend([
                (hsv_mask, 'hsv'),
                (lab_mask, 'lab'),
                (combined_mask, 'combined')
            ])

    contours = []
    for mask, name in masks_to_check:
        found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend([(cnt, name) for cnt in found])

    for contour, source in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        shape_score = calculate_shape_score(contour)
        if shape_score < shape_threshold:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        candidates.append(((x, y, w, h), shape_score, source))

    return candidates


def detect_multiscale(orig_img, scales=[0.3, 0.6, 1., 1.4], debug_ctx: Optional[dict] = None, combined_mask_only=False, skip_wb: bool = False):
    # Detect traffic signs at multiple scales for better small/large sign detection.

    img_height, img_width = orig_img.shape[:2]
    img_area = img_height * img_width

    base_min_area = max(200, int(img_area * 0.0001))
    base_max_area = int(img_area * 0.4)

    all_candidates = []

    # Only override skip_wb if present in debug_ctx
    if debug_ctx and isinstance(debug_ctx, dict) and 'skip_wb' in debug_ctx:
        skip_wb = debug_ctx['skip_wb']
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
        candidates = detect_single_scale(scaled_img, scaled_min_area, scaled_max_area, debug_ctx=debug_ctx, combined_mask_only=combined_mask_only, skip_wb=skip_wb)

        # Scale bounding boxes back to original image coordinates
        for (x, y, w, h), shape_score, source in candidates:
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            orig_w = int(w / scale)
            orig_h = int(h / scale)

            # Store with scale and source information for scoring and grouping
            all_candidates.append(((orig_x, orig_y, orig_w, orig_h), shape_score, scale, source))

    # If no candidates found, return empty list
    if not all_candidates:
        return []

    # Prepare for NMS - we need boxes and scores
    boxes = [bbox for bbox, _, _, _ in all_candidates]
    scores = [shape_score for _, shape_score, _, _ in all_candidates]

    # Apply NMS across all scales with standard threshold
    keep_indices = non_max_suppression(boxes, scores, overlap_thresh=0.4)

    # Return kept detections (without scale info)
    # Include source in final detections by mapping indices back
    sources = [src for _, _, _, src in all_candidates]
    final_detections = [(boxes[i], scores[i], sources[i]) for i in keep_indices]

    return final_detections


def detect_red_regions(img, debug_ctx: Optional[dict] = None):
    _, _, combined_mask = build_red_masks(img, debug_ctx=debug_ctx)
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
         cnn_threshold: float = 0.5, skip_wb: bool = False, skip_histogram: bool = False, validate_bg: bool = False,
         combined_mask_only: bool = False, cnn_extra_confidence: float = 0.9):
    
    import time
    start_time = time.time()
    logging.info(f"Starting traffic sign detection pipeline")
    logging.info(f"Classifier: {classifier_type.upper()}")
    
    directory_path = "data/raw/kaggle_roadsign/images"
    
    # Create classifier-specific output directory
    classifier_suffix = classifier_type.lower()
    results_path = f"data/processed/found_chips_{classifier_suffix}"
    
    os.makedirs(results_path, exist_ok=True)
    logging.info(f"Output directory: {results_path}")

    # Try to load color signatures for histogram-based checks
    color_sig_path = os.path.join('computations', 'color_signatures.json')
    if os.path.exists(color_sig_path):
        load_color_signatures(color_sig_path)
    else:
        logging.warning(f"Color signatures not found at {color_sig_path}; histogram checks disabled")

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
    if classifier_type.lower() == 'ensemble':
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
            # Default to best CNN checkpoint in computations/cnn_checkpoints
            cnn_model_path = os.path.join("computations", "cnn_checkpoints", "best_model.pth")
        
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
            cnn_model_path = os.path.join("computations", "cnn_checkpoints", "best_model.pth")
        
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

    if classifier_type.lower() in ['cnn', 'ensemble']:
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
            label_counts = process_single_image(
                road_sign_image, directory_path, results_path, clf,
                classifier_type=classifier_type, cnn_threshold=cnn_threshold,
                skip_wb=skip_wb, skip_histogram=skip_histogram, validate_bg=validate_bg,
                combined_mask_only=combined_mask_only
            )
            num_detections = label_counts['stop'] + label_counts['other']
            if num_detections > 0:
                detection_stats['images_with_detections'] += 1
                detection_stats['total_detections'] += num_detections
            detection_stats['images_processed'].append({
                'name': road_sign_image,
                'detections': num_detections,
                'detections_by_label': label_counts
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
    
    elapsed = time.time() - start_time
    logging.info(f"Total runtime (seconds): {elapsed:.2f}")
    return detection_stats

def detect_and_classify_frame(orig_img, clf, classifier_type='hog', cnn_threshold=0.85, output_path=None, file_name=None, scales=None, skip_wb: bool = False, skip_histogram: bool = False, validate_bg: bool = False, combined_mask_only: bool = False, cnn_extra_confidence: float = 0.9):
    global debug_shape, draw_hist, debug_mask
    
    # Step 0: No white balance for HOG evaluation
    wb_full_img = orig_img

    # Step 1: Multi-scale detection
    results = []
    bounding_boxes = []
    detection_scores = []
    if scales is None:
        scales = [0.3, 0.6, 1., 1.4]
    debug_ctx = None
    if debug_mask and output_path and file_name:
        debug_dir = os.path.join(output_path, 'debug_masks')
        try:
            os.makedirs(debug_dir, exist_ok=True)
        except Exception:
            pass
        debug_ctx = {
            'output_dir': debug_dir,
            'prefix': os.path.splitext(file_name)[0],
            'save_masks': True,
        }
    candidates = detect_multiscale(orig_img, scales=scales, debug_ctx=debug_ctx, combined_mask_only=combined_mask_only, skip_wb=skip_wb)

    # Step 2: Extract chips
    candidate_sources = []
    for (x, y, w, h), shape_score, source in candidates:
        chip = extract_chip_with_padding(orig_img, x, y, w, h, target_size=HOG_CHIP_SIZE, padding_ratio=0.0, keep_aspect_ratio=False)
        if chip is not None:
            results.append(chip)
            bounding_boxes.append((x, y, w, h))
            detection_scores.append(shape_score)
            candidate_sources.append(source)

    # Step 3: Apply Non-Maximum Suppression
    final_scores = []
    if len(bounding_boxes) > 0:
        keep_indices = non_max_suppression(bounding_boxes, detection_scores, overlap_thresh=0.3)
        results = [results[i] for i in keep_indices]
        bounding_boxes = [bounding_boxes[i] for i in keep_indices]
        final_scores = [detection_scores[i] for i in keep_indices]

    # Step 4: Save candidate chips with overlays (optional)
    if debug_shape and output_path and file_name and len(bounding_boxes) > 0:
        cand_dir = os.path.join(output_path, 'candidates')
        for idx, (chip, score, src) in enumerate(zip(results, final_scores, candidate_sources), start=1):
            chip_annot = chip.copy()
            try:
                cv2.putText(chip_annot, f"shape:{score:.2f}".upper(), (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                if draw_hist:
                    try:
                        bx, by, bw, bh = bounding_boxes[idx-1] if idx-1 < len(bounding_boxes) else (0,0,chip.shape[1], chip.shape[0])
                        chip_wb = extract_chip_with_padding(wb_full_img, bx, by, bw, bh, target_size=128, padding_ratio=0.0, keep_aspect_ratio=False)
                        hist = extract_color_histogram(chip_wb).astype(float)
                        labels = ['R','Y','B','O','W','K']
                        y_off = 24
                        for lab, val in zip(labels, hist[:6]):
                            cv2.putText(chip_annot, f"{lab}:{val:.2f}".upper(), (4, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
                            y_off += 12
                    except Exception:
                        pass
                mask_type = src.split('_')[0] if '_' in src else src
                mask_dir = os.path.join(cand_dir, mask_type)
                os.makedirs(mask_dir, exist_ok=True)
                from utils import save_chip
                save_chip(mask_dir, os.path.splitext(file_name)[0], idx, chip_annot, prefix=f'cand_{src}')
            except Exception:
                pass

    # Step 5: Classify and draw final detections
    final_img, num_detections, detected_boxes, label_counts, detected_labels = test(
        results, bounding_boxes, final_scores, orig_img.copy(), clf, classifier_type, cnn_threshold, output_path, file_name,
        wb_full_img=wb_full_img, skip_histogram=skip_histogram, validate_bg=validate_bg, return_labels=True, cnn_extra_confidence=cnn_extra_confidence)
    # Return only (box, label) pairs
    box_label_pairs = list(zip(detected_boxes, detected_labels))
    return box_label_pairs

def process_single_image(road_sign_image, directory_path, results_path, clf, 
                        classifier_type: str = 'hog', cnn_threshold: float = 0.85,
                        skip_wb: bool = False, skip_histogram: bool = False, validate_bg: bool = False,
                        combined_mask_only: bool = False, cnn_extra_confidence: float = 0.9):

    path = os.path.join(directory_path, road_sign_image)
    orig_img = cv2.imread(path)
    if orig_img is None:
        logging.warning(f"Failed to read image: {road_sign_image}")
        return {'stop': 0, 'other': 0}

    output_path = os.path.join(results_path, os.path.splitext(road_sign_image)[0])
    # Clean previous outputs (chips/candidates/validated) for this image BEFORE any outputs are written
    remove_previous_outputs(output_path)

    # Get list of (box, label) pairs from detection/classification
    box_label_pairs = detect_and_classify_frame(
        orig_img, clf, classifier_type, cnn_threshold, output_path, road_sign_image,
        skip_wb=skip_wb, skip_histogram=skip_histogram, validate_bg=validate_bg,
        combined_mask_only=combined_mask_only, cnn_extra_confidence=cnn_extra_confidence
    )
    if box_label_pairs is None:
        box_label_pairs = []

    # Draw overlays if requested (preserve all debug/overlay options)
    final_img = orig_img.copy()

    # Count labels
    label_counts = {'stop': 0, 'other': 0}
    for box, label in box_label_pairs:
        
        x, y, w, h = box
        color = (0, 255, 0) if label.lower() == 'stop' else (0, 165, 255)
        cv2.rectangle(final_img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(final_img, label.upper(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if label in VALID_LABELS:
            label_counts[label] += 1
        else:
            label_counts['other'] += 1

    cv2.imwrite(os.path.join(output_path, "result.png"), final_img)

    # --- Save debug mask versions and create debug visualization if debug_mask is enabled ---
    if debug_mask:
        debug_dir = os.path.join(output_path, 'debug_masks')
        os.makedirs(debug_dir, exist_ok=True)
        base_prefix = os.path.splitext(road_sign_image)[0]
        from utils import save_chip, save_debug_mask_visualization
        if skip_wb:
            hsv_mask, lab_mask, combined_mask = build_red_masks(orig_img, skip_wb=True)
            save_chip(debug_dir, base_prefix + '.png', 1, hsv_mask, prefix=f"{base_prefix}_hsv_mask")
            save_chip(debug_dir, base_prefix + '.png', 1, lab_mask, prefix=f"{base_prefix}_lab_mask")
            save_chip(debug_dir, base_prefix + '.png', 1, combined_mask, prefix=f"{base_prefix}_combined_mask")
            save_debug_mask_visualization(debug_dir, base_prefix, final_img, hsv_mask, lab_mask, None, None)
        else:
            hsv_mask, lab_mask, combined_mask = build_red_masks(orig_img, skip_wb=True)
            hsv_mask_wb, lab_mask_wb, combined_mask_wb = build_red_masks(orig_img, skip_wb=False)
            save_chip(debug_dir, base_prefix + '.png', 1, hsv_mask, prefix=f"{base_prefix}_hsv_mask")
            save_chip(debug_dir, base_prefix + '.png', 1, lab_mask, prefix=f"{base_prefix}_lab_mask")
            save_chip(debug_dir, base_prefix + '.png', 1, combined_mask, prefix=f"{base_prefix}_combined_mask")
            save_chip(debug_dir, base_prefix + '.png', 1, hsv_mask_wb, prefix=f"{base_prefix}_hsv_mask_wb")
            save_chip(debug_dir, base_prefix + '.png', 1, lab_mask_wb, prefix=f"{base_prefix}_lab_mask_wb")
            save_chip(debug_dir, base_prefix + '.png', 1, combined_mask_wb, prefix=f"{base_prefix}_combined_mask_wb")
            save_debug_mask_visualization(debug_dir, base_prefix, final_img, hsv_mask, lab_mask, hsv_mask_wb, lab_mask_wb)
    return label_counts


def test(results: list, bounding_boxes: list, scores: list, orig_img, clf, classifier_type: str = 'hog', cnn_threshold: float = 0.85, output_path=None, file_name=None, wb_full_img=None, skip_histogram: bool = False, validate_bg: bool = False, return_labels: bool = False, cnn_extra_confidence: float = 0.9):
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
        wb_full_img: White-balanced version of the full image for histogram extraction
        return_labels: If True, return detected_labels as fifth return value
    Returns:
        orig_img: Annotated image
    """
    global debug_shape, draw_hist, debug_mask, debug_cnn_conf
    
    detected_boxes = []
    detected_labels = []
    label_counts = {lbl: 0 for lbl in VALID_LABELS}
    label_counts['other'] = 0
    num_detections = 0

    # cnn_extra_confidence: confidence for definite STOP in ensemble mode (passed as parameter)

    if classifier_type.lower() == 'ensemble':
        try:
            hog_clf, cnn_clf = clf
            features = []
            chips_for_cnn = []
            for result in results:
                if result is not None and result.size != 0 and len(result.shape) == 3 and result.shape[2] == 3 and result.shape[0] >= 8 and result.shape[1] >= 8:
                    chip = cv2.resize(result, (128, 128), interpolation=cv2.INTER_LINEAR)
                    from hog_svm_baseline import extract_improved_features
                    features.append(extract_improved_features(chip))
                    chips_for_cnn.append(result)
                else:
                    features.append(None)
                    chips_for_cnn.append(None)

            # Determine available class labels from color signatures or model if possible
            available_labels = list(_color_signatures.keys()) if _color_signatures else VALID_LABELS + ['bg']

            # CNN predictions: initialize with 'bg'
            default_cnn_label = 'bg'
            cnn_preds = [default_cnn_label] * len(chips_for_cnn)
            cnn_confs = [0.0] * len(chips_for_cnn)
            if any(c is not None for c in chips_for_cnn):
                valid_cnn_idx = [i for i, c in enumerate(chips_for_cnn) if c is not None]
                chips_array = np.array([chips_for_cnn[i] for i in valid_cnn_idx])
                preds, confs = cnn_clf.predict_with_confidence(chips_array)
                for k, i in enumerate(valid_cnn_idx):
                    cnn_preds[i] = preds[k]
                    cnn_confs[i] = confs[k]

            # HOG predictions: initialize with first available label (usually 'bg' or 'unknown')
            default_hog_label = available_labels[0] if available_labels else 'bg'
            hog_predictions = [default_hog_label] * len(features)
            if any(f is not None for f in features):
                valid_hog_idx = [i for i, f in enumerate(features) if f is not None]
                features_array = np.array([features[i] for i in valid_hog_idx])
                if hasattr(hog_clf, 'decision_function'):
                    decision_scores = hog_clf.decision_function(features_array)
                    preds = [VALID_LABELS[0] if score > -0.1 else default_hog_label for score in decision_scores]
                else:
                    preds = hog_clf.predict(features_array)
                for k, i in enumerate(valid_hog_idx):
                    hog_predictions[i] = preds[k]

            for i in range(len(results)):
                chip_hist = None
                hog_label = hog_predictions[i]
                cnn_label = cnn_preds[i]
                cnn_conf = cnn_confs[i]
                hist_valid = True
                matched_label = None
                # Histogram validation for HOG valid label (currently only 'stop')
                if hog_label in VALID_LABELS and not skip_histogram:
                    chip_img = results[i]
                    if chip_img is not None and _color_signatures and hog_label in _color_signatures:
                        chip_hist = extract_color_histogram(chip_img)
                        try:
                            hist_valid = validate_histogram_against_signature(
                                chip_hist,
                                _color_signatures[hog_label],
                                primary_bins=6,
                                std_multiplier=1.5,
                                presence_epsilon=0.01,
                                require_ratio=True,
                            )
                        except Exception:
                            hist_valid = False
                    else:
                        hist_valid = False

                # 1. CNN is very confident for a valid label (matched dynamically; currently only 'stop')
                if cnn_label in VALID_LABELS and cnn_conf >= cnn_extra_confidence:
                    matched_label = cnn_label
                # 2. Both agree on valid label, CNN above threshold, HOG valid, histogram valid, SIFT passes (currently only 'stop')
                elif cnn_label in VALID_LABELS and cnn_conf >= cnn_threshold and cnn_conf < cnn_extra_confidence:
                    if hog_label == cnn_label and hist_valid:
                        try:
                            if verify_with_sift is not None and verify_with_sift(results[i], min_matches=2):
                                matched_label = cnn_label
                        except Exception:
                            pass
                # 3. Validate BG chips if requested (HOG bg, but histogram matches a valid label; currently only 'stop')
                elif hog_label == 'bg' and validate_bg:
                    chip_img = results[i]
                    for valid_lbl in VALID_LABELS:
                        if chip_img is not None and _color_signatures and valid_lbl in _color_signatures:
                            chip_hist = extract_color_histogram(chip_img)
                            try:
                                hist_valid = validate_histogram_against_signature(
                                    chip_hist,
                                    _color_signatures[valid_lbl],
                                    primary_bins=6,
                                    std_multiplier=1.5,
                                    presence_epsilon=0.01,
                                    require_ratio=True,
                                )
                            except Exception:
                                hist_valid = False
                            if hist_valid and cnn_label == valid_lbl and cnn_conf >= cnn_threshold:
                                matched_label = valid_lbl
                                break
                # Only draw/save/count if matched_label is not background
                if matched_label is not None and matched_label != 'bg':
                    x, y, w, h = bounding_boxes[i]
                    num_detections += 1
                    label_counts[matched_label] += 1
                    detected_boxes.append((x, y, w, h))
                    detected_labels.append(matched_label)
                    color = (0, 255, 0)
                    if output_path and file_name:
                        from utils import save_chip
                        save_chip(output_path, os.path.splitext(file_name)[0], num_detections, results[i], label=matched_label.upper())
                    orig_img = cv2.rectangle(orig_img, (x, y), (x + w, y + h), color, 2)
                    label_text = matched_label.upper()
                    if debug_cnn_conf:
                        label_text += f" ({cnn_conf:.2f})"
                    cv2.putText(orig_img, label_text.upper(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    if draw_hist and chip_hist is not None and len(chip_hist) >= 6:
                        draw_histogram_overlay(orig_img, chip_hist, x, y, w, color)
        except Exception as e:
            logging.error(f"Error during ensemble classification: {e}")

        if return_labels:
            return orig_img, num_detections, detected_boxes, label_counts, detected_labels
        return orig_img, num_detections, detected_boxes, label_counts

    elif classifier_type.lower() == 'hog':
        # HOG-only mode: label as 'STOP' if classified as stop and passes hist (or skip_hist), 'OTHER' if classified as stop but fails hist, and label as 'STOP' if bg and validate_bg and passes hist
        try:
            features = []
            for result in results:
                if result is not None and result.size != 0 and len(result.shape) == 3 and result.shape[2] == 3 and result.shape[0] >= 8 and result.shape[1] >= 8:
                    chip = cv2.resize(result, (128, 128), interpolation=cv2.INTER_LINEAR)
                    from hog_svm_baseline import extract_improved_features
                    features.append(extract_improved_features(chip))
                else:
                    features.append(None)

            predictions = ['bg'] * len(features)
            if any(f is not None for f in features):
                valid_indices = [i for i, f in enumerate(features) if f is not None]
                features_array = np.array([features[i] for i in valid_indices])
                if hasattr(clf, 'decision_function'):
                    decision_scores = clf.decision_function(features_array)
                    preds = ['stop' if score > -0.3 else 'bg' for score in decision_scores]
                else:
                    preds = clf.predict(features_array)
                for k, i in enumerate(valid_indices):
                    predictions[i] = preds[k]

            for i in range(len(results)):
                label = None
                should_validate_histogram = False
                # Only process if prediction is a valid label or validate_bg is enabled
                if predictions[i] in VALID_LABELS:
                    label = predictions[i]
                    if not skip_histogram:
                        should_validate_histogram = True
                elif predictions[i] == 'bg' and validate_bg:
                    should_validate_histogram = True
                chip_hist = None
                if should_validate_histogram:
                    chip_img = results[i]
                    if chip_img is not None:
                        chip_hist = extract_color_histogram(chip_img)
                if should_validate_histogram and chip_hist is not None:
                    if _color_signatures and predictions[i] in _color_signatures:
                        try:
                            all_within = validate_histogram_against_signature(
                                chip_hist,
                                _color_signatures[predictions[i]],
                                primary_bins=6,
                                std_multiplier=1.5,
                                presence_epsilon=0.01,
                                require_ratio=True,
                            )
                            if predictions[i] in VALID_LABELS:
                                label = predictions[i] if all_within else 'other'  # Use actual valid label, else other
                            elif predictions[i] == 'bg' and validate_bg:
                                if all_within:
                                    label = predictions[i]
                        except Exception as e:
                            logging.debug(f"Error checking color histogram: {e}")
                            if predictions[i] in VALID_LABELS:
                                label = predictions[i]
                # Only add detection if label is a valid label or 'other' (never for bg or None)
                if label in VALID_LABELS + ['other']:
                    color = (0, 255, 0) if label in VALID_LABELS else (0, 165, 255)
                    num_detections += 1
                    label_key = predictions[i] if label in VALID_LABELS else 'other'
                    label_counts[label_key] += 1
                    detected_labels.append(label)
                    detected_boxes.append(bounding_boxes[i])
                    if output_path and file_name:
                        from utils import save_chip
                        save_chip(output_path, os.path.splitext(file_name)[0], num_detections, results[i], label=label)
                    orig_img = cv2.rectangle(orig_img, (bounding_boxes[i][0], bounding_boxes[i][1]), (bounding_boxes[i][0] + bounding_boxes[i][2], bounding_boxes[i][1] + bounding_boxes[i][3]), color, 2)
                    cv2.putText(orig_img, label.upper(), (bounding_boxes[i][0], bounding_boxes[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    if draw_hist and chip_hist is not None and len(chip_hist) >= 6:
                        draw_histogram_overlay(orig_img, chip_hist, bounding_boxes[i][0], bounding_boxes[i][1], bounding_boxes[i][2], color)
        except Exception as e:
            logging.error(f"Error during HOG classification: {e}")
        if return_labels:
            return orig_img, num_detections, detected_boxes, label_counts, detected_labels
        return orig_img, num_detections, detected_boxes, label_counts

    elif classifier_type.lower() == 'cnn':
        # CNN-only mode:
        try:
            chips_for_cnn = []
            for result in results:
                if result is not None and result.size != 0 and len(result.shape) == 3 and result.shape[2] == 3 and result.shape[0] >= 8 and result.shape[1] >= 8:
                    chips_for_cnn.append(result)
                else:
                    chips_for_cnn.append(None)

            predictions = ['bg'] * len(chips_for_cnn)
            confs = [0.0] * len(chips_for_cnn)
            if any(c is not None for c in chips_for_cnn):
                valid_cnn_idx = [i for i, c in enumerate(chips_for_cnn) if c is not None]
                chips_array = np.array([chips_for_cnn[i] for i in valid_cnn_idx])
                preds, cnn_confs = clf.predict_with_confidence(chips_array)
                for k, i in enumerate(valid_cnn_idx):
                    predictions[i] = preds[k]
                    confs[i] = cnn_confs[k]

            for i in range(len(results)):
                label = predictions[i] if confs[i] >= cnn_threshold else 'bg'
                if label == 'bg':
                    continue  # Skip drawing/saving for bg
                num_detections += 1
                label_counts[label] += 1
                detected_labels.append(label)
                detected_boxes.append(bounding_boxes[i])
                if debug_cnn_conf:
                    orig_img =  cv2.putText(orig_img, str(confs[i]).upper(), (bounding_boxes[i][0], bounding_boxes[i][1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if output_path and file_name:
                    from utils import save_chip
                    save_chip(output_path, os.path.splitext(file_name)[0], num_detections, results[i], label=label)
        except Exception as e:
            logging.error(f"Error during CNN classification: {e}")

    # Ensure that for all other classifier types, we always return the correct tuple
    if return_labels:
        return orig_img, num_detections, detected_boxes, label_counts, detected_labels
    return orig_img, num_detections, detected_boxes, label_counts


    


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Traffic sign detection using color, shape, and ML classifiers')
    parser.add_argument('--classifier', type=str, choices=['hog', 'cnn', 'ensemble'], default='hog',
                        help='Type of classifier to use: hog (HOG-SVM), cnn (CNN), or ensemble (hybrid logic). Default: hog')
    parser.add_argument('--cnn-model', type=str, default=None,
                        help='Path to CNN model checkpoint (.pth file) - required if using CNN or ensemble classifier')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Minimum confidence for CNN to consider a detection (default: 0.5)')
    parser.add_argument('--cnn-extra-confidence', type=float, default=0.9,
                        help='CNN confidence required for automatic STOP in ensemble mode (default: 0.9)')
    parser.add_argument('--debug-shape-candidates', action='store_true', help='Save candidate chips with shape scores overlay to candidates/ per image')
    parser.add_argument('--debug-cnn-conf', action='store_true', help='Draw CNN confidence scores on result images (if using CNN or ensemble)')
    parser.add_argument('--draw-hist', action='store_true', help='Draw color histogram values (including Orange) on result images and candidate overlays')
    parser.add_argument('--debug-mask', action='store_true', help='Save processed images after each mask (HSV, LAB, combined)')
    parser.add_argument('--skip-wb', action='store_true', help='Skip white balance preprocessing (default: apply WB)')
    parser.add_argument('--skip-histogram', action='store_true', help='Skip histogram validation gating (default: apply histogram validation)')
    parser.add_argument('--combined-mask-only', action='store_true', help='Use only the combined mask for detection (default: HSV, LAB, and combined masks are all used independently)')
    parser.add_argument('--validate-bg', action='store_true', help='Apply histogram validation to background-classified chips to find additional stops (default: off)')
    parser.add_argument('--images', nargs='+', default=None,
                        help='Process only the specified image file names (e.g., road55.png road112.png). If omitted, processes all images.')
    

    args = parser.parse_args()
    # Set global debug flags from argparse
    debug_shape = args.debug_shape_candidates
    draw_hist = args.draw_hist
    debug_mask = args.debug_mask
    debug_cnn_conf = args.debug_cnn_conf

    import time
    if args.images:
        start_time = time.time()
        directory_path = "data/raw/kaggle_roadsign/images"
        classifier_type = args.classifier
        results_path = f"data/processed/found_chips_{classifier_type.lower()}"
        os.makedirs(results_path, exist_ok=True)

        color_sig_path = os.path.join('computations', 'color_signatures.json')
        if os.path.exists(color_sig_path):
            load_color_signatures(color_sig_path)

        if classifier_type in ['cnn', 'ensemble']:
            if args.cnn_model is None:
                args.cnn_model = os.path.join("computations", "cnn_checkpoints", "best_model.pth")
            if not os.path.exists(args.cnn_model):
                print(f"Error: CNN model not found at {args.cnn_model}")
                exit(1)
        clf = None
        if classifier_type == 'ensemble':
            hog_path = os.path.join("computations", "hog_svm_stop_and_bg.pkl")
            hog_clf = joblib.load(hog_path)
            cnn_clf = CNNClassifier(model_path=args.cnn_model, input_size=224)
            cnn_clf.threshold = args.threshold
            clf = (hog_clf, cnn_clf)
            init_sift_verifier()
        elif classifier_type == 'cnn':
            clf = CNNClassifier(model_path=args.cnn_model, input_size=224)
            clf.threshold = args.threshold
            init_sift_verifier()
        else:
            computations_path = os.path.join("computations", "hog_svm_stop_and_bg.pkl")
            clf = joblib.load(computations_path)

        # Process only provided images
        detection_stats = {
            'total_images': len(args.images),
            'images_with_detections': 0,
            'total_detections': 0,
            'images_processed': []
        }
        for img_name in args.images:
            try:
                label_counts = process_single_image(
                    img_name, directory_path, results_path, clf,
                    classifier_type=classifier_type, cnn_threshold=args.threshold,
                    skip_wb=args.skip_wb, skip_histogram=args.skip_histogram, validate_bg=args.validate_bg,
                    combined_mask_only=args.combined_mask_only,
                    cnn_extra_confidence=args.cnn_extra_confidence
                )
                num_detections = label_counts['stop'] + label_counts['other']
                if num_detections > 0:
                    detection_stats['images_with_detections'] += 1
                    detection_stats['total_detections'] += num_detections
                detection_stats['images_processed'].append({
                    'name': img_name,
                    'detections': num_detections,
                    'detections_by_label': label_counts
                })
            except Exception as e:
                logging.warning(f"Error processing {img_name}: {e}")
                continue
        stats_file = os.path.join(results_path, f'detection_stats_{classifier_type}.json')
        with open(stats_file, 'w') as f:
            json.dump(detection_stats, f, indent=2)
        logging.info("Detection pipeline complete (selected images)!")
        logging.info(f"Total detections: {detection_stats['total_detections']}")
        logging.info(f"Images with detections: {detection_stats['images_with_detections']}/{len(args.images)}")
        logging.info(f"Statistics saved to: {stats_file}")
        elapsed = time.time() - start_time
        logging.info(f"Total runtime (seconds): {elapsed:.2f}")
    else:
        # Route to main based on classifier type
        if args.classifier in ['cnn', 'ensemble']:
            if args.cnn_model is None or not os.path.exists(args.cnn_model):
                print(f"Error: --cnn-model is required and must exist when using {args.classifier} classifier")
                exit(1)
        # Call main() for all images in the dataset
        main(
            classifier_type=args.classifier,
            cnn_model_path=args.cnn_model,
            cnn_threshold=args.threshold,
            skip_wb=args.skip_wb,
            skip_histogram=args.skip_histogram,
            validate_bg=args.validate_bg,
            combined_mask_only=args.combined_mask_only,
            cnn_extra_confidence=args.cnn_extra_confidence
        )
