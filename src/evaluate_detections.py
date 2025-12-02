import os
import json
import csv
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def load_ground_truth(csv_path: str, positive_labels: Tuple[str, ...] = ("stop",)) -> Tuple[Dict[str, List[Tuple[int, int, int, int]]], Dict[str, List[Tuple[int, int, int, int]]]]:
    """Load ground truth annotations from CSV, separating positive and other labels.

    Returns:
        positive_gt: Dict mapping image_name -> list of bboxes for positive labels (e.g., stop)
        other_gt: Dict mapping image_name -> list of bboxes for other (non-bg) labels
    """
    positive_gt = {}
    other_gt = {}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = (row.get('label') or '').strip().lower()
            if label == 'bg' or not label:
                continue

            img_path = row['img_path']
            img_name = os.path.basename(img_path)

            # Parse bounding box
            bbox = (
                int(row['xmin']),
                int(row['ymin']),
                int(row['xmax']),
                int(row['ymax'])
            )

            if label in positive_labels:
                if img_name not in positive_gt:
                    positive_gt[img_name] = []
                positive_gt[img_name].append(bbox)
            else:
                # Non-positive, non-bg label (e.g., other signs)
                if img_name not in other_gt:
                    other_gt[img_name] = []
                other_gt[img_name].append(bbox)

    return positive_gt, other_gt


def calculate_iou(box1: Tuple[int, int, int, int], 
                  box2: Tuple[int, int, int, int]) -> float:
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


def count_detections_in_image(results_dir: str, img_name: str) -> int:
    img_base = os.path.splitext(img_name)[0]
    img_dir = os.path.join(results_dir, img_base)
    
    if not os.path.exists(img_dir):
        return 0
    
    # Count PNG files (excluding result.png)
    count = 0
    for file in os.listdir(img_dir):
        if file.endswith('.png') and file != 'result.png':
            count += 1
    
    return count


def evaluate_detections(positive_gt: Dict[str, List], 
                       other_gt: Dict[str, List],
                       stats_path: str,
                       results_dir: str = None) -> Dict:
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives_other = 0  # 'other' detections matching non-stop GT
    
    # Scope GT to only evaluated images to avoid mismatched totals
    all_images = set(item['name'] for item in stats['images_processed'])
    scoped_positive_gt = {img: boxes for img, boxes in positive_gt.items() if img in all_images}
    scoped_other_gt = {img: boxes for img, boxes in other_gt.items() if img in all_images}
    images_with_gt = set(scoped_positive_gt.keys())
    total_gt_signs = sum(len(boxes) for boxes in scoped_positive_gt.values())
    
    image_results = []
    
    # Create directories for TP/FP/FN if results_dir provided
    if results_dir:
        tp_dir = os.path.join(results_dir, 'true_positives')
        fp_dir = os.path.join(results_dir, 'false_positives')
        fn_dir = os.path.join(results_dir, 'false_negatives')
        
        # Clear existing directories
        for dir_path in [tp_dir, fp_dir, fn_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path, exist_ok=True)
    
    for img_name in all_images:
        gt_boxes = scoped_positive_gt.get(img_name, [])
        other_boxes = scoped_other_gt.get(img_name, [])
        
        # Count detections by label
        num_stop_detections = 0
        num_other_detections = 0
        
        for item in stats['images_processed']:
            if item['name'] == img_name:
                # Check if stats include label breakdown
                if 'detections_by_label' in item:
                    num_stop_detections = item['detections_by_label'].get('stop', 0)
                    num_other_detections = item['detections_by_label'].get('other', 0)
                else:
                    # Fallback: assume all detections are 'stop'
                    num_stop_detections = item['detections']
                break
        
        # Evaluate stop sign detections
        has_gt_stop = len(gt_boxes) > 0
        has_stop_detection = num_stop_detections > 0
        
        # Get chip paths for this image
        img_base = os.path.splitext(img_name)[0]
        img_dir = os.path.join(results_dir, img_base) if results_dir else None
        
        if has_gt_stop and has_stop_detection:
            # True positives
            tp_count = min(num_stop_detections, len(gt_boxes))
            true_positives += tp_count
            
            # Copy result.png to TP directory
            if img_dir and os.path.exists(img_dir):
                result_img = os.path.join(img_dir, 'result.png')
                if os.path.exists(result_img):
                    dst = os.path.join(tp_dir, f'{img_base}_result.png')
                    shutil.copy2(result_img, dst)
            
            # False positives (extra detections)
            if num_stop_detections > len(gt_boxes):
                false_positives += (num_stop_detections - len(gt_boxes))
                
        elif has_gt_stop and not has_stop_detection:
            # False negatives - missed all GT stop signs
            false_negatives += len(gt_boxes)
            
            # Copy result.png to FN directory
            if img_dir and os.path.exists(img_dir):
                result_img = os.path.join(img_dir, 'result.png')
                if os.path.exists(result_img):
                    dst = os.path.join(fn_dir, f'{img_base}_result.png')
                    shutil.copy2(result_img, dst)
                    
        elif not has_gt_stop and has_stop_detection:
            # False positives - detections where there are no GT stop signs
            false_positives += num_stop_detections
            
            # Copy result.png to FP directory
            if img_dir and os.path.exists(img_dir):
                result_img = os.path.join(img_dir, 'result.png')
                if os.path.exists(result_img):
                    dst = os.path.join(fp_dir, f'{img_base}_result.png')
                    shutil.copy2(result_img, dst)
        
        # Evaluate 'other' detections
        has_gt_other = len(other_boxes) > 0
        if has_gt_other and num_other_detections > 0:
            # 'other' detections matching non-stop GT signs = true negatives (correct rejection of non-stop)
            true_negatives_other += min(num_other_detections, len(other_boxes))
        elif not has_gt_other and num_other_detections > 0:
            # 'other' detections where there are no other signs = acceptable (rejected as non-stop)
            pass  # Not counted as FP
        
        image_results.append({
            'image': img_name,
            'gt_stop_count': len(gt_boxes),
            'gt_other_count': len(other_boxes),
            'detected_stop': num_stop_detections,
            'detected_other': num_other_detections,
            'has_gt_stop': has_gt_stop,
            'has_stop_detection': has_stop_detection
        })
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # False positive rate
    images_without_gt = len(all_images) - len(images_with_gt)
    fpr = false_positives / stats['total_detections'] if stats['total_detections'] > 0 else 0
    
    images_without_gt = len(all_images) - len(images_with_gt)
    
    results = {
        'total_images': len(all_images),
        'images_with_gt_stop_signs': len(images_with_gt),
        'images_without_gt_stop_signs': images_without_gt,
        'total_gt_stop_signs': total_gt_signs,
        'total_gt_other_signs': sum(len(boxes) for boxes in scoped_other_gt.values()),
        'total_detections': stats['total_detections'],
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_negatives_other': true_negatives_other,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'false_positive_rate': fpr,
        'images_with_false_positives': len([r for r in image_results if not r['has_gt_stop'] and r['detected_stop'] > 0]),
        'avg_detections_per_image': stats['total_detections'] / len(all_images),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate detection performance against ground truth')
    parser.add_argument('--annotations', type=str, 
                       default='data/annotations/labels.csv',
                       help='Path to ground truth annotations CSV')
    parser.add_argument('--hog-dir', type=str, 
                       default='data/processed/found_chips_hog',
                       help='Directory containing HOG-SVM results')
    parser.add_argument('--cnn-dir', type=str, 
                       default='data/processed/found_chips_cnn',
                       help='Directory containing CNN results')
    parser.add_argument('--ensemble-dir', type=str,
                       default='data/processed/found_chips_ensemble',
                       help='Directory containing ensemble results')
    parser.add_argument('--output', type=str,
                       default='data/processed/evaluation_results.json',
                       help='Output path for evaluation results')
    parser.add_argument('--positive-labels', type=str, default='stop',
                        help='Comma-separated list of labels considered positives (default: stop). Others are treated as negatives.')
    
    args = parser.parse_args()
    positive_labels = tuple([s.strip().lower() for s in args.positive_labels.split(',') if s.strip()]) or ("stop",)
    positive_gt, other_gt = load_ground_truth(args.annotations, positive_labels=positive_labels)

    results = {}
    
    # Evaluate HOG-SVM
    hog_stats_path = os.path.join(args.hog_dir, 'detection_stats_hog.json')
    if os.path.exists(hog_stats_path):
        logging.info("Evaluating HOG-SVM detections...")
        results['hog_svm'] = evaluate_detections(positive_gt, other_gt, hog_stats_path, args.hog_dir)
    else:
        logging.warning(f"HOG-SVM stats not found at {hog_stats_path}")
    
    # Evaluate CNN
    cnn_stats_path = os.path.join(args.cnn_dir, 'detection_stats_cnn.json')
    if os.path.exists(cnn_stats_path):
        logging.info("Evaluating CNN detections...")
        results['cnn'] = evaluate_detections(positive_gt, other_gt, cnn_stats_path, args.cnn_dir)
    else:
        logging.warning(f"CNN stats not found at {cnn_stats_path}")
    
    # Evaluate Ensemble
    ensemble_stats_path = os.path.join(args.ensemble_dir, 'detection_stats_ensemble.json')
    if os.path.exists(ensemble_stats_path):
        logging.info("Evaluating Ensemble detections...")
        results['ensemble'] = evaluate_detections(positive_gt, other_gt, ensemble_stats_path, args.ensemble_dir)
    else:
        logging.info(f"Ensemble stats not found (optional) at {ensemble_stats_path}")
    
    # Print results
    logging.info("="*70)
    logging.info("DETECTION EVALUATION RESULTS")
    logging.info("="*70)
    
    for classifier_name, metrics in results.items():
        logging.info(f"\n{classifier_name.upper()}:")
        logging.info(f"  Dataset Statistics:")
        logging.info(f"    Total images: {metrics['total_images']}")
        logging.info(f"    Images with GT stop signs: {metrics['images_with_gt_stop_signs']}")
        logging.info(f"    Images without GT stop signs: {metrics['images_without_gt_stop_signs']}")
        logging.info(f"    Total GT stop signs: {metrics['total_gt_stop_signs']}")
        logging.info(f"    Total GT other signs: {metrics['total_gt_other_signs']}")
        logging.info(f"  Detection Statistics:")
        logging.info(f"    Total detections: {metrics['total_detections']}")
        logging.info(f"    True Positives (TP): {metrics['true_positives']}")
        logging.info(f"    False Positives (FP): {metrics['false_positives']}")
        logging.info(f"    False Negatives (FN): {metrics['false_negatives']}")
        logging.info(f"    True Negatives - Other matched: {metrics['true_negatives_other']}")
        logging.info(f"    Images with false positives: {metrics['images_with_false_positives']}")
        logging.info(f"  Performance Metrics:")
        logging.info(f"    Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
        logging.info(f"    Recall: {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
        logging.info(f"    F1-Score: {metrics['f1_score']:.3f}")
        logging.info(f"    False Positive Rate: {metrics['false_positive_rate']:.3f} ({metrics['false_positive_rate']*100:.1f}% of detections)")
        logging.info(f"    Avg detections per image: {metrics['avg_detections_per_image']:.2f}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to: {output_path}")


if __name__ == '__main__':
    main()
