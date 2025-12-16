import os
import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def load_ground_truth(csv_path: str) -> Dict[str, List[Tuple[int, int, int, int]]]:
    # Load ground truth annotations from CSV.
    ground_truth = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = row['img_path']
            img_name = os.path.basename(img_path)
            
            # Parse bounding box
            bbox = (
                int(row['xmin']),
                int(row['ymin']),
                int(row['xmax']),
                int(row['ymax'])
            )
            
            if img_name not in ground_truth:
                ground_truth[img_name] = []
            ground_truth[img_name].append(bbox)
    
    return ground_truth


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


def evaluate_detections(ground_truth: Dict[str, List], 
                       stats_path: str) -> Dict:
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    images_with_gt = set(ground_truth.keys())
    total_gt_signs = sum(len(boxes) for boxes in ground_truth.values())
    
    image_results = []
    
    all_images = set()
    for item in stats['images_processed']:
        all_images.add(item['name'])
    
    for img_name in all_images:
        gt_boxes = ground_truth.get(img_name, [])
        num_detections = 0
        
        # Count detections from saved chips
        for item in stats['images_processed']:
            if item['name'] == img_name:
                num_detections = item['detections']
                break
        
        # For now, use a simple heuristic:
        # - If image has GT stop signs and we detected something: potential TP
        # - If image has GT stop signs and we detected nothing: FN
        # - If image has no GT stop signs and we detected something: FP
        # - If image has no GT stop signs and we detected nothing: TN
        
        has_gt = len(gt_boxes) > 0
        has_detection = num_detections > 0
        
        if has_gt and has_detection:
            # Assume detections are correct for images with GT
            # (Conservative estimate - actual TP might be lower)
            true_positives += min(num_detections, len(gt_boxes))
            if num_detections > len(gt_boxes):
                false_positives += (num_detections - len(gt_boxes))
        elif has_gt and not has_detection:
            # Missed all GT signs
            false_negatives += len(gt_boxes)
        elif not has_gt and has_detection:
            # All detections are false positives
            false_positives += num_detections
        # else: TN (no GT, no detection) - not counted
        
        image_results.append({
            'image': img_name,
            'gt_count': len(gt_boxes),
            'detected_count': num_detections,
            'has_gt': has_gt,
            'has_detection': has_detection
        })
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # False positive rate
    images_without_gt = len(all_images) - len(images_with_gt)
    fpr = false_positives / stats['total_detections'] if stats['total_detections'] > 0 else 0
    
    results = {
        'total_images': len(all_images),
        'images_with_gt_signs': len(images_with_gt),
        'images_without_gt_signs': images_without_gt,
        'total_gt_signs': total_gt_signs,
        'total_detections': stats['total_detections'],
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'false_positive_rate': fpr,
        'images_with_false_positives': len([r for r in image_results if not r['has_gt'] and r['has_detection']]),
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
    
    args = parser.parse_args()    
    ground_truth = load_ground_truth(args.annotations)

    results = {}
    
    # Evaluate HOG-SVM
    hog_stats_path = os.path.join(args.hog_dir, 'detection_stats_hog.json')
    if os.path.exists(hog_stats_path):
        logging.info("\nEvaluating HOG-SVM detections...")
        results['hog_svm'] = evaluate_detections(ground_truth, hog_stats_path)
    else:
        logging.warning(f"HOG-SVM stats not found at {hog_stats_path}")
    
    # Evaluate CNN
    cnn_stats_path = os.path.join(args.cnn_dir, 'detection_stats_cnn.json')
    if os.path.exists(cnn_stats_path):
        logging.info("\nEvaluating CNN detections...")
        results['cnn'] = evaluate_detections(ground_truth, cnn_stats_path)
    else:
        logging.warning(f"CNN stats not found at {cnn_stats_path}")
    
    # Evaluate Ensemble
    ensemble_stats_path = os.path.join(args.ensemble_dir, 'detection_stats_ensemble.json')
    if os.path.exists(ensemble_stats_path):
        logging.info("\nEvaluating Ensemble detections...")
        results['ensemble'] = evaluate_detections(ground_truth, ensemble_stats_path)
    else:
        logging.info(f"Ensemble stats not found (optional) at {ensemble_stats_path}")
    
    print("\n" + "="*70)
    print("DETECTION EVALUATION RESULTS")
    print("="*70)
    
    for classifier_name, metrics in results.items():
        print(f"\n{classifier_name.upper()}:")
        print(f"  Dataset Statistics:")
        print(f"    Total images: {metrics['total_images']}")
        print(f"    Images with GT stop signs: {metrics['images_with_gt_signs']}")
        print(f"    Images without GT stop signs: {metrics['images_without_gt_signs']}")
        print(f"    Total GT stop signs: {metrics['total_gt_signs']}")
        print(f"  Detection Statistics:")
        print(f"    Total detections: {metrics['total_detections']}")
        print(f"    True Positives (TP): {metrics['true_positives']}")
        print(f"    False Positives (FP): {metrics['false_positives']}")
        print(f"    False Negatives (FN): {metrics['false_negatives']}")
        print(f"    Images with false positives: {metrics['images_with_false_positives']}")
        print(f"  Performance Metrics:")
        print(f"    Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
        print(f"    Recall: {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
        print(f"    F1-Score: {metrics['f1_score']:.3f}")
        print(f"    False Positive Rate: {metrics['false_positive_rate']:.3f} ({metrics['false_positive_rate']*100:.1f}% of detections)")
        print(f"    Avg detections per image: {metrics['avg_detections_per_image']:.2f}")

    def fmt_pct(x):
        try:
            return f"{x*100:.1f}%"
        except Exception:
            return "-"

    headers = [
        "Classifier", "TP", "FP", "FN", "Precision", "Recall", "F1",
        "Detections", "FP Rate", "Avg/Img"
    ]
    col_widths = [12, 6, 6, 6, 10, 10, 8, 11, 10, 8]

    def row_line(cols, widths):
        return " | ".join(str(c).ljust(w) for c, w in zip(cols, widths))

    print("\n" + "-" * (sum(col_widths) + 3 * (len(col_widths) - 1)))
    print(row_line(headers, col_widths))
    print("-" * (sum(col_widths) + 3 * (len(col_widths) - 1)))

    for name in ["hog_svm", "cnn", "ensemble"]:
        if name in results:
            m = results[name]
            row = [
                name.upper(),
                m["true_positives"],
                m["false_positives"],
                m["false_negatives"],
                f"{m['precision']:.3f}",
                f"{m['recall']:.3f}",
                f"{m['f1_score']:.3f}",
                m["total_detections"],
                f"{m["false_positive_rate"]*100:.1f}%",
                f"{m['avg_detections_per_image']:.2f}"
            ]
            print(row_line(row, col_widths))

            print("-" * (sum(col_widths) + 3 * (len(col_widths) - 1)))
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
