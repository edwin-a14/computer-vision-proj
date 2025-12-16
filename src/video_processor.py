import cv2
import numpy as np
from speed_estimation import SpeedEstimator
from lane_detection import LaneDetector
from detect_color_shape import detect_and_classify_frame
from action_inference import determine_driver_action

class VideoProcessor:
    def __init__(self, width=640, height=480, fps=30, classifier_type='hog'):
        self.width = width
        self.height = height
        self.fps = fps
        self.classifier_type = classifier_type
        
        self.speed_est = SpeedEstimator(width, height, fps)
        self.lane_det = LaneDetector(width, height)
        
        self.frame_count = 0
        self.trackers = []
        self.last_detected_boxes = []
        self.last_action = "Unknown"
        self.last_action_color = (0, 255, 0)
        self.video_scales = [0.4, 0.7, 1.0, 1.3]
        self.frame_skip = 3

    def process_frame(self, frame, clf):
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            self.width = w
            self.height = h
            self.speed_est = SpeedEstimator(w, h, self.fps)
            self.lane_det = LaneDetector(w, h)
        
        self.speed_est.estimate_speed(frame)
        
        processed_frame = self.lane_det.detect_lanes(frame)
        
        detections = []
        num_dets = 0
        
        if self.frame_count % self.frame_skip == 0:
            _, num_dets, detections = detect_and_classify_frame(
                frame, clf, classifier_type=self.classifier_type, 
                cnn_threshold=0.50, scales=self.video_scales
            )
            
            self.trackers = []
            for (x, y, w, h) in detections:
                tracker = cv2.TrackerMIL_create()
                tracker.init(frame, (x, y, w, h))
                self.trackers.append(tracker)
            
            action, color = determine_driver_action(detections, self.width, self.height)
            
            if detections:
                largest_det = max(detections, key=lambda b: b[2] * b[3])
                self.speed_est.process_stop_sign(largest_det, self.frame_count)
            
            self.last_detected_boxes = detections
            self.last_action = action
            self.last_action_color = color
            
        else:
            updated_boxes = []
            for tracker in self.trackers:
                success, box = tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in box]
                    updated_boxes.append((x, y, w, h))
            
            if not updated_boxes and self.last_detected_boxes:
                 updated_boxes = self.last_detected_boxes
            
            detections = updated_boxes
            num_dets = len(detections)

        for (x, y, w, h) in detections:
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(processed_frame, "Stop", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        cv2.putText(processed_frame, f"ACTION: {self.last_action}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.last_action_color, 2)
        
        self.speed_est.draw_speed(processed_frame)
        
        self.frame_count += 1
        return processed_frame, num_dets, detections
