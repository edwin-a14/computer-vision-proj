import cv2
import numpy as np

class LaneDetector:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # ROI: Trapezoid at bottom of screen
        # (Bottom-Left, Top-Left, Top-Right, Bottom-Right)
        self.roi_vertices = np.array([[
            (int(width * 0.1), height),
            (int(width * 0.45), int(height * 0.6)),
            (int(width * 0.55), int(height * 0.6)),
            (int(width * 0.9), height)
        ]], dtype=np.int32)

    def region_of_interest(self, img):
        mask = np.zeros_like(img)
        match_mask_color = 255
        cv2.fillPoly(mask, self.roi_vertices, match_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def detect_lanes(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, 50, 150)
        cropped = self.region_of_interest(canny)
        
        lines = cv2.HoughLinesP(cropped,
                                rho=2,
                                theta=np.pi/180,
                                threshold=50,
                                lines=np.array([]),
                                minLineLength=40,
                                maxLineGap=100)
        
        line_image = np.zeros_like(frame)
        
        if lines is not None:
            self.draw_lines(line_image, lines)
            
        # Overlay on original frame
        # Weighted add: frame * 1.0 + line_image * 1.0
        combo_image = cv2.addWeighted(frame, 1.0, line_image, 1.0, 0)
        return combo_image

    def draw_lines(self, img, lines, color=[0, 255, 255], thickness=5):
        left_lines = []
        right_lines = []
        
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 == x1: continue # Avoid division by zero
                slope = (y2 - y1) / (x2 - x1)
                
                # Filter horizontal lines (slope close to 0)
                if abs(slope) < 0.5: continue
                
                # Left lane: negative slope in image coordinates
                # Right lane: positive slope
                if slope < 0:
                    left_lines.append((slope, y1 - slope * x1))
                else:
                    right_lines.append((slope, y1 - slope * x1))
        
        y_global_min = int(self.height * 0.65)
        y_max = self.height
        
        left_points = None
        right_points = None

        if left_lines:
            left_avg = np.average(left_lines, axis=0)
            slope, intercept = left_avg
            try:
                x1 = int((y_max - intercept) / slope)
                x2 = int((y_global_min - intercept) / slope)
                left_points = ((x1, y_max), (x2, y_global_min))
                cv2.line(img, (x1, y_max), (x2, y_global_min), color, thickness)
            except OverflowError:
                pass

        # Draw Right Lane
        if right_lines:
            right_avg = np.average(right_lines, axis=0)
            slope, intercept = right_avg
            try:
                x1 = int((y_max - intercept) / slope)
                x2 = int((y_global_min - intercept) / slope)
                right_points = ((x1, y_max), (x2, y_global_min))
                cv2.line(img, (x1, y_max), (x2, y_global_min), color, thickness)
            except OverflowError:
                pass
        
        # Draw polygon between lanes
        if left_points and right_points:
            # Points: Bottom-Left, Top-Left, Top-Right, Bottom-Right
            pts = np.array([
                left_points[0],
                left_points[1],
                right_points[1],
                right_points[0]
            ], dtype=np.int32)
            
            cv2.fillPoly(img, [pts], (255, 0, 0))
