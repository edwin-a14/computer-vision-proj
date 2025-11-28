import cv2
import numpy as np

class SpeedEstimator:
    def __init__(self, width, height, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        
        # Region of Interest for speed estimation (focus on the road)
        # Bottom 40% of the screen, centered horizontally
        self.roi_top = int(height * 0.6)
        self.roi_bottom = int(height * 0.9)
        self.roi_left = int(width * 0.2)
        self.roi_right = int(width * 0.8)
        
        self.prev_gray = None
        self.prev_points = None
        
        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Parameters for feature detection
        self.feature_params = dict(maxCorners=100,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)
        
        self.current_speed = 0.0
        self.alpha = 0.1
        self.current_flow = 0.0
        
        # Calibration factor (pixels/frame -> mph)
        self.mph_per_pixel = 0.03 
        
        self.last_stop_width = None
        self.last_stop_frame = None
        self.stop_sign_real_width = 0.00047348 # 30 inches in miles

        # Assume 90 degree FOV for focal length estimation
        # f = (w/2) / tan(FOV/2). tan(45) = 1.
        self.focal_length = width / 2.0

    def estimate_speed(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        mask = np.zeros_like(gray)
        mask[self.roi_top:self.roi_bottom, self.roi_left:self.roi_right] = 255

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
            return 0.0

        if self.prev_points is None or len(self.prev_points) < 10:
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
            self.prev_gray = gray
            return self.current_speed

        next_points, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_points, None, **self.lk_params)
        
        if next_points is not None:
            good_new = next_points[status == 1]
            good_old = self.prev_points[status == 1]
            
            magnitudes = []
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                
                dist = np.sqrt((a - c)**2 + (b - d)**2)
                
                # Filter out small movements (noise) and extremely large ones (errors)
                if 2 < dist < 50:
                    magnitudes.append(dist)
            
            if magnitudes:
                avg_flow = np.median(magnitudes)
                self.current_flow = avg_flow
                
                # Speed = Flow * FPS * Calibration
                estimated_speed_raw = avg_flow * self.fps * self.mph_per_pixel
                self.current_speed = (self.alpha * estimated_speed_raw) + ((1 - self.alpha) * self.current_speed)
            else:
                self.current_flow = 0.0

                # Decay speed towards 0 when no motion is detected
                self.current_speed = (1 - self.alpha) * self.current_speed
                if self.current_speed < 0.5:
                    self.current_speed = 0.0
            
            self.prev_gray = gray.copy()
            self.prev_points = good_new.reshape(-1, 1, 2)
            
        else:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
            
        return self.current_speed

    def process_stop_sign(self, bbox, frame_idx):
        x, y, w, h = bbox
        
        # Z = f * W_real / w_pixel
        distance = (self.focal_length * self.stop_sign_real_width) / w
        
        if self.last_stop_width is not None and self.last_stop_frame is not None:
            dt_frames = frame_idx - self.last_stop_frame
            if 0 < dt_frames < 10:
                dt_seconds = dt_frames / self.fps
                
                prev_distance = (self.focal_length * self.stop_sign_real_width) / self.last_stop_width
                
                # Speed = delta_distance / delta_time
                speed_miles_per_sec = (prev_distance - distance) / dt_seconds
                
                if speed_miles_per_sec > 0 and self.current_flow > 0:
                    speed_mph = speed_miles_per_sec * 3600 # miles per second to miles per hour
                    
                    # Calculate new calibration factor
                    # speed_mph = flow * fps * factor
                    # factor = speed_mph / (flow * fps)
                    new_factor = speed_mph / (self.current_flow * self.fps)
                    
                    # Sanity check for factor (shouldn't be wildly different)
                    if 0.01 < new_factor < 0.2:
                        self.mph_per_pixel = (0.9 * self.mph_per_pixel) + (0.1 * new_factor)

        self.last_stop_width = w
        self.last_stop_frame = frame_idx

    def draw_speed(self, frame):
        text = f"Speed: {int(self.current_speed)} mph"
        color = (0, 255, 0)
            
        cv2.putText(frame, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return frame
