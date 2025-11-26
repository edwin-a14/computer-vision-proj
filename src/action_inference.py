import cv2

def determine_driver_action(detections, frame_width, frame_height):
    if not detections:
        return "DRIVE", (0, 255, 0) # Green
        
    # Find the largest stop sign by width
    max_width = 0
    for (x, y, w, h) in detections:
        if w > max_width:
            max_width = w
            
    # If the stop sign width is greater than 5% of the frame width, show "stop"
    stop_threshold = frame_width * 0.05 
    
    if max_width > stop_threshold:
        return "STOP", (0, 0, 255) # Red
    else:
        return "SLOW DOWN", (0, 255, 255) # Yellow
