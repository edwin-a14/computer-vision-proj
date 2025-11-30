import sys
import os
import cv2
import numpy as np
import base64
import logging
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
sys.path.append(src_dir)
sys.path.append(project_root)

try:
    from detect_color_shape import init_sift_verifier
    from process_video import load_classifiers
    from video_processor import VideoProcessor
except ImportError as e:
    logger.error(f"Import Error: {e}")
    logger.error(f"Sys path: {sys.path}")
    raise

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

clf = None
CLASSIFIER_TYPE = 'hog'

processor = VideoProcessor(fps=10, classifier_type=CLASSIFIER_TYPE)

def init_model():
    global clf
    
    # We need to make sure we are in the project root for relative paths to work
    os.chdir(project_root)
    init_sift_verifier()
    
    clf = load_classifiers(
        classifier_type=CLASSIFIER_TYPE, 
        cnn_path='computations/cnn_checkpoints/best_model.pth',
        hog_path='computations/hog_svm_stop_and_bg.pkl'
    )
    
    if clf is None:
        logger.error("Failed to load classifier!")
    else:
        logger.info(f"Successfully loaded {CLASSIFIER_TYPE} classifier")

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('process_frame')
def handle_frame(data):
    if clf is None:
        emit('error', {'message': 'Classifier not loaded'})
        return

    try:
        if 'image' not in data:
            return

        # Decode base64 image
        # Format is "data:image/jpeg;base64,......"
        image_data = data['image'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return

        processed_frame, num_dets, detections = processor.process_frame(frame, clf)

        # Encode back to base64
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_image_data = base64.b64encode(buffer).decode('utf-8')
        
        emit('frame_processed', {
            'image': f'data:image/jpeg;base64,{processed_image_data}',
            'detections': num_dets,
            'box_count': len(detections)
        })
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        emit('error', {'message': str(e)})

if __name__ == '__main__':
    init_model()
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)
