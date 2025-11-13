import os
import logging
import numpy as np
import cv2
import torch

from cnn_baseline import SmallCNNPlus, IMAGENET_MEAN, IMAGENET_STD, simple_preprocess

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class CNNClassifier:
    """
    Wrapper around SmallCNNPlus to provide sklearn-compatible interface
    for use in detect_color_shape.py pipeline.
    """
    
    def __init__(self, model_path, input_size=224, device=None, threshold=0.5):
        self.input_size = input_size
        self.threshold = threshold
        
        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logging.info(f"Loading CNN model from {model_path}")
        logging.info(f"Using device: {self.device}")
        
        # Load model
        self.model = SmallCNNPlus(dropout=0.30)
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            logging.info("CNN model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load CNN model: {e}")
            raise
    
    def _preprocess(self, img):
        # Convert BGR to RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        
        # Apply simple preprocessing (resize, pad, normalize)
        processed = simple_preprocess(img_rgb, self.input_size)
        
        # Convert to tensor: (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(processed.transpose(2, 0, 1)).float()
        
        return tensor
    
    @torch.no_grad()
    def predict(self, images):
        # Handle single image
        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            images = [images]
        
        # Preprocess all images
        batch = []
        for img in images:
            if img is None or img.size == 0:
                continue
            tensor = self._preprocess(img)
            batch.append(tensor)
        
        if len(batch) == 0:
            return np.array([])
        
        # Stack into batch
        batch_tensor = torch.stack(batch).to(self.device)
        
        # Forward pass
        with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
            logits = self.model(batch_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        
        # Threshold to get predictions
        predictions = np.where(probs >= self.threshold, 'stop', 'bg')
        
        return predictions
    
    @torch.no_grad()
    def predict_proba(self, images):
        # Handle single image
        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            images = [images]
        
        # Preprocess all images
        batch = []
        for img in images:
            if img is None or img.size == 0:
                continue
            tensor = self._preprocess(img)
            batch.append(tensor)
        
        if len(batch) == 0:
            return np.array([])
        
        # Stack into batch
        batch_tensor = torch.stack(batch).to(self.device)
        
        # Forward pass
        with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
            logits = self.model(batch_tensor)
            probs_stop = torch.sigmoid(logits).cpu().numpy()
        
        # Convert to [P(bg), P(stop)] format
        probs_bg = 1 - probs_stop
        proba = np.stack([probs_bg, probs_stop], axis=1)
        
        return proba
    
    @torch.no_grad()
    def predict_with_confidence(self, images):
        # Handle single image
        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            images = [images]
        
        # Preprocess all images
        batch = []
        for img in images:
            if img is None or img.size == 0:
                continue
            tensor = self._preprocess(img)
            batch.append(tensor)
        
        if len(batch) == 0:
            return np.array([]), np.array([])
        
        # Stack into batch
        batch_tensor = torch.stack(batch).to(self.device)
        
        # Forward pass
        with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
            logits = self.model(batch_tensor)
            probs_stop = torch.sigmoid(logits).cpu().numpy()
        
        # Get predictions and confidences
        predictions = np.where(probs_stop >= self.threshold, 'stop', 'bg')
        # Confidence is the probability of the predicted class
        confidences = np.where(probs_stop >= self.threshold, probs_stop, 1 - probs_stop)
        
        return predictions, confidences.flatten()
