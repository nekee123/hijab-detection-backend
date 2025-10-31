"""
YOLO Service
Handles YOLOv8 model loading and hijab detection
"""
from ultralytics import YOLO
import os

class YOLOService:
    """Service class for YOLO-based hijab detection"""
    
    def __init__(self, model_path: str = "best.pt"):
        """
        Initialize YOLO model
        
        Args:
            model_path: Path to the trained YOLO model file
        """
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO model"""
        try:
            if os.path.exists(self.model_path):
                print(f"Loading YOLO model from {self.model_path}")
                self.model = YOLO(self.model_path)
                print("YOLO model loaded successfully")
            else:
                print(f"Warning: Model file {self.model_path} not found.")
                print("Using default YOLOv8n model as placeholder.")
                print("Please replace with your trained hijab detection model (best.pt)")
                # Load a default model as placeholder
                self.model = YOLO("yolov8n.pt")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def detect_hijabs(self, image_path: str, confidence_threshold: float = 0.5) -> int:
        """
        Detect hijabs in an image and return the count
        
        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence score for detection
            
        Returns:
            Number of hijabs detected
        """
        try:
            # Run inference
            results = self.model(image_path, conf=confidence_threshold)
            
            # Count detections
            # For a trained hijab model, this will count hijab class detections
            # Assuming class 0 is 'hijab' in your trained model
            hijab_count = 0
            
            for result in results:
                if result.boxes is not None:
                    # Count all detections (assuming single-class hijab model)
                    # If multi-class, filter by class ID: result.boxes.cls == 0
                    hijab_count += len(result.boxes)
            
            print(f"Detected {hijab_count} hijabs in {image_path}")
            return hijab_count
            
        except Exception as e:
            print(f"Error during detection: {e}")
            raise
    
    def detect_with_details(self, image_path: str, confidence_threshold: float = 0.5) -> dict:
        """
        Detect hijabs and return detailed information including bounding boxes
        
        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence score for detection
            
        Returns:
            Dictionary with detection details
        """
        try:
            results = self.model(image_path, conf=confidence_threshold)
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        detection = {
                            "confidence": float(box.conf[0]),
                            "class_id": int(box.cls[0]),
                            "bbox": box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                        }
                        detections.append(detection)
            
            return {
                "count": len(detections),
                "detections": detections
            }
            
        except Exception as e:
            print(f"Error during detailed detection: {e}")
            raise
