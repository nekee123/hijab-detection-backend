"""
YOLOv8 Hijab Detection Model Training Script
Using Roboflow Dataset

This script demonstrates how to:
1. Download a hijab detection dataset from Roboflow
2. Train a YOLOv8 model for hijab detection
3. Validate and export the trained model
4. Test the model on sample images

Prerequisites:
- Roboflow account with a hijab detection dataset
- Roboflow API key
- Python 3.10+
"""

from ultralytics import YOLO
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Roboflow Configuration
# Get your API key from: https://app.roboflow.com/settings/api
ROBOFLOW_API_KEY = "your_roboflow_api_key_here"
ROBOFLOW_WORKSPACE = "your-workspace"
ROBOFLOW_PROJECT = "hijab-detection"
ROBOFLOW_VERSION = 1

# Training Configuration
MODEL_SIZE = "yolov8n"  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
EPOCHS = 100            # Number of training epochs
IMAGE_SIZE = 640        # Input image size
BATCH_SIZE = 16         # Batch size (adjust based on GPU memory)
DEVICE = 0              # GPU device (0 for first GPU, 'cpu' for CPU)

# Paths
DATASET_DIR = "datasets/hijab_detection"
OUTPUT_DIR = "runs/detect/train"


# ============================================================================
# STEP 1: DOWNLOAD DATASET FROM ROBOFLOW
# ============================================================================

def download_roboflow_dataset():
    """
    Download hijab detection dataset from Roboflow
    
    Instructions:
    1. Create a Roboflow account at https://roboflow.com
    2. Create a new project for hijab detection
    3. Upload and label your images (or use a public dataset)
    4. Generate a dataset version
    5. Get your API key from Settings > API
    6. Update the configuration variables above
    """
    try:
        from roboflow import Roboflow
        
        print("=" * 70)
        print("DOWNLOADING DATASET FROM ROBOFLOW")
        print("=" * 70)
        
        # Initialize Roboflow
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        
        # Get project
        project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
        
        # Download dataset in YOLOv8 format
        dataset = project.version(ROBOFLOW_VERSION).download("yolov8", location=DATASET_DIR)
        
        print(f"\n✓ Dataset downloaded successfully to: {DATASET_DIR}")
        print(f"✓ Dataset YAML path: {dataset.location}/data.yaml")
        
        return f"{dataset.location}/data.yaml"
        
    except ImportError:
        print("\n⚠ Roboflow package not installed.")
        print("Install it with: pip install roboflow")
        print("\nAlternatively, you can manually download your dataset:")
        print("1. Go to your Roboflow project")
        print("2. Click 'Export' > 'YOLOv8' format")
        print("3. Download and extract to the 'datasets/hijab_detection' folder")
        return None
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        print("\nPlease check:")
        print("- Your Roboflow API key is correct")
        print("- Workspace, project, and version names are correct")
        print("- You have internet connection")
        return None


# ============================================================================
# STEP 2: PREPARE DATASET MANUALLY (ALTERNATIVE METHOD)
# ============================================================================

def create_manual_dataset_structure():
    """
    Create dataset structure for manual labeling
    
    If you don't use Roboflow, create this structure:
    
    datasets/hijab_detection/
    ├── data.yaml
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
    """
    print("\n" + "=" * 70)
    print("MANUAL DATASET STRUCTURE")
    print("=" * 70)
    
    # Create directories
    base_path = Path(DATASET_DIR)
    for split in ['train', 'valid', 'test']:
        (base_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (base_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Create data.yaml
    yaml_content = f"""# Hijab Detection Dataset Configuration
# Generated for YOLOv8 training

path: {base_path.absolute()}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: valid/images    # val images (relative to 'path')
test: test/images    # test images (optional)

# Classes
names:
  0: hijab

# Number of classes
nc: 1
"""
    
    yaml_path = base_path / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✓ Dataset structure created at: {base_path}")
    print(f"✓ Configuration file created: {yaml_path}")
    print("\nNext steps:")
    print("1. Add your images to train/images, valid/images, test/images")
    print("2. Add corresponding YOLO format labels to train/labels, valid/labels, test/labels")
    print("3. Label format: <class_id> <x_center> <y_center> <width> <height> (normalized 0-1)")
    
    return str(yaml_path)


# ============================================================================
# STEP 3: TRAIN YOLOV8 MODEL
# ============================================================================

def train_model(data_yaml_path):
    """
    Train YOLOv8 model for hijab detection
    
    Args:
        data_yaml_path: Path to the dataset YAML configuration file
    """
    print("\n" + "=" * 70)
    print("TRAINING YOLOV8 MODEL")
    print("=" * 70)
    
    # Load a pretrained YOLOv8 model
    model = YOLO(f"{MODEL_SIZE}.pt")
    
    print(f"\n✓ Loaded pretrained {MODEL_SIZE} model")
    print(f"✓ Dataset: {data_yaml_path}")
    print(f"✓ Training for {EPOCHS} epochs")
    print(f"✓ Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"✓ Batch size: {BATCH_SIZE}")
    print(f"✓ Device: {DEVICE}")
    
    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        name="hijab_detection",
        patience=50,          # Early stopping patience
        save=True,            # Save checkpoints
        plots=True,           # Save training plots
        val=True,             # Validate during training
        verbose=True,         # Verbose output
        # Augmentation parameters
        hsv_h=0.015,          # HSV-Hue augmentation
        hsv_s=0.7,            # HSV-Saturation augmentation
        hsv_v=0.4,            # HSV-Value augmentation
        degrees=0.0,          # Rotation augmentation
        translate=0.1,        # Translation augmentation
        scale=0.5,            # Scale augmentation
        shear=0.0,            # Shear augmentation
        perspective=0.0,      # Perspective augmentation
        flipud=0.0,           # Vertical flip probability
        fliplr=0.5,           # Horizontal flip probability
        mosaic=1.0,           # Mosaic augmentation probability
        mixup=0.0,            # Mixup augmentation probability
    )
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"\n✓ Best model saved to: runs/detect/hijab_detection/weights/best.pt")
    print(f"✓ Last model saved to: runs/detect/hijab_detection/weights/last.pt")
    print(f"✓ Training results: runs/detect/hijab_detection/")
    
    return model


# ============================================================================
# STEP 4: VALIDATE MODEL
# ============================================================================

def validate_model(model_path="runs/detect/hijab_detection/weights/best.pt", data_yaml_path=None):
    """
    Validate the trained model on the validation set
    
    Args:
        model_path: Path to the trained model
        data_yaml_path: Path to the dataset YAML file
    """
    print("\n" + "=" * 70)
    print("VALIDATING MODEL")
    print("=" * 70)
    
    # Load the trained model
    model = YOLO(model_path)
    
    # Validate
    metrics = model.val(data=data_yaml_path)
    
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    print(f"✓ mAP50: {metrics.box.map50:.4f}")
    print(f"✓ mAP50-95: {metrics.box.map:.4f}")
    print(f"✓ Precision: {metrics.box.mp:.4f}")
    print(f"✓ Recall: {metrics.box.mr:.4f}")
    
    return metrics


# ============================================================================
# STEP 5: TEST MODEL ON SAMPLE IMAGES
# ============================================================================

def test_model(model_path="runs/detect/hijab_detection/weights/best.pt", test_images_dir="test_images"):
    """
    Test the trained model on sample images
    
    Args:
        model_path: Path to the trained model
        test_images_dir: Directory containing test images
    """
    print("\n" + "=" * 70)
    print("TESTING MODEL ON SAMPLE IMAGES")
    print("=" * 70)
    
    # Load the trained model
    model = YOLO(model_path)
    
    # Check if test images directory exists
    if not os.path.exists(test_images_dir):
        print(f"\n⚠ Test images directory not found: {test_images_dir}")
        print("Create a 'test_images' folder and add some images to test the model")
        return
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    test_images = [
        os.path.join(test_images_dir, f) 
        for f in os.listdir(test_images_dir) 
        if os.path.splitext(f)[1].lower() in image_extensions
    ]
    
    if not test_images:
        print(f"\n⚠ No images found in {test_images_dir}")
        return
    
    print(f"\n✓ Found {len(test_images)} test images")
    
    # Run inference
    results = model(test_images, save=True, conf=0.5)
    
    print(f"\n✓ Predictions saved to: runs/detect/predict/")
    
    # Print detection summary
    for i, result in enumerate(results):
        num_detections = len(result.boxes)
        print(f"  - {os.path.basename(test_images[i])}: {num_detections} hijabs detected")
    
    return results


# ============================================================================
# STEP 6: EXPORT MODEL FOR DEPLOYMENT
# ============================================================================

def export_model(model_path="runs/detect/hijab_detection/weights/best.pt"):
    """
    Export the trained model to different formats for deployment
    
    Args:
        model_path: Path to the trained model
    """
    print("\n" + "=" * 70)
    print("EXPORTING MODEL")
    print("=" * 70)
    
    # Load the trained model
    model = YOLO(model_path)
    
    # Export to ONNX format (recommended for deployment)
    print("\n✓ Exporting to ONNX format...")
    model.export(format="onnx")
    
    # Copy best.pt to project root for FastAPI backend
    import shutil
    destination = "best.pt"
    shutil.copy(model_path, destination)
    print(f"\n✓ Model copied to project root: {destination}")
    print("✓ Ready to use with FastAPI backend!")
    
    return destination


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training pipeline"""
    print("\n" + "=" * 70)
    print("YOLOV8 HIJAB DETECTION MODEL TRAINING")
    print("=" * 70)
    
    # Step 1: Download or prepare dataset
    print("\nChoose dataset source:")
    print("1. Download from Roboflow (recommended)")
    print("2. Use manually prepared dataset")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        data_yaml_path = download_roboflow_dataset()
        if not data_yaml_path:
            print("\n✗ Failed to download dataset. Please check configuration.")
            return
    elif choice == "2":
        data_yaml_path = create_manual_dataset_structure()
        print("\n⚠ Please add your images and labels before training!")
        proceed = input("Have you added your dataset? (y/n): ").strip().lower()
        if proceed != 'y':
            print("\n✗ Training cancelled. Add your dataset and run again.")
            return
    else:
        print("\n✗ Invalid choice. Exiting.")
        return
    
    # Step 2: Train model
    print("\n⚠ Training will start. This may take several hours depending on your dataset size.")
    proceed = input("Continue? (y/n): ").strip().lower()
    if proceed != 'y':
        print("\n✗ Training cancelled.")
        return
    
    model = train_model(data_yaml_path)
    
    # Step 3: Validate model
    validate_model(data_yaml_path=data_yaml_path)
    
    # Step 4: Test on sample images (optional)
    test_choice = input("\nTest model on sample images? (y/n): ").strip().lower()
    if test_choice == 'y':
        test_model()
    
    # Step 5: Export model
    export_model()
    
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETED!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review training results in: runs/detect/hijab_detection/")
    print("2. Check validation metrics and confusion matrix")
    print("3. Use best.pt with your FastAPI backend")
    print("4. Deploy your application!")
    print("\n" + "=" * 70)


# ============================================================================
# QUICK INFERENCE EXAMPLE
# ============================================================================

def quick_inference_example():
    """
    Quick example of using the trained model for inference
    """
    from ultralytics import YOLO
    
    # Load model
    model = YOLO("best.pt")
    
    # Run inference on an image
    results = model("path/to/your/image.jpg", conf=0.5)
    
    # Get detection count
    hijab_count = len(results[0].boxes)
    
    print(f"Detected {hijab_count} hijabs")
    
    # Save results
    results[0].save("result.jpg")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Install required packages
    print("Checking required packages...")
    try:
        import ultralytics
        print("✓ ultralytics installed")
    except ImportError:
        print("✗ ultralytics not installed. Run: pip install ultralytics")
        exit(1)
    
    # Run main training pipeline
    main()
