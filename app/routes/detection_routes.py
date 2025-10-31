from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os

# If you have your YOLOv8 model already loaded, import it here
# from app.models.yolo_model import yolo_model

router = APIRouter()

@router.post("/detect")
async def detect_hijab(file: UploadFile = File(...)):
    """
    Detect and count hijab wearers in an uploaded image.
    Returns JSON: {"message": "🧕 X hijab wearers detected!"}
    """
    # Save uploaded image
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_location = os.path.join(upload_dir, file.filename)

    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # --- YOLOv8 detection logic ---
    # Replace this with your actual model inference
    # Example:
    # results = yolo_model.predict(file_location)
    # hijab_boxes = [box for box in results if box.class_name == "hijab"]
    # count = len(hijab_boxes)

    # Placeholder count for now
    count = 3  # Replace with actual count from model

    # Return JSON response
    return JSONResponse(content={"message": f"🧕 {count} hijab wearers detected!"})
