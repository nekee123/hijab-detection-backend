"""
FastAPI Main Application
Hijab Detection and Counting Backend with CRUD
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import APIRouter
from pydantic import BaseModel
import os
import shutil

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Hijab Detection API",
    description="Detect and count hijab wearers in crowd images using YOLOv8 with CRUD",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static and upload folders
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# In-memory store for uploaded images
images_db = {}  # {filename: filepath}

# --------------------
# Pydantic Models
# --------------------
class DetectionResult(BaseModel):
    count: int

class ImageResponse(BaseModel):
    message: str
    filename: str

class ImageListResponse(BaseModel):
    images: list[str]

# --------------------
# Root and Health Endpoints
# --------------------
@app.get("/", summary="Serve Frontend")
async def root():
    return FileResponse("static/index.html")

@app.get("/health", summary="Health Check")
async def health_check():
    return {"status": "ok"}

# --------------------
# Hijab Detection Endpoint
# --------------------
@app.post(
    "/api/detect",
    response_model=DetectionResult,
    summary="Detect Hijab in Uploaded Image",
    description="Upload an image and detect how many people are wearing hijab"
)
async def detect_hijab(file: UploadFile = File(...)):
    file_location = f"uploads/{file.filename}"
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Save in memory store
    images_db[file.filename] = file_location

    # --- YOLOv8 detection logic ---
    count = 5  # placeholder; replace with actual model logic

    return {"count": count}

# --------------------
# CRUD Endpoints for Images
# --------------------
@app.post(
    "/api/images",
    response_model=ImageResponse,
    summary="Upload Image",
    description="Upload an image and save it to the server"
)
async def upload_image(file: UploadFile = File(...)):
    file_location = f"uploads/{file.filename}"
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)
    images_db[file.filename] = file_location
    return {"message": "Image uploaded", "filename": file.filename}

@app.get(
    "/api/images",
    response_model=ImageListResponse,
    summary="List All Images",
    description="Get a list of all uploaded image filenames"
)
async def list_images():
    return {"images": list(images_db.keys())}

@app.get(
    "/api/images/{filename}",
    summary="Get Image by Filename",
    description="Retrieve an uploaded image by its filename"
)
async def get_image(filename: str):
    if filename not in images_db:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(images_db[filename])

@app.put(
    "/api/images/{filename}",
    response_model=ImageResponse,
    summary="Update Image",
    description="Replace an existing uploaded image with a new file"
)
async def update_image(filename: str, file: UploadFile = File(...)):
    if filename not in images_db:
        raise HTTPException(status_code=404, detail="Image not found")
    file_location = f"uploads/{filename}"
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)
    images_db[filename] = file_location
    return {"message": "Image updated", "filename": filename}

@app.delete(
    "/api/images/{filename}",
    response_model=ImageResponse,
    summary="Delete Image",
    description="Delete an uploaded image by its filename"
)
async def delete_image(filename: str):
    if filename not in images_db:
        raise HTTPException(status_code=404, detail="Image not found")
    os.remove(images_db[filename])
    images_db.pop(filename)
    return {"message": "Image deleted", "filename": filename}
