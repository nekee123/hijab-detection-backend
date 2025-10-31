"""
Pydantic Schemas
Data validation and serialization models
"""
from pydantic import BaseModel, Field
from typing import Optional

class DetectionResponse(BaseModel):
    """Response model for hijab detection"""
    image_name: str = Field(..., description="Name of the processed image")
    hijab_count: int = Field(..., ge=0, description="Number of hijabs detected")
    timestamp: str = Field(..., description="ISO format timestamp of detection")
    message: str = Field(default="Detection completed", description="Status message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_name": "hijab_detection_20251011_100530.jpg",
                "hijab_count": 5,
                "timestamp": "2025-10-11T10:05:30.123456",
                "message": "Detection completed successfully"
            }
        }


class DetectionRecord(BaseModel):
    """Model for detection record stored in database"""
    image_name: str = Field(..., description="Name of the processed image")
    hijab_count: int = Field(..., ge=0, description="Number of hijabs detected")
    timestamp: str = Field(..., description="ISO format timestamp of detection")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_name": "hijab_detection_20251011_100530.jpg",
                "hijab_count": 5,
                "timestamp": "2025-10-11T10:05:30.123456"
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str = Field(..., description="Error message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "File must be an image"
            }
        }
