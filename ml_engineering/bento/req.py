import numbers
from typing import List
from pydantic import BaseModel


class Image_Metadata(BaseModel):
    req_id: int
    filename: str
    image_bytes: bytearray


class OOD_Detector_Request(BaseModel):
    batch_request: List[Image_Metadata]
