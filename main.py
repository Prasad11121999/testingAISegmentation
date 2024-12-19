from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os
import zipfile
from io import BytesIO

# Initialize FastAPI app
app = FastAPI()

# Load the YOLO model
model = YOLO("best.pt")  # Path to your trained model

# Directory to temporarily store cropped images
CROPPED_IMAGES_DIR = "cropped_images"
os.makedirs(CROPPED_IMAGES_DIR, exist_ok=True)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform prediction
    results = model.predict(source=image, conf=0.5)

    # Annotate the image
    annotated_frame = results[0].plot()

    # Save the annotated image to a temporary file
    output_path = "result.jpg"
    cv2.imwrite(output_path, annotated_frame)

    # Return the annotated image as a response
    return FileResponse(output_path, media_type="image/jpeg")

@app.post("/predict/cropped/")
async def get_cropped_images(file: UploadFile = File(...)):
    # Read the uploaded file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform prediction
    results = model.predict(source=image, conf=0.5)

    # Extract crops and save them
    cropped_image_paths = []
    for i, box in enumerate(results[0].boxes):
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers

        # Crop the image
        cropped_image = image[y1:y2, x1:x2]

        # Save the cropped image
        cropped_path = os.path.join(CROPPED_IMAGES_DIR, f"crop_{i}.jpg")
        cv2.imwrite(cropped_path, cropped_image)
        cropped_image_paths.append(cropped_path)

    # If no crops, return a message
    if not cropped_image_paths:
        return JSONResponse(content={"message": "No objects detected."}, status_code=200)

    # Create a zip file containing all cropped images
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for image_path in cropped_image_paths:
            zip_file.write(image_path, os.path.basename(image_path))
    zip_buffer.seek(0)

    # Return the zip file as a response
    return StreamingResponse(zip_buffer, media_type="application/zip", headers={"Content-Disposition": "attachment; filename=cropped_images.zip"})
