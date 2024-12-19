from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the YOLO model
model = YOLO("best.pt")  # Path to your trained model

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

