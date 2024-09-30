import numpy as np
import cv2
import os
import easyocr
import openpyxl
from openpyxl import Workbook
from datetime import datetime
import re
from ultralytics import YOLO

# Paths to directories and files
plates_dir = r"D:\B.Tech_IIT BHUBANESWAR\SEMESTER 7\Number Plate Detection\Number Plate\Number Plate\plates"
excel_path = r"D:\B.Tech_IIT BHUBANESWAR\SEMESTER 7\Number Plate Detection\Number Plate\Number Plate\Detected Plates.xlsx"
video_path = r"D:\B.Tech_IIT BHUBANESWAR\SEMESTER 7\Number Plate Detection\Number Plate\Number Plate\mycarplate.mp4"  # Path to the input video file

# Create directories if they do not exist
if not os.path.exists(plates_dir):
    os.makedirs(plates_dir)

# Create an Excel workbook and sheet if not exists
if not os.path.exists(excel_path):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Detected Plates"
    sheet.append(["Image Path", "Plate Number", "Date", "Time"])
else:
    workbook = openpyxl.load_workbook(excel_path)
    sheet = workbook.active

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Load the YOLO model
model = YOLO(r'D:\B.Tech_IIT BHUBANESWAR\SEMESTER 7\Number Plate Detection\Number Plate\Number Plate\model\best.pt')

# Function to preprocess image for better OCR results
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def is_valid_plate(text):
    # Generalized pattern to match different formats of number plates
    pattern = re.compile(r'^[A-Z0-9]{1,3}[-\s]?[A-Z0-9]{1,4}[-\s]?[A-Z0-9]{1,4}$')
    return bool(pattern.match(text.replace(" ", "").replace("-", "")))

# Function to get OCR results
def get_ocr_text(img, coords):
    x, y, w, h = map(int, coords)
    roi = img[y:h, x:w]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    results = reader.readtext(gray)
    ocr_text = ""

    for result in results:
        if len(results) == 1:
            ocr_text = result[1]
        if len(results) > 1 and len(result[1]) > 6 and result[2] > 0.2:
            ocr_text = result[1]

    return ocr_text

# Function to save extracted text and image path to Excel
def save_to_excel(img_path, plate_text):
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date, time = current_datetime.split()
    sheet.append([img_path, plate_text, date, time])
    workbook.save(excel_path)
    print(f"Saved to Excel: {plate_text}")

# Store previously detected plates and their confidence scores
detected_plates = {}

# Start video capture from the video file
cap = cv2.VideoCapture(video_path)
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform detection
    results = model.predict(image_rgb)

    # Parse detection results
    for result in results:
        for box in result.boxes:
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = box.cls[0]

            # Extract the region of interest (ROI) containing the number plate
            roi = frame[ymin:ymax, xmin:xmax]
            preprocessed_img = preprocess_image(roi)

            # Get OCR text
            plate_text = get_ocr_text(frame, (xmin, ymin, xmax, ymax))
            plate_text = plate_text.replace(" ", "").upper()
            print(f"OCR Text: {plate_text}")

            # Validate the plate text
            if is_valid_plate(plate_text):
                # Only save if it's a new plate or more confident detection
                if plate_text not in detected_plates or detected_plates[plate_text] < confidence:
                    detected_plates[plate_text] = confidence

                    # Draw the bounding box and extracted text on the frame
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, plate_text, (xmin, ymax + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)

                    # Save the ROI image
                    img_path = os.path.join(plates_dir, f"scanned_img_{count}.jpg")
                    cv2.imwrite(img_path, roi)
                    print(f"Image saved: {img_path}")

                    # Save the extracted text and metadata to the Excel file
                    save_to_excel(img_path, plate_text)
                    count += 1

    cv2.imshow("Result", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
        break

cap.release()
cv2.destroyAllWindows()
