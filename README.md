import cv2
import numpy as np
import pandas as pd
import pytesseract
import ipywidgets as widgets
from IPython.display import display
from PIL import Image
import io

# Install Tesseract OCR and its dependencies
!apt install tesseract-ocr
!apt install libtesseract-dev
!pip install pytesseract

# Function to preprocess the image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply preprocessing techniques to enhance text extraction
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return gray

# Function to extract text from image using OCR
def ocr_image(image):
    custom_config = r'--oem 3 --psm 6'  # Customize as needed
    text = pytesseract.image_to_string(image, config=custom_config)
    return text

# Function to parse text into tabular format
def parse_tabular_data(text):
    # Split text into rows
    rows = text.strip().split('\n')

    # Split each row into columns while handling quoted values
    tabular_data = []
    for row in rows:
        columns = []
        in_quotes = False
        start_idx = 0
        for i, char in enumerate(row):
            if char == '"':
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                columns.append(row[start_idx:i])
                start_idx = i + 1
        columns.append(row[start_idx:])
        tabular_data.append(columns)

    # Ensure all rows have the same number of columns
    max_columns = max(len(row) for row in tabular_data)
    for row in tabular_data:
        while len(row) < max_columns:
            row.append('')  # Add empty strings to fill the row

    return tabular_data

# Function to perform corrections on the DataFrame
def perform_corrections(df):
    # Add your correction logic here
    # For example, if a certain column contains invalid values, replace them with correct values
    # df['column_name'] = df['column_name'].apply(lambda x: correct_value(x))
    return df

# Function to handle image upload and processing with correction
def handle_image_upload(change):
    uploaded_filename = next(iter(uploader.value))
    content = uploader.value[uploaded_filename]['content']

    image = np.array(Image.open(io.BytesIO(content)))
    processed_image = preprocess_image(image)
    text = ocr_image(processed_image)
    tabular_data = parse_tabular_data(text)
    df = pd.DataFrame(tabular_data[1:], columns=tabular_data[0])

    # Perform document corrections
    df_corrected = perform_corrections(df)

    # Display the corrected DataFrame
    display(df_corrected)

# Create file uploader widget
uploader = widgets.FileUpload(accept='.jpg,.png', multiple=False)
uploader.observe(handle_image_upload, names='value')

# Display the uploader widget
display(uploader)
