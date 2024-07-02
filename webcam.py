# pip install opencv-python requests

import cv2
import numpy as np
import requests
from ultralytics import YOLO
import base64

# Load YOLOv8 model (adjust the path to your YOLOv8 model if necessary)
model = YOLO('best.pt')
print(model.names)

# Initialize the camera (0 is usually the built-in webcam)
webcamera = cv2.VideoCapture(0)

# Get the camera's frame width and height
frame_width = int(webcamera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(webcamera.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Function to perform OCR using OCR.space API
def ocr_space_api(image, api_key, language='eng', ocr_engine=2):
    url_api = "https://api.ocr.space/parse/image"
    _, encoded_image = cv2.imencode('.jpg', image)
    base64_image = base64.b64encode(encoded_image).decode('utf-8')
    base64_image = f"data:image/jpeg;base64,{base64_image}"
    data = {
        'apikey': api_key,
        'base64Image': base64_image,
        'language': language,
        'isOverlayRequired': False,
        'OCREngine': ocr_engine
    }
    response = requests.post(url_api, data=data)
    return response.json()

# Function to perform card search using Scryfall API
def scryfall_card_search(card_name):
    card_name = card_name.lower()
    card_name = card_name.replace(' ', '+')
    base_url = "https://api.scryfall.com/cards/named"
    params = {'fuzzy': card_name}
    response = requests.get(base_url, params=params)
    return response.json()

# Function to parse Scryfall API response and format the data
def parse_scryfall_response(response):
    card_info = {
        'Name': response.get('name', 'N/A'),
        'Set Name': response.get('set_name', 'N/A'),
        'Type Line': response.get('type_line', 'N/A'),
        'Mana Cost': response.get('mana_cost', 'N/A'),
        'Oracle Text': response.get('oracle_text', 'N/A'),
        'Power': response.get('power', 'N/A'),
        'Toughness': response.get('toughness', 'N/A'),
        'Colors': ', '.join(response.get('colors', [])),
        'Rarity': response.get('rarity', 'N/A'),
        'Artist': response.get('artist', 'N/A'),
        'Image URL': response.get('image_uris', {}).get('large', 'N/A'),
        'Released At': response.get('released_at', 'N/A'),
        'CMC': response.get('cmc', 'N/A'),
        'Reserved': response.get('reserved', 'N/A'),
        'Foil': response.get('foil', 'N/A'),
        'Nonfoil': response.get('nonfoil', 'N/A'),
        'Set': response.get('set', 'N/A'),
        'Set Type': response.get('set_type', 'N/A'),
        'Collector Number': response.get('collector_number', 'N/A'),
        'Digital': response.get('digital', 'N/A'),
        'Rarity': response.get('rarity', 'N/A'),
        'Prices': response.get('prices', {}),
    }
    return card_info

# Function to display card info in the second window
def display_card_info(card_info):
    card_image_url = card_info['Image URL']
    card_image = None
    if card_image_url != 'N/A':
        resp = requests.get(card_image_url, stream=True).raw
        card_image = np.asarray(bytearray(resp.read()), dtype="uint8")
        card_image = cv2.imdecode(card_image, cv2.IMREAD_COLOR)

    window_name = 'Card Info'
    text_image = np.zeros((600, 800, 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)
    thickness = 1
    line_height = 25

    y = 20
    text_lines = []
    for key, value in card_info.items():
        if key == 'Image URL':
            continue
        text = f"{key}: {value}"
        text_lines.append(text)

    text_start_y = 0

    def draw_text_image():
        text_image.fill(0)
        y = 20
        for i in range(text_start_y, min(len(text_lines), text_start_y + 24)):
            cv2.putText(text_image, text_lines[i], (10, y), font, font_scale, color, thickness, cv2.LINE_AA)
            y += line_height

    draw_text_image()

    if card_image is not None:
        card_image_resized = cv2.resize(card_image, (400, 550))
        combined_image = np.zeros((600, 1200, 3), dtype=np.uint8)
        combined_image[:600, :800] = text_image
        combined_image[:550, 800:1200] = card_image_resized
    else:
        combined_image = text_image

    cv2.imshow(window_name, combined_image)

    while True:
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('u') and text_start_y > 0:
            text_start_y -= 1
            draw_text_image()
        elif key == ord('d') and text_start_y < len(text_lines) - 24:
            text_start_y += 1
            draw_text_image()
        if card_image is not None:
            combined_image[:600, :800] = text_image
            cv2.imshow(window_name, combined_image)
        else:
            cv2.imshow(window_name, text_image)

api_key = 'K85988233388957'
consecutive_title_frames = 0
required_consecutive_frames = 5
detected_title = None

while True:
    success, frame = webcamera.read()
    if not success:
        break
    
    # Perform detection
    results = model.track(frame, classes=[0, 1], conf=0.6, imgsz=1088)
    
    detected_label_1 = False
    label_0_count = 0
    label_1_count = 0

    # Create a blank image with the same dimensions as the original frame
    blank_frame = np.zeros_like(frame)
    
    # Process detection results
    for box in results[0].boxes:
        if box.cls == 1:  # Check if the label is 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Expand the detection by 10 pixels on every side and an additional 10 pixels to the left
            x1 = max(x1 - 20, 0)  # Further 10 pixels to the left
            y1 = max(y1 - 10, 0)
            x2 = min(x2 + 10, frame_width)
            y2 = min(y2 + 10, frame_height)
            
            # Crop the image using the expanded bounding box coordinates
            cropped_img = frame[y1:y2, x1:x2]
            
            # Save the cropped image for inspection
            cv2.imwrite('detected_title.jpg', cropped_img)

            # Increase the counter for consecutive title frames
            consecutive_title_frames += 1
            
            # If the title has been detected for the required number of consecutive frames
            if consecutive_title_frames >= required_consecutive_frames:
                detected_title = cropped_img
                response = ocr_space_api(detected_title, api_key)
                print("OCR Response: ", response)

                if 'ParsedResults' in response and response['ParsedResults']:
                    parsed_text = response['ParsedResults'][0]['ParsedText'].strip()
                    if parsed_text:
                        print("Parsed Text: ", parsed_text)
                        scryfall_response = scryfall_card_search(parsed_text)
                        print("Scryfall Response: ", scryfall_response)
                        if 'object' in scryfall_response and scryfall_response['object'] == 'card':
                            card_info = parse_scryfall_response(scryfall_response)
                            display_card_info(card_info)
                else:
                    print("OCR Error Message: ", response.get('ErrorMessage', 'No error message'))
                    print("OCR Error Details: ", response.get('ErrorDetails', 'No error details'))

                consecutive_title_frames = 0  # Reset the counter after locking on the title
                break  # Exit the loop once the title is locked
            
            detected_label_1 = True
            label_1_count += 1
            break  # Assuming we only want to crop to the first detected label 1
        elif box.cls == 0:
            label_0_count += 1

    if detected_title is not None:
        cropped_img = detected_title
        cropped_height, cropped_width = cropped_img.shape[:2]
        center_y, center_x = frame_height // 2, frame_width // 2
        top_left_y = max(center_y - cropped_height // 2, 0)
        top_left_x = max(center_x - cropped_width // 2, 0)
        bottom_right_y = top_left_y + cropped_height
        bottom_right_x = top_left_x + cropped_width
        
        blank_frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = cropped_img
        display_frame = blank_frame
        
        detected_title = None  # Reset after processing
    else:
        display_frame = frame
    
    # Add the legend to the frame
    legend = f"Cards: {label_0_count}\nTitles: {label_1_count}"
    y0, dy = 30, 30
    for i, line in enumerate(legend.split('\n')):
        y = y0 + i * dy
        cv2.putText(display_frame, line, (frame_width - 200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 139), 2, cv2.LINE_AA)  # Dark red color (0, 0, 139)
    
    # Display the frame
    cv2.imshow("Live Camera", display_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close all OpenCV windows
webcamera.release()
cv2.destroyAllWindows()
