import numpy as np
from tensorflow.keras.models import load_model
import pickle
import cv2
import requests
import imutils
import time

def capture_images(url, interval_seconds=5, num_images=10):
    image_list = []

    for _ in range(num_images):
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        img = imutils.resize(img, width=1000, height=1800)

        image_list.append(img)
        time.sleep(interval_seconds)

    return image_list
# url_to_capture = "http://10.12.60.98:8080//shot.jpg"
# captured_images = capture_images(url=url_to_capture, interval_seconds=2, num_images=10)

def face_cropped_from_list(images_list):
    # Load the face classifier
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    all_faces_list = []

    for img in images_list:
        faces_list = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        faces = face_classifier.detectMultiScale(gray, 1.2, 5)

        # Iterate through detected faces
        for (x, y, w, h) in faces:
            # Crop the face from the original image
            cropped_face = img[y: y+h, x:x+w]

            # Resize the face to a common size (e.g., 200x200)
            cropped_face = cv2.resize(cropped_face, (200, 200))

            # Convert the face to grayscale
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)

            # Append the face to the faces_list
            faces_list.append(cropped_face)

        # Append the faces_list to the all_faces_list
        all_faces_list.append(faces_list)

    return all_faces_list

# all_faces_list = face_cropped_from_list(captured_images)

def recognize_faces(image_lists, model_path='RecognitionModel.h5', map_path='StudentsMap.pkl'):
    # Load the model
    loaded_model = load_model(model_path)

    # Load the name mapping
    with open(map_path, 'rb') as file:
        loaded_object = pickle.load(file)

    # Initialize the result list
    results = []

    # Process each list of images
    for image_list in image_lists:
        # Initialize the list for this set of images
        names = []

        # Process each image in the list
        for img_array in image_list:
            resized_image = cv2.resize(img_array, (64, 64))
            gray_image = resized_image

            # Expand dimensions to make it compatible with the model input shape
            test_image = np.expand_dims(gray_image, axis=0)
            test_image = np.expand_dims(test_image, axis=-1)  # Add channel dimension for grayscale

            # Now, you can use the loaded model for prediction
            result = loaded_model.predict(test_image, verbose=0)
            names.append(loaded_object[np.argmax(result)])


        # Append the list of names for this set of images to the result
        results.append(names)

    return results

# result = recognize_faces(all_faces_list)
# print(result)