import tensorflow as tf
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

loaded_model = tf.keras.models.load_model('model.keras')

MODEL_WIDTH = 224
MODEL_HEIGHT = 224

# Initialize the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

with open('class_names.txt', 'r') as file:
    class_names = [line.strip() for line in file]

counter = 0
label = ""
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame was not grabbed, break the loop
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting...")
        break
    
        
    
    resized_frame = cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT))

    preprocessed_frame = preprocess_input(resized_frame)

    expanded_frame = np.expand_dims(preprocessed_frame, axis=0)

    if counter%10 == 0:
      prediction = loaded_model.predict(expanded_frame)
      class_index = np.argmax(prediction)
      label = class_names[class_index]

    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # Display the resulting frame in a window
    cv2.imshow('Webcam Live Feed', frame)

    # Press 'q' on the keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    counter+=1

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
