import cv2
import os

# Create a folder to save images
folder_name = "scissors"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

cam = cv2.VideoCapture(0) # 0 is usually the default webcam
count = 0

print("Press 's' to save an image, or 'q' to quit.")

while True:
    ret, frame = cam.read()
    if not ret:
        break
        
    cv2.imshow("Webcam Feed", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'): # Press 's' to save
        img_name = f"{folder_name}/image_{count}.png"
        cv2.imwrite(img_name, frame)
        print(f"Saved: {img_name}")
        count += 1
    elif key == ord('q'): # Press 'q' to exit
        break

cam.release()
cv2.destroyAllWindows()
