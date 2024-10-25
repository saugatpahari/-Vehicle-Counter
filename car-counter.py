from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# Initialize the model
model = YOLO("../yolo-weights/yolov8l.pt")  # Path to your YOLO model file

# Open the video capture (0 for the first webcam, or replace with video file path)
data = cv2.VideoCapture("../videos/cars.mp4")

# Class Names for the objects detected by the webcam
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
              "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Importing the car masked image to detect the video on certain region
vehicle_mask = cv2.imread("vehicle_mask.png")

# Check and resize vehicle_mask if necessary
if vehicle_mask is not None:
    if len(vehicle_mask.shape) == 2:  # If grayscale
        vehicle_mask = cv2.cvtColor(vehicle_mask, cv2.COLOR_GRAY2BGR)
    # Resize the mask to match the video frame size
    vehicle_mask = cv2.resize(vehicle_mask, (int(data.get(3)), int(data.get(4))))

# Tracking the Vehicle
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Creating a line to start counting after the line is touched or crossed
trackingLine = [100, 450, 650, 450]

# Initializing the counter value
totalCounter = []

# Error handling for video source
if not data.isOpened():
    print("Error: Could not open video source.")
    exit()

while True:
    # Capture frame-by-frame
    success, video = data.read()

    # Error handling for frame read of video
    if not success:
        print("Error: Could not read frame.")
        break

    # Check if vehicle_mask is loaded and has the same size as video
    if vehicle_mask is not None:
        vehicleRegion = cv2.bitwise_and(video, vehicle_mask)
    else:
        vehicleRegion = video

    # Predict with YOLO model
    output = model(vehicleRegion, stream=True)

    # Creating a detection using numpy
    detection = np.empty((0, 5))

    # Visualize results on the frame
    for o in output:
        boxes = o.boxes
        # contains the bounding box information
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),  int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1

            # Getting the confidence level of the object detected
            confidence_value = math.ceil(box.conf[0]*100)/100

            # Getting the value of the object detected to associate with the classnames
            name = int(box.cls[0])

            currentClass = classNames[name]

            # detecting certain classNames only
            if currentClass == "car" or currentClass == "bus" or currentClass == "truck"\
                    or currentClass == "motorbike" and confidence_value > 0.3:

                currentArray = np.array([x1, y1, x2, y2, confidence_value])
                detection = np.vstack((detection, currentArray))

    trackerResults = tracker.update(detection)

    cv2.line(video, (trackingLine[0], trackingLine[1]), (trackingLine[2], trackingLine[3]),
             (255, 255, 255), 4)

    for results in trackerResults:
        x1, y1, x2, y2, ID = results
        x1, y1, x2, y2, ID = int(x1), int(y1), int(x2), int(y2), int(ID)
        w, h = x2 - x1, y2 - y1

        # Creating a circular point to count when it touches the line created
        x, y = x1+w//2, y1+h//2
        cv2.circle(video, (x, y), 3, (182, 249, 159), cv2.FILLED)

        # creating the rectangle around the object detected using cvzone
        cvzone.cornerRect(video, (x1, y1, w, h), 7, 4, 2, (182, 249, 159), (27, 19, 37))

        # Displaying the captured ID
        cvzone.putTextRect(video, f'{ID}', (max(0, x1), max(50, y1)), scale=0.7, thickness=1, colorT=(0, 0, 0),
                           colorR=(255, 255, 255), offset=5)

        if trackingLine[0] < x < trackingLine[2] and trackingLine[1] - 17 < y < trackingLine[3] + 17:
            if totalCounter.count(ID) == 0:
                totalCounter.append(ID)
                cv2.line(video, (trackingLine[0], trackingLine[1]), (trackingLine[2], trackingLine[3]),
                         (0, 255, 0), 4)

        # Displaying the total Counter
        cvzone.putTextRect(video, f' Count : {len(totalCounter)}', (50, 50), colorT=(0, 0, 0), colorR=(255, 255, 255))

    # Display the resulting frame
    cv2.imshow("Vehicle Counter", video)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
data.release()
cv2.destroyAllWindows()
