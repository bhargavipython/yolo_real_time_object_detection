import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet('weights/yolov3-tiny.weights', 'cfg/yolov3-tiny.cfg')

# Load class labels
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load video
cap = cv2.VideoCapture('usa-street.mp4')

ret, frame = cap.read()
if ret:
    height, width, channels = frame.shape
    print(f"Frame shape: {frame.shape}")

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    print("Detection completed successfully")
    print(f"Number of output layers: {len(outs)}")
    print(f"First output shape: {outs[0].shape}")

    # Count detections
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    print(f"Number of detections above threshold: {len(boxes)}")
    print(f"Detected classes: {[classes[i] for i in class_ids]}")

else:
    print("Failed to read frame")

cap.release()
