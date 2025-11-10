import cv2
import numpy as np
import time

net = cv2.dnn.readNet('weights/yolov3-tiny.weights', 'cfg/yolov3-tiny.cfg')
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

cap = cv2.VideoCapture('usa-street.mp4')
ret, frame = cap.read()
if ret:
    height, width, channels = frame.shape
    print(f'Frame shape: {frame.shape}')
    print(f'Frame mean pixel value: {np.mean(frame)}')
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    print('Detection successful, outs length:', len(outs))
    for i, out in enumerate(outs):
        print(f'Output layer {i}: shape {out.shape}')
    # Process detections
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            print(f"Max confidence: {confidence}, class: {class_id}")
            if confidence > 0.01:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    print('Detections found:', len(boxes))
    if len(boxes) > 0:
        for i, box in enumerate(boxes):
            print(f'Detection {i}: class {classes[class_ids[i]]}, confidence {confidences[i]:.2f}, box {box}')
    else:
        print('No detections above confidence threshold.')
else:
    print('Failed to read frame')
cap.release()
