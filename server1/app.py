from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import io
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load YOLOv3 model with correct paths
net = cv2.dnn.readNet(
    os.path.join(current_dir, "yolov3.weights"),
    os.path.join(current_dir, "cfg", "yolov3.cfg")
)
with open(os.path.join(current_dir, "data", "coco.names"), "r") as f:
    classes = f.read().strip().split("\n")

# Define the endpoint for object detection
@app.route('/detect-objects', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Read the image
    image_file = request.files['image']
    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Get image dimensions
    height, width, _ = image.shape

    # Prepare the image for YOLOv3
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Run forward pass
    detections = net.forward(output_layers)

    # Process detections
    boxes = []
    confidences = []
    class_ids = []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels
    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        label = str(classes[class_ids[i]])
        confidence = confidences[i]

        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Put label and confidence
        cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert the processed image to bytes
    _, buffer = cv2.imencode('.png', image)
    byte_stream = io.BytesIO(buffer)

    # Return the processed image as a response
    return send_file(byte_stream, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)