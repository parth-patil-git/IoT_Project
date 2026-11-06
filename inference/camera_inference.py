import paho.mqtt.client as mqtt
import cv2
import numpy as np
from ultralytics import YOLO
import json
import base64
import os

# Configuration
MQTT_BROKER = os.getenv("MQTT_BROKER", "mqtt-broker")
MQTT_PORT = 1883
TOPIC_RAW = "camera/raw"             # Subscribes to frames from Data Collector
TOPIC_PROCESSED = "camera/processed" # Publishes to Task 5

# Initialize YOLO model
model = YOLO('yolov8n.pt')

# COCO dataset class IDs for vehicles: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
VEHICLE_CLASSES = [2, 3, 5, 7]

def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with code {rc}")
    client.subscribe(TOPIC_RAW)

def on_message(client, userdata, msg):
    try:
        # Decode raw byte payload to OpenCV image
        nparr = np.frombuffer(msg.payload, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            print("Failed to decode image payload.")
            return

        # Run inference filtering for vehicle classes
        results = model(img, classes=VEHICLE_CLASSES)
        
        vehicle_count = 0
        
        # Process bounding boxes
        for r in results:
            for box in r.boxes:
                vehicle_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw red bounding box (BGR format: 0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        print(f"Processed frame. Detected {vehicle_count} vehicles.")

        # Encode the modified image back to JPEG, then to base64 for JSON transmission
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Construct JSON payload for Task 5
        payload = {
            "vehicle_count": vehicle_count,
            "image_base64": img_base64
        }

        # Publish results
        client.publish(TOPIC_PROCESSED, json.dumps(payload))
        
    except Exception as e:
        print(f"Inference error: {e}")

if __name__ == "__main__":
    print("Starting YOLO Inference Service...")
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()