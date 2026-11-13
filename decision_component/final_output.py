import paho.mqtt.client as mqtt
import requests
import json
import os
import urllib3
import base64
from datetime import datetime

# Suppress insecure request warnings if the university dashboard uses a self-signed cert on port 8000
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BROKER = os.environ.get("MQTT_BROKER", "localhost")
PM25_TOPIC = "sensor/classification"
CAMERA_TOPIC = "camera/processed"
API_URL = "http://dashboard.cloud.ncl.ac.uk:8000/api/records"

# In-memory state for the latest air quality reading
latest_pm25 = None

def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker. Subscribing to Decision Topics...")
    client.subscribe([(PM25_TOPIC, 1), (CAMERA_TOPIC, 1)])

def on_message(client, userdata, msg):
    global latest_pm25
    
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        
        # 1. Update latest PM2.5 classification state
        if msg.topic == PM25_TOPIC:
            latest_pm25 = payload
            
        # 2. Evaluate camera inference for the business logic trigger
        elif msg.topic == CAMERA_TOPIC:
            vehicle_count = payload.get("vehicle_count", 0)
            image_data = payload.get("image_base64", "")
            
            if vehicle_count > 5:
                if latest_pm25:
                    print(f"Threshold exceeded! {vehicle_count} vehicles detected. Transmitting to dashboard...")
                    send_to_dashboard(image_data)
                else:
                    print(f"High traffic ({vehicle_count} vehicles) detected, but awaiting first PM2.5 reading.")
                    
    except json.JSONDecodeError:
        print(f"Failed to decode JSON from topic {msg.topic}")
    except Exception as e:
        print(f"Error processing message: {e}")

def send_to_dashboard(image_data):
    # 1. Convert the string timestamp back to a float (Unix epoch time)
    timestamp_str = latest_pm25.get("Timestamp")
    try:
        dt_obj = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        timestamp_float = dt_obj.timestamp()
    except ValueError:
        # Fallback just in case it is already a float
        timestamp_float = float(timestamp_str)
        
    # 2. Set the standard form fields (excluding the image)
    api_payload = {
        "timestamp": timestamp_float,
        "value": float(latest_pm25.get("Value")),
        "predicted_quality": str(latest_pm25.get("Classification"))
    }
    
    # 3. Decode the base64 image back into raw file bytes
    image_bytes = base64.b64decode(image_data)
    
    # 4. Package the bytes as a file upload to satisfy FastAPI's UploadFile requirement
    api_files = {
        "image": ("traffic_capture.jpg", image_bytes, "image/jpeg")
    }
    
    try:
        # Send using BOTH data= (for fields) and files= (for the image)
        response = requests.post(API_URL, data=api_payload, files=api_files, timeout=15, verify=False) 
        
        if response.status_code in [200, 201]:
            print(f"Success! Dashboard accepted record. Status: {response.status_code}")
        else:
            print(f"API Error. Status: {response.status_code}, Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Dashboard connection failed: {e}")

if __name__ == "__main__":
    print("Starting Decision Module (Final Output)...")
    
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    client.on_connect = on_connect
    client.on_message = on_message
    
    client.connect(BROKER, 1883, 120)
    client.loop_forever()