import requests
import time
import os
from datetime import datetime
import paho.mqtt.client as mqtt

# Dynamically pull the broker from Docker environment, fallback to localhost
BROKER = os.environ.get("MQTT_BROKER", "localhost")
TOPIC = "camera/raw"

SENSOR_URL = "http://smartcam.uo.ncl.ac.uk:8001/sensor"
CAMERA_URL = "http://smartcam.uo.ncl.ac.uk:8001/camera"
POLL_INTERVAL_SECONDS = 5

def fetch_and_process_sensor_data():
    try:
        response = requests.get(SENSOR_URL, timeout=10)
        response.raise_for_status()
        data = response.json()

        pm25_data = []

        for reading in data:
            if reading.get("Variable") == "PM2.5":
                timestamp = reading.get("Timestamp")
                value = reading.get("Value")
            
                if timestamp is not None and value is not None:
                    pm25_data.append({
                        "Timestamp": timestamp,
                        "Value": value
                    })

                    if float(value) > 50:
                        print(f"[OUTLIER DETECTED] PM2.5 Value: {value} at {timestamp}")
        
        return pm25_data
    
    except requests.exceptions.RequestException as e:
        print(f"Sensor API Error: {e}")
        return []

def fetch_camera_frame():
    try:
        # FIXED: Changed SENSOR_URL to CAMERA_URL
        response = requests.get(CAMERA_URL, timeout=10)
        response.raise_for_status()

        image_bytes = response.content
        print(f"Camera frame fetched successfully: {len(image_bytes)} bytes")
        return image_bytes

    except requests.exceptions.RequestException as e:
        print(f"Camera API Error: {e}")
        return None

def main():
    print(f"Starting Data Collector Component. Connecting to broker: {BROKER}")

    client = mqtt.Client()
    client.connect(BROKER, 1883, 60)

    while True:
        sensor_data = fetch_and_process_sensor_data()
        if sensor_data:
            print(f"Processed {len(sensor_data)} valid PM2.5 readings.")

        camera_frame = fetch_camera_frame()

        # FIXED: Added defensive check to prevent publishing None
        if camera_frame:
            client.publish(TOPIC, camera_frame, qos=1)

        time.sleep(POLL_INTERVAL_SECONDS)
    
    client.disconnect()

if __name__ == "__main__":
    main()