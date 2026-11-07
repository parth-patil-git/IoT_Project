import paho.mqtt.client as mqtt
import json
import base64

TOPIC = "camera/processed"
BROKER = "localhost"

def on_connect(client, userdata, flags, rc):
    print(f"Connected to broker. Listening on '{TOPIC}'...")
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    try:
        # Parse the JSON payload
        payload = json.loads(msg.payload.decode('utf-8'))
        
        # Decode the base64 image string back into bytes
        img_bytes = base64.b64decode(payload["image_base64"])
        
        # Save to disk
        filename = "latest_detection.jpg"
        with open(filename, "wb") as f:
            f.write(img_bytes)
            
        print(f"Saved {filename} (Detected {payload['vehicle_count']} vehicles)")
        
    except Exception as e:
        print(f"Error decoding message: {e}")

if __name__ == "__main__":
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    
    client.connect(BROKER, 1883, 60)
    client.loop_forever()