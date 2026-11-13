import paho.mqtt.client as mqtt
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tflite_runtime.interpreter as tflite

BROKER = os.environ.get("MQTT_BROKER", "localhost")
SUBSCRIBE_TOPIC = "sensor/pm25"
PUBLISH_TOPIC = "sensor/classification"
TARGET_POINTS = 8000
OUTPUT_DIR = "/app/output"

pm25_dataset = []

def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker. Subscribing to {SUBSCRIBE_TOPIC}...")
    client.subscribe(SUBSCRIBE_TOPIC)

def on_message(client, userdata, msg):
    global pm25_dataset
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        pm25_dataset.append(payload)
        print(f"Collected {len(pm25_dataset)}/{TARGET_POINTS} points for classification...", end='\r')
    except Exception as e:
        pass

def process_and_classify(client):
    print("\n\n--- Starting Batch Inference ---")
    df = pd.DataFrame(pm25_dataset)
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df.dropna(subset=['Value'], inplace=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.sort_values('Timestamp', inplace=True)

    # 1. Load TFLite Model
    interpreter = tflite.Interpreter(model_path="pm25_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    class_names = {0: 'Green', 1: 'Yellow', 2: 'Red'}
    classifications = []

    # 2. Perform Batch Inference
    print("Running TFLite Inference on Edge...")
    for val in df['Value']:
        input_data = np.array([[val]], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        pred_class = np.argmax(output_data[0])
        classifications.append(class_names[pred_class])

    df['Classification'] = classifications

    # 3. Log Results & Publish to MQTT (For Task 5)
    print("\nInference Summary:")
    print(df['Classification'].value_counts())
    
    for _, row in df.iterrows():
        payload = {
            "Timestamp": row['Timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            "Value": row['Value'],
            "Classification": row['Classification']
        }
        client.publish(PUBLISH_TOPIC, json.dumps(payload), qos=1)
    
    print(f"Published {len(df)} classification records to {PUBLISH_TOPIC}")

    # 4. Generate Visualizations
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    color_map = {'Green': 'green', 'Yellow': 'yellow', 'Red': 'red'}

    # A. Bar Chart: Count of Classifications
    plt.figure(figsize=(8, 5))
    df['Classification'].value_counts().plot(kind='bar', color=[color_map[c] for c in df['Classification'].value_counts().index])
    plt.title('Classification Counts')
    plt.ylabel('Count')
    plt.savefig(os.path.join(OUTPUT_DIR, 'class_counts.png'))
    plt.close()

    # B. Time-Series Scatter Plot (Color-Coded)
    plt.figure(figsize=(12, 6))
    for c_level in ['Green', 'Yellow', 'Red']:
        subset = df[df['Classification'] == c_level]
        plt.scatter(subset['Timestamp'], subset['Value'], color=color_map[c_level], label=c_level, s=10)
    plt.title('Time-Series PM2.5 Values by Classification')
    plt.xlabel('Timestamp')
    plt.ylabel('PM2.5 Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'class_timeseries.png'))
    plt.close()

    # C. Frequency Distribution (Histogram)
    plt.figure(figsize=(10, 5))
    
    # Isolate the PM2.5 values by their calculated classification
    green_data = df[df['Classification'] == 'Green']['Value']
    yellow_data = df[df['Classification'] == 'Yellow']['Value']
    red_data = df[df['Classification'] == 'Red']['Value']
    
    # Plot as a stacked histogram matching your exact color mapping
    plt.hist([green_data, yellow_data, red_data], 
             bins=50, 
             stacked=True, 
             color=['green', 'yellow', 'red'], 
             edgecolor='black', 
             label=['Green', 'Yellow', 'Red'])
             
    plt.title('PM2.5 Value Frequency Distribution')
    plt.xlabel('PM2.5 Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'pm25_distribution.png'))
    plt.close()

    print("All classification visualizations saved to output directory.")

if __name__ == "__main__":
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, 1883, 120)
    
    client.loop_start()
    
    import time
    while len(pm25_dataset) < TARGET_POINTS:
        time.sleep(1)
        
    client.loop_stop()
    process_and_classify(client)
    client.disconnect()