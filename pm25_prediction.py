import paho.mqtt.client as mqtt
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add the cloned ML Engine directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'CSC8112_MLEngine'))
# Note: Adjust the import below based on the actual main function name in the provided repo
from ml_model import predict_pm25 

BROKER = os.environ.get("MQTT_BROKER", "localhost")
TOPIC = "sensor/pm25"
TARGET_POINTS = 8706
OUTPUT_DIR = "/app/output"

pm25_dataset = []

def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker. Subscribing to {TOPIC}...")
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    global pm25_dataset
    
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        if isinstance(payload, list):
            pm25_dataset.extend(payload)
        else:
            pm25_dataset.append(payload)
            
        print(f"Collected {len(pm25_dataset)}/{TARGET_POINTS} PM2.5 data points...", end='\r')
        
        # (a) Collect until 8706 data points are reached
        if len(pm25_dataset) >= TARGET_POINTS:
            print(f"\nTarget of {TARGET_POINTS} points reached. Disconnecting to process data.")
            client.disconnect()
            
    except Exception as e:
        print(f"\nError processing message: {e}")

def process_and_predict():
    print("\n--- Starting Data Processing ---")
    df = pd.DataFrame(pm25_dataset)
    
    # Ensure Value is numeric
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    
    # (b) Filter duplicates and sort by timestamp
    df.drop_duplicates(subset=['Timestamp'], inplace=True)
    df.sort_values(by='Timestamp', inplace=True)
    
    # (d) Convert timestamp to YYYY-MM-DD HH:MM:SS format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Formatted_Timestamp'] = df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    print("Formatted PM2.5 Data:")
    print(df[['Formatted_Timestamp', 'Value']].tail()) 
    # Verification: Your last two values here should print as 2.925 and 3.295
    
    # Set index for time-series resampling
    df.set_index('Timestamp', inplace=True)
    
    # (c) Average the PM2.5 data over 10-minute periods
    df_10min = df.resample('10min').mean()
    
    # (e) Average daily data and visualize using Matplotlib
    df_daily = df.resample('D').mean()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(df_daily.index, df_daily['Value'], marker='o', linestyle='-', color='b')
    plt.title('Daily Averaged PM2.5 Data')
    plt.xlabel('Date')
    plt.ylabel('PM2.5 Value')
    plt.grid(True)
    plt.tight_layout()
    
    daily_plot_path = os.path.join(OUTPUT_DIR, 'daily_pm25_trend.png')
    plt.savefig(daily_plot_path)
    print(f"Daily trend chart saved to {daily_plot_path}")
    
    # (f) & (i) Feed data to ML engine and save predefined graphs to bind mount
    print("\n--- Initializing Machine Learning Engine ---")
    try:
        # Pass the 10-minute averaged dataframe and the output directory for graph generation
        # Adjust arguments based on the specific function signature of the provided repository
        predict_pm25(data=df_10min, output_path=OUTPUT_DIR)
        print(f"ML prediction completed. Output graphs saved to {OUTPUT_DIR}")
    except Exception as e:
        print(f"ML Engine execution failed: {e}")

if __name__ == "__main__":
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    
    client.connect(BROKER, 1883, 60)
    
    # Blocks until 8706 points are reached and disconnect() is called
    client.loop_forever() 
    
    # Proceeds to analytics and ML after the MQTT loop ends
    process_and_predict()