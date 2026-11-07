import paho.mqtt.client as mqtt
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add the cloned ML Engine directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'CSC8112_MLEngine'))
from ml_engine import MLPredictor

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
        
        if len(pm25_dataset) >= TARGET_POINTS:
            print(f"\nTarget of {TARGET_POINTS} points reached. Disconnecting to process data.")
            client.disconnect()
            
    except Exception as e:
        print(f"\nError processing message: {e}")

def process_and_predict():
    print("\n--- Starting Data Processing ---")
    df = pd.DataFrame(pm25_dataset)
    
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df.drop_duplicates(subset=['Timestamp'], inplace=True)
    df.sort_values(by='Timestamp', inplace=True)
    
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Formatted_Timestamp'] = df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    print("Formatted PM2.5 Data:")
    print(df[['Formatted_Timestamp', 'Value']].tail())
    
    df.set_index('Timestamp', inplace=True)
    
    # Average the PM2.5 data over 10-minute periods
    df_10min = df.resample('10min').mean()
    
    # Average daily data and visualize
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
    
    print("\n--- Initializing Machine Learning Engine ---")
    try:
        # MLPredictor requires Timestamp and Value as columns, not as an index
        df_10min.reset_index(inplace=True)
        
        # Initialize, train, and predict using the provided class
        predictor = MLPredictor(df_10min)
        predictor.train()
        forecast = predictor.predict()
        
        # Plot and save
        fig = predictor.plot_result(forecast)
        pred_plot_path = os.path.join(OUTPUT_DIR, 'pm25_prediction.png')
        fig.savefig(pred_plot_path)
        
        print(f"ML prediction completed. Forecast graph saved to {pred_plot_path}")
    except Exception as e:
        print(f"ML Engine execution failed: {e}")

if __name__ == "__main__":
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    
    client.connect(BROKER, 1883, 60)
    client.loop_forever() 
    
    process_and_predict()