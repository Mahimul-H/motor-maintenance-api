import pandas as pd
import numpy as np
import os

# 1. Setup Directory Logic
folder_name = "data"
file_name = "motor_maintenance_data.csv"
file_path = os.path.join(folder_name, file_name)

# Create the folder if it doesn't exist
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Directory '{folder_name}' created.")

# 2. Generation Parameters
np.random.seed(42)
n_rows = 100000

print(f"Generating {n_rows} rows of motor telemetry...")

# 3. Feature Generation
voltage = np.random.normal(230, 10, n_rows)
current = np.random.normal(12, 4, n_rows)
temperature = np.random.normal(60, 15, n_rows)
vibration = np.random.weibull(1.5, n_rows) * 0.1  # Weibull is great for wear-and-tear data

# 4. Failure Logic (The Ground Truth)
# We create a 'Failure' if sensors hit critical thresholds
risk = (
    (voltage < 190) | (voltage > 270) | 
    (current > 18) | 
    (temperature > 85) | 
    (vibration > 0.3)
)

# Convert boolean to integer (1 for Failure, 0 for Healthy)
failure = risk.astype(int)

# 5. Assemble and Save
df = pd.DataFrame({
    'voltage_v': np.round(voltage, 2),
    'current_a': np.round(current, 2),
    'temp_c': np.round(temperature, 2),
    'vibration_g': np.round(vibration, 4),
    'failure': failure
})

df.to_csv(file_path, index=False)

print("-" * 30)
print(f"Success! File saved to: {file_path}")
print(f"Dataset Summary:")
print(f"   - Total Rows: {len(df)}")
print(f"   - Failures Detected: {df['failure'].sum()} ({df['failure'].mean()*100:.2f}%)")
print("-" * 30)
