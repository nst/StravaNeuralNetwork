import pandas as pd
import numpy as np
import tensorflow as tf
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the model
model = tf.keras.models.load_model('strava_model.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

# Load the scalers
scaler_X_cont = load('scaler_X_cont.joblib')
scaler_y = load('scaler_y.joblib')

def predict_elapsed_time(distance_km, total_elevation_gain_m, is_race):
    # Prepare the input data
    input_data_cont = np.array([[distance_km, total_elevation_gain_m]])
    input_data_bin = np.array([[is_race]])
    
    # Scale the continuous features
    input_data_cont_scaled = scaler_X_cont.transform(input_data_cont)
    
    # Combine the scaled continuous features with the binary feature
    input_data_scaled = np.hstack((input_data_cont_scaled, input_data_bin))
    
    # Predict elapsed time
    predicted_time_scaled = model.predict(input_data_scaled)
    
    # Inverse transform to get the actual time
    predicted_time = scaler_y.inverse_transform(predicted_time_scaled.reshape(-1, 1))
    return predicted_time[0][0]

def minutes_to_hhmm(minutes):
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours:02d}:{mins:02d}"

# Example usage
distance_km = 10  # Example distance in km
total_elevation_gain_m = 600  # Example elevation gain in meters
is_race = 0  # Example workout type

for (km, d_plus, is_race) in [(10,600,0), (31,2300,1), (5,850,0), (10,90,0)]:
	predicted_time = predict_elapsed_time(km, d_plus, is_race)
	hhmm = minutes_to_hhmm(predicted_time)
	print(f'Predicted time for {km} km, {d_plus} D+, race {is_race}: {hhmm}')
