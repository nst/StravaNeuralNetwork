import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
from joblib import dump, load

# Load data from CSV file
df = pd.read_csv('strava.csv')

# Prepare the data
X_continuous = df[['distance_km', 'total_elevation_gain_meters']].values
X_binary = df[['is_race']].values
y = df['elapsed_time_minutes'].values.reshape(-1, 1)

# Split the data into training and test sets
X_cont_train, X_cont_test, X_bin_train, X_bin_test, y_train, y_test = train_test_split(
    X_continuous, X_binary, y, test_size=0.2, random_state=42)

# Standardize the continuous features
scaler_X_cont = MinMaxScaler()
X_cont_train_scaled = scaler_X_cont.fit_transform(X_cont_train)
X_cont_test_scaled = scaler_X_cont.transform(X_cont_test)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Save the scalers
dump(scaler_X_cont, 'scaler_X_cont.joblib')
dump(scaler_y, 'scaler_y.joblib')

# Combine the scaled continuous features with the binary feature
X_train_scaled = np.hstack((X_cont_train_scaled, X_bin_train))
X_test_scaled = np.hstack((X_cont_test_scaled, X_bin_test))

# Convert to DataFrame and save the train/test split data to CSV
X_train_df = pd.DataFrame(X_train_scaled, columns=['distance_km', 'total_elevation_gain_meters', 'is_race'])
X_test_df = pd.DataFrame(X_test_scaled, columns=['distance_km', 'total_elevation_gain_meters', 'is_race'])
y_train_df = pd.DataFrame(y_train_scaled, columns=['elapsed_time_minutes'])
y_test_df = pd.DataFrame(y_test_scaled, columns=['elapsed_time_minutes'])

X_train_df.to_csv('X_train.csv', index=False)
X_test_df.to_csv('X_test.csv', index=False)
y_train_df.to_csv('y_train.csv', index=False)
y_test_df.to_csv('y_test.csv', index=False)

# Build a simple neural network with TensorFlow
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(3,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='mse')

# Train the model
history = model.fit(X_train_scaled, y_train_scaled, epochs=1000, validation_split=0.2, verbose=1)

# Save the model
model.save('strava_model.h5')

# Evaluate the model on the test set
test_loss = model.evaluate(X_test_scaled, y_test_scaled)
print(f'Test Loss: {test_loss}')

# Calculate custom loss (MSE) on the test set
y_pred = model.predict(X_test_scaled).flatten()
custom_mse_loss = np.mean((y_pred - y_test_scaled.flatten()) ** 2)
print(f'Test MSE Loss: {custom_mse_loss}')

# Plot the loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss over Epochs')
plt.show()