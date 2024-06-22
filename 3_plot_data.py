import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data from CSV file
df = pd.read_csv('strava.csv')

# Prepare the data
X = df[['distance_km', 'total_elevation_gain_meters']]
y = df['elapsed_time_minutes']
workout_type = df['is_race']

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the data points with different colors for workout_type
colors = {0: 'blue', 1: 'red'}
sc = ax.scatter(y, X['distance_km'], X['total_elevation_gain_meters'], c=workout_type.map(colors), marker='o')

# Add axis labels and title
ax.set_xlabel('Elapsed Time (minutes)')
ax.set_ylabel('Distance (km)')
ax.set_zlabel('Total Elevation Gain (m)')
ax.set_title('3D Scatter Plot of Strava Runs')

# Create a legend
red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Race')
blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Training')
ax.legend(handles=[red_patch, blue_patch])

# Show plot
plt.show()
