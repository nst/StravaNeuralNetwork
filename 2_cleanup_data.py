#!/usr/bin/env python3

import json
import pandas as pd

# Load data from JSON file
with open('strava.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Filter and process activities
activities = []
for activity in data:
    if activity["sport_type"] != "Run" or activity["id"] == 7101822110:
        continue
    
    activities.append({
        #"name": activity["name"],
        "distance_km": activity["distance"] / 1000.0,
        "elapsed_time_minutes": activity["elapsed_time"] / 60.0,
        "total_elevation_gain_meters": activity["total_elevation_gain"],
        "is_race": 1 if activity.get("workout_type") == 1 else 0
    })

# Create a DataFrame and save to CSV
df = pd.DataFrame(activities)
print(df)
df.to_csv('strava.csv', index=False)
