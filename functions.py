import os
import time
from datetime import datetime, timedelta
import math

# Function to delete CSV files older than one day
def cleanup_csv_files(directory):
    now = time.time()
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.endswith('.csv'):
            # Get the file's modification time
            modified_time = os.path.getmtime(filepath)
            # Calculate the time difference
            time_difference = now - modified_time
            # Define the threshold (1 day in seconds)
            threshold = 24 * 60 * 60
            if time_difference > threshold:
                os.remove(filepath)



def round_next(num):
    next_num = math.ceil(num*10)/10
    if next_num - num < 0.1:
        return round(next_num, 1)
    else:
        return round(num, 1)

