import pandas as pd 
import numpy as np
import sys


# import the nba players csv



# Prep divy trips

def prep_divy():
    divy_trips = pd.read_csv('data/Divvy_Trips_2020_Q1.csv')
    divy_trips['started_at'] = pd.to_datetime(divy_trips['started_at'])
    divy_trips['ended_at'] = pd.to_datetime(divy_trips['ended_at'])
    
    # Create ride time feature for the purpose of plotting a continous variable
    divy_trips['ride_time'] = divy_trips['ended_at'] - divy_trips['started_at']
    divy_trips['ride_time'] = divy_trips['ride_time'].apply(lambda x: x.seconds)
    
    divy_trips['weekday'] = divy_trips['started_at'].apply(lambda x: x.isoweekday())
    divy_trips['hour'] = divy_trips['started_at'].apply(lambda x: x.hour)
    
    return divy_trips
