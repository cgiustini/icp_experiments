import numpy as np
import IPython

start = np.load('radar_start.npz')
stop = np.load('radar_stop.npz')

start_points = start['points']
start_coords = start['coords']

stop_points = stop['points']
stop_coords = stop['coords']

start_pos = start_coords[0:3]
stop_pos = stop_coords[0:3]


IPython.embed()