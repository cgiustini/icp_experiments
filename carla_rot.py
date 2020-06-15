import numpy as np
import IPython
import matplotlib.pyplot as plt
from ls_fit import ls_fit, find_closest_idxs
from functools import reduce
import transforms3d


def load_carla_radar_npz(npz_file):
	data = np.load(npz_file)
	points = data['points']
	coords = data['coords']
	return points, coords

def convert_carla_radar_to_point_cloud(points):

	# Trig.
	d = np.cos(points[:, 2]) * points[:, 3]
	z = np.sin(points[:, 2]) * points[:, 3]
	x = np.cos(points[:, 1]) * d
	y = np.sin(points[:, 1]) * d

	# Point cloud
	p = np.zeros((3, len(d)), dtype=float)
	p[0, :] = x
	p[1, :] = y
	p[2, :] = z

	return p

def load_radar_pcloud(npz_file):
	points, coords = load_carla_radar_npz(npz_file)
	return convert_carla_radar_to_point_cloud(points), coords

def create_pcloud_xline(start, stop, n):
	pcloud = np.zeros((3, n), dtype=float)
	pcloud[0, :] = np.linspace(start, stop, n)
	return pcloud

def get_translation_matrix(pcloud, t):	
	return np.tile(np.array([t]).T, (1, pcloud.shape[1]))


def filter_pcloud(points, x_min=None, x_max=None, y_min=None, y_max=None, z_min=None, z_max=None):

	if x_min is not None:
		x_min_idx = points[0, :] > x_min
	else:
		x_min_idx = np.full((points.shape[1]), fill_value=True, dtype=bool)

	if y_min is not None:
		y_min_idx = points[1, :] > y_min
	else:
		y_min_idx = np.full((points.shape[1]), fill_value=True, dtype=bool)

	if z_min is not None:
		z_min_idx = points[2, :] > z_min
	else:
		z_min_idx = np.full((points.shape[1]), fill_value=True, dtype=bool)

	if x_max is not None:
		x_max_idx = points[0, :] < x_max
	else:
		x_max_idx = np.full((points.shape[1]), fill_value=True, dtype=bool)

	if y_max is not None:
		y_max_idx = points[1, :] < y_max
	else:
		y_max_idx = np.full((points.shape[1]), fill_value=True, dtype=bool)

	if z_max is not None:
		z_max_idx = points[2, :] < z_max
	else:
		z_max_idx = np.full((points.shape[1]), fill_value=True, dtype=bool)

	filter_idx = reduce(
		np.logical_and,
		[x_min_idx, y_min_idx, z_min_idx, x_max_idx, y_max_idx, z_max_idx]
	)

	return points[:, filter_idx]

def icp(s, d):

	e = np.copy(s)

	for i in range(100):
		closest_idxs = find_closest_idxs(e, d)
		d_ = d[:, closest_idxs]
		print(closest_idxs)

		R, t, U, S, VT = ls_fit(e, d_)
		e = np.matmul(R, e) + get_translation_matrix(e, t)
		# print(t)

	return R, t, U, S, VT, e


if __name__ == '__main__':

	# s, start_coords = load_radar_pcloud('radar_start.npz')
	# d, stop_coords = load_radar_pcloud('radar_stop.npz')

	s = create_pcloud_xline(0, 10, 10)

	R = transforms3d.euler.euler2mat(0, 0, np.pi/2, 'sxyz')
	s = np.concatenate((s, np.matmul(R, s)), axis=1)
	d = s + get_translation_matrix(s, [20, 0, 0])

	# s = filter_pcloud(s, x_min=46, x_max=48, y_min=-1.0, y_max=1.5 ,z_min=0.5, z_max=5)
	# d = filter_pcloud(d, x_min=36, x_max=38, y_min=-1.0, y_max=1.5 ,z_min=0.5, z_max=5)

	# l = min(s.shape[1], d.shape[1])
	# s = s[:, 0:l]
	# d = d[:, 0:l]

	# s = s[:, ::10]
	# d = d[:, ::10]

	# closest_idxs = find_closest_idxs(s, d)
	# d[:, :] = d[:, closest_idxs]
	# R, t, U, S, VT = ls_fit(s, d)

	R, t, U, S, VT, e = icp(s, d)

	# t_array = np.tile(np.array([t]).T, (1, s.shape[1]))
	# e = np.matmul(R, s) + t_array

	plt.plot(s[0, :], s[1, :], 'b+', label='source')
	plt.plot(d[0, :], d[1, :], 'g+', label='destination')
	plt.plot(e[0, :], e[1, :], 'r+', label='estimated destination')
	plt.xlim([-10, 40])
	plt.ylim([-10, 40])

	IPython.embed()