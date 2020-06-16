import numpy as np
import IPython
import matplotlib.pyplot as plt
from icp import (ls_fit, find_closest_idxs, icp, icp_randsampl,
				    create_pcloud_xline, get_translation_matrix, filter_pcloud)
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




if __name__ == '__main__':

	s, start_coords = load_radar_pcloud('radar_start.npz')
	d, stop_coords = load_radar_pcloud('radar_stop.npz')

	# s = create_pcloud_xline(0, 10, 10)

	# R = transforms3d.euler.euler2mat(0, 0, np.pi/2, 'sxyz')
	# s = np.concatenate((s, np.matmul(R, s)), axis=1)
	# d = s + get_translation_matrix(s, [20, 0, 0])

	# s = filter_pcloud(s, x_min=46, x_max=48, y_min=-1.0, y_max=1.5 ,z_min=0.5, z_max=5)
	# d = filter_pcloud(d, x_min=36, x_max=38, y_min=-1.0, y_max=1.5 ,z_min=0.5, z_max=5)

	s = filter_pcloud(s, z_min=0.5, z_max=5)
	d = filter_pcloud(d, z_min=0.5, z_max=5)

	l = min(s.shape[1], d.shape[1])
	s = s[:, 0:l]
	d = d[:, 0:l]

	# s = s[:, ::10]
	# d = d[:, ::10]

	# closest_idxs = find_closest_idxs(s, d)
	# d[:, :] = d[:, closest_idxs]
	# R, t, U, S, VT = ls_fit(s, d)
	s_expected = s + get_translation_matrix(s, [-9.99995422e+00, 0, 0])
	R, t, e = icp_randsampl(s, s_expected)

	# t_array = np.tile(np.array([t]).T, (1, s.shape[1]))
	# e = np.matmul(R, s) + t_array

	

	plt.plot(s[0, :], s[1, :], 'b+', label='source')
	plt.plot(s_expected[0, :], s_expected[1, :], 'g+', label='source')
	# plt.plot(d[0, :], d[1, :], 'g+', label='destination')
	plt.plot(e[0, :], e[1, :], 'r+', label='estimated destination')
	# plt.xlim([-10, 40])
	# plt.ylim([-10, 40])

	IPython.embed()