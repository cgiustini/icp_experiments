import numpy as np
import IPython
import matplotlib.pyplot as plt
from icp import (ls_fit, find_closest_idxs, icp, icp_randsampl,
				    create_pcloud_xline, get_translation_matrix, filter_pcloud)
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




if __name__ == '__main__':


	s = create_pcloud_xline(0, 10, 10)

	R = transforms3d.euler.euler2mat(0, 0, np.pi/2, 'sxyz')
	s = np.concatenate((s, np.matmul(R, s)), axis=1)
	d = np.matmul(transforms3d.euler.euler2mat(0, 0, 0, 'sxyz'), s) +\
		get_translation_matrix(s, [20, 0, 0])

	R, t, e = icp_randsampl(s, d)

	e_ = np.matmul(R, s) + get_translation_matrix(s, t)
	np.allclose(e, e_, atol=1e-6)

	plt.plot(s[0, :], s[1, :], 'b+', label='source')
	plt.plot(d[0, :], d[1, :], 'g+', label='destination')
	plt.plot(e[0, :], e[1, :], 'r+', label='estimated destination')
	plt.xlim([-10, 40])
	plt.ylim([-10, 40])

	IPython.embed()