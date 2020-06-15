import numpy as np
import matplotlib.pyplot as plt
import transforms3d
import IPython

def plot_points(ps, marker='b+', label=''):
	plt.plot(ps[0, :], ps[1, :], marker, label=label)


def find_closest_idxs(p1, p2):

	p2_idxs = []

	p1 = np.copy(p1).T
	p2 = np.copy(p2).T

	for i, a in enumerate(p1):

		idx = np.arange(len(p2))
		mask = np.ones(len(p2), np.bool)
		# mask[p2_idxs] = 0

		idx = idx[mask]
		error = np.abs(a - p2[mask])
		error = np.linalg.norm(error, axis=1)

		min_error_filt_idx = np.argmin(error)

		p2_idxs.append(idx[min_error_filt_idx])

	return p2_idxs
	
def icp(s, d):

	e = np.copy(s)

	for i in range(100):
		closest_idxs = find_closest_idxs(e, d)
		d_ = d[:, closest_idxs]

		R, t, U, S, VT = ls_fit(e, d_)
		e = np.matmul(R, e) + get_translation_matrix(e, t)
		# print(t)

	return R, t, U, S, VT, e

def ls_fit(s, d):

	n = s.shape[1]

	sc = np.tile(np.array([np.mean(s, axis=1)]).T, (1, n))
	dc = np.tile(np.array([np.mean(d, axis=1)]).T, (1, n))
	s_ = s - sc
	d_ = d - dc

	_dim = s.shape[0]

	H = np.zeros((_dim, _dim), dtype=float)

	for i in range(n):
		H = H + np.matmul(np.array([s_[:, i]]).T, np.array([d_[:, i]]))

	U, S, VT = np.linalg.svd(H, full_matrices=True)

	R = np.matmul(VT.T, U.T)
	t = dc[:, 0] - np.matmul(R, sc[:, 0])

	return R, t, U, S, VT


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


if __name__ == '__main__':
	
	# pure translation example
	# s = np.zeros((2, 3), dtype=float)
	# s[:, 0] = [-1, 0]
	# s[:, 1] = [0, 0]
	# s[:, 2] = [1, 0]
	# d = s + 5

	# 45 degree rotation, no translation example
	s = np.zeros((2, 3), dtype=float)
	s[:, 0] = [-1, 0]
	s[:, 1] = [0, 0]
	s[:, 2] = [1, 0]
	d = np.zeros((2, 3), dtype=float)
	d[:, 0] = [-1.0/np.sqrt(2), -1.0/np.sqrt(2)]
	d[:, 1] = [0, 0]
	d[:, 2] = [1.0/np.sqrt(2), 1.0/np.sqrt(2)]

	# 45 degree rotation with translation example
	# s = np.zeros((2, 3), dtype=float)
	# s[:, 0] = [-1, 0]
	# s[:, 1] = [0, 0]
	# s[:, 2] = [1, 0]
	# d = np.zeros((2, 3), dtype=float)
	# d[:, 0] = [2 - 1.0/np.sqrt(2), 2 - 1.0/np.sqrt(2)]
	# d[:, 1] = [2, 2]
	# d[:, 2] = [2 + 1.0/np.sqrt(2), 2 + 1.0/np.sqrt(2)]

	R, t, U, S, VT = ls_fit(s, d)
	n = s.shape[1]
	de = np.matmul(R, s) + np.tile(np.array([t]).T, (1, n))

	plot_points(s, 'bo', 'orig')
	plot_points(d, 'go', 'dest')
	plot_points(de, 'r+', 'ls fit')
	plt.legend()
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('2D Least Squares Fit Example')

	idxs = find_closest_idxs(s, d)

	IPython.embed()

