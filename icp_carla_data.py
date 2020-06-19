import numpy as np
import IPython
import matplotlib.pyplot as plt
from icp import (ls_fit, find_closest_idxs, icp, icp_randsampl,
				 create_pcloud_xline, get_translation_matrix,
				 filter_pcloud, get_normals, filter_normals)
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

	# Point cloud. Flip x/y variables to get right-handed coordinate
	# system from the left-handed coordinate system of CARLA.
	p = np.zeros((3, len(d)), dtype=float)
	p[1, :] = x
	p[0, :] = y
	p[2, :] = z

	return p

def load_radar_pcloud(npz_file):
	points, coords = load_carla_radar_npz(npz_file)

	# Negate z and y coordinates to get right-handed coordinate data
	# from coordinates of the left-handed coordinate system of CARLA.
	coords[1] = -coords[1]
	coords[2] = -coords[2]
	return convert_carla_radar_to_point_cloud(points), coords


if __name__ == '__main__':

	# Select scenario
	npz_dict = {'pure_translation': ('Pure Translation', 'data/radar_start.npz', 'data/radar_stop.npz')}
	scenario = 'pure_translation'
	scenario_cfg = npz_dict[scenario]

	# Selet and configure the algorithms
	algorithms = {
		'icp': (icp, 1, {''}, 'ICP with all points'),
		# 'icp_randsampl': (icp_randsampl, 10, {''}, 'ICP with random subsampling')
	}

	# Select and load npz files
	s, start_coords = load_radar_pcloud(scenario_cfg[1])
	d, stop_coords = load_radar_pcloud(scenario_cfg[2])

	t_truth = (stop_coords - start_coords)[0:3]
	rax_truth, rax_angle = transforms3d.euler.euler2axangle(
		(stop_coords - start_coords)[3],
		(stop_coords - start_coords)[4],
		(stop_coords - start_coords)[5],
		axes='sxyz'
	)

	# Filter the points clouds
	s = filter_pcloud(s, z_min=0.5, z_max=5)
	d = filter_pcloud(d, z_min=0.5, z_max=5)

	algorithm_results = {}

	# Run the algorithms and collect results
	for alg, alg_cfg in iter(algorithms.items()):
		icp_func = alg_cfg[0]
		icp_iter = alg_cfg[1]
		icp_args = alg_cfg[2]

		s_ = np.copy(s)
		d_ = np.copy(d)

		ts = []
		Rs = []

		for n in np.arange(icp_iter):
			print(n)
			R, t, e = icp_func(s_, d_)
			ts.append(t)
			Rs.append(R)

		algorithm_results[alg] = {'ts': ts, 'Rs': Rs}

	for alg, alg_results in iter(algorithm_results.items()):

		ts = alg_results['ts']
		Rs = alg_results['Rs']

		t_error = np.array([np.array(t) - t_truth for t in ts])
		t_error_norm = np.linalg.norm(t_error, 2, axis=1)
		min_error_idx = np.argmin(t_error_norm)

		best_t = ts[min_error_idx]
		best_R = Rs[min_error_idx]

		alg_name = algorithms[alg][3]

		plt.figure()
		plt.title('%s: Translation Error Histogram' % alg_name)
		plt.hist(t_error_norm)
		plt.ylabel('Occurrences')
		plt.xlabel('Translation Error')

		plt.figure()
		e = np.matmul(best_R, s) + get_translation_matrix(s, best_t)
		plt.plot(d[0, :], d[1, :], 'g+', label='stop pcloud')
		plt.plot(e[0, :], e[1, :], 'r+', label='transformed start pcloud')
		plt.title('%s\n Stop and Transformed Point Cloud (Top View)' % alg_name)
		plt.xlabel('x (m)')
		plt.ylabel('y (m)')
		plt.legend()

	IPython.embed()