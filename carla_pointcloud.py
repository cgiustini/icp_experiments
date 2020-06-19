import carla
import random
import time
import numpy as np
# import matplotlib.pyplot as plt
import cv2
import queue
import IPython
import matplotlib.pyplot as plt
from kalman2 import LooseAccelPosKF
from extkalman import ImuGPSEKf
from copy import copy
import transforms3d
import math

from functools import reduce

actor_list = []

IM_WIDTH = 640
IM_HEIGHT = 480
SHOW_PREVIEW = False

TOWN03_LAT = 49.0
TOWN03_LON = 8.0
TOWN03_ALT = 0.0
EARTH_RADIUS = 6378137.0
scale = np.cos(TOWN03_LAT * np.pi / 180.0)

# straight_trajectory = False
straight_trajectory = True

def GeoLocationToLocation(lat, lon, alt, scale):

	x = scale * EARTH_RADIUS * lon * np.pi / 180.0
	y = - EARTH_RADIUS * np.log(np.tan((90.0+lat) * np.pi / 360.0))
	z = alt

	return x, y, z

ref_x, ref_y, ref_z = GeoLocationToLocation(TOWN03_LAT, TOWN03_LON, TOWN03_ALT, scale)


class CarEnv(object):

	SHOW_CAM = SHOW_PREVIEW
	STEER_AMT = 1.0
	im_width = IM_WIDTH
	im_height = IM_HEIGHT
	front_camera = None

	def __init__(self):
		self.client = carla.Client('localhost', 2000)
		self.client.set_timeout(2.0)
		self.world = self.client.get_world()
		self.map = self.world.get_map()
		self.blueprint_library = self.world.get_blueprint_library()
		self.vehicle_bp = self.blueprint_library.filter("model3")[0]
		self.actor_list = []

	def reset(self):

		straight_transform = carla.Transform(
			carla.Location(x=-36.6, y=-195.92, z=1),
			carla.Rotation(pitch=0, yaw=90, roll=0)
		)

		# uphill_transform = carla.Transform(
		# 	carla.Location(x=80, y=-5, z=1),
		# 	carla.Rotation(pitch=0, yaw=-88, roll=0)
		# )

		# uphill_transform = carla.Transform(
		# 	carla.Location(x=80, y=-25, z=5),
		# 	carla.Rotation(pitch=0, yaw=-30, roll=0)
		# )

		circle_transform = carla.Transform(
			carla.Location(x=-3, y=22, z=1), carla.Rotation(pitch=0, yaw=0, roll=0)
		)

		# random.choice(self.world.get_map().get_spawn_points())
		if straight_trajectory is True:
			self.transform = straight_transform
		else:
			self.transform = circle_transform

		self.vehicle = self.world.spawn_actor(self.vehicle_bp, self.transform)
		self.actor_list.append(self.vehicle)

		time.sleep(2)

		self.rgb_cam_bp = self.blueprint_library.find("sensor.camera.rgb")
		self.rgb_cam_bp.set_attribute("image_size_x", f"{self.im_width}")
		self.rgb_cam_bp.set_attribute("image_size_y", f"{self.im_height}")
		self.rgb_cam_bp.set_attribute("fov", "110")
		transform = carla.Transform(carla.Location(x=0, y=0, z=5), carla.Rotation(pitch=-90, yaw=0, roll=0))
		self.rgb_cam = self.world.spawn_actor(self.rgb_cam_bp, transform, attach_to=self.vehicle)
		self.rgb_cam.listen(lambda image: self.process_rgb(image))
		self.actor_list.append(self.rgb_cam)
		self.image = None

		# self.kf = LooseAccelPosKF()
		self.kf = ImuGPSEKf()
		self.imu_iter = 0
		self.handle_iter = 0
		self.gnss_ready = False
		self.imu_ready = False

		self.radar_bp = self.blueprint_library.find('sensor.other.radar')
		self.radar_bp.set_attribute("horizontal_fov", "30")
		self.radar_bp.set_attribute("vertical_fov", "30")
		self.radar_bp.set_attribute('points_per_second', "100000")
		self.radar_bp.set_attribute('range', "500")
		self.radar_bp.set_attribute("sensor_tick", "1.1")
		transform = carla.Transform(carla.Location(x=0, y=0, z=0), carla.Rotation(pitch=0, yaw=0, roll=0))
		self.radar = self.world.spawn_actor(self.radar_bp, transform, attach_to=self.vehicle)
		self.radar.listen(lambda meas: self.process_radar(meas))
		self.actor_list.append(self.radar)


		self.gnss_x, self.gnss_y, self.gnss_z = (0, 0, 0)

		self.truth_data = np.zeros((2000, 13, 1), dtype=float)
		self.gnss_data = np.zeros((2000, 3, 1), dtype=float)
		self.imu_data = np.zeros((2000, 3, 1), dtype=float)
		self.kf_data = np.zeros((2000, 14, 1), dtype=float)

		self.radar_plotted = False
		self.load_realistic_sensors()
		

		# self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=-0.0, brake=100))

	def process_radar(self, radar_data):
		# print(meas.horizontal_angle)
		# self.radar_meas = meas
		# print(meas.transform.rotation.yaw)

		current_rot = radar_data.transform.rotation
		if (len(radar_data) > 100000) and (self.radar_plotted == False):
			print(len(radar_data))
			for detect in radar_data:
				azi = math.degrees(detect.azimuth)
				alt = math.degrees(detect.altitude)
				# The 0.25 adjusts a bit the distance so the dots can
				# be properly seen
				fw_vec = carla.Vector3D(x=detect.depth - 0.25)
				carla.Transform(
					carla.Location(),
					carla.Rotation(
						pitch=current_rot.pitch + alt,
						yaw=current_rot.yaw + azi,
						roll=current_rot.roll)).transform(fw_vec)

				def clamp(min_v, max_v, value):
					return max(min_v, min(value, max_v))

				norm_velocity = detect.velocity # range [-1, 1]
				r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
				g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
				b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
				self.world.debug.draw_point(
					radar_data.transform.location + fw_vec,
					size=0.075,
					life_time=10,
					persistent_lines=False,
					color=carla.Color(r, g, b))

			self.radar_plotted  = True

			# velocity is 1st, azimuth is 2nd, altitude is 3rd, depth is 4th
			points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
			points = np.reshape(points, (len(radar_data), 4))
			self.points = points
			self.radar_data = radar_data
			coords = np.array([
				radar_data.transform.location.x,
				radar_data.transform.location.y,
				radar_data.transform.location.z,
				radar_data.transform.rotation.roll,
				radar_data.transform.rotation.pitch,
				radar_data.transform.rotation.yaw,
			])

			np.savez('radar.npz', points=points, coords=coords)

			# d = np.cos(points[:, 2]) * points[:, 3]
			# z = np.sin(points[:, 2]) * points[:, 3]
			# x = np.cos(points[:, 1]) * d
			# y = np.sin(points[:, 1]) * d

			# for i, p in enumerate(points):

			# 	fw_vec = carla.Vector3D(x=float(x[i]), y=float(y[i]), z=float(z[i]))

			# 	self.world.debug.draw_point(
			# 		radar_data.transform.location + fw_vec,
			# 		size=0.075,
			# 		life_time=10,
			# 		persistent_lines=False,
			# 		color=carla.Color(1, 0, 0))






		# # data = list() 
		# # for location in self.radar_meas:
		# # 	data.append([location.x, location.y, location.z])             
		# # data =  np.array(data).reshape((-1, 3))
		# # self.radar_data = 
		# points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
		# points = np.reshape(points, (len(radar_data), 4))
		# # points = np.reshape(points, (int(points.shape[0] / 3), 3))
		# # radar_data = np.array(points[:, :2])
		# # radar_data *= min(self.hud.dim) / 100.0
		# # # radar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
		# # radar_data = np.fabs(radar_data)  # pylint: disable=E1111
		# # radar_data = radar_data.astype(np.int32)
		# # radar_data = np.reshape(radar_data, (-1, 2))
		# # data = radar_data


		# start_c = carla.Location(x=self.radar_meas.transform.location.x,
		# 						 y=self.radar_meas.transform.location.y,
		# 						 z=self.radar_meas.transform.location.z)

		# for d in data:
		# 	end_c = carla.Location(x=float(d[0]),
		# 						   y=float(d[1]), z=float(d[2]))
		# # for location in self.radar_meas:
		# 	# end_c = carla.Location(x=start_c.x+location.x,
		# 	# 					   y=start_c.y+location.y, z=start_c.z + location.z)
		# 	self.world.debug.draw_point(end_c, size=0.05, color=carla.Color(255, 0, 0), life_time=1)
		# # y_c = carla.Location(x=start_c.x+out[0,1], y=start_c.y+out[1,1], z=start_c.z-out[2,1])
		# # z_c = carla.Location(x=start_c.x+out[0,2], y=start_c.y+out[1,2], z=start_c.z-out[2,2])



		# IPython.embed()

	def load_realistic_sensors(self):
		self.gnss_bp = self.blueprint_library.find("sensor.other.gnss")
		self.gnss_bp.set_attribute("sensor_tick", "0.25")
		self.gnss_bp.set_attribute("noise_alt_stddev", "0")
		self.gnss_bp.set_attribute("noise_lat_stddev", "0")
		self.gnss_bp.set_attribute("noise_lon_stddev", "0")
		transform = carla.Transform(carla.Location())
		self.gnss_sensor = self.world.spawn_actor(self.gnss_bp, transform, attach_to=self.vehicle)
		self.gnss_sensor.listen(lambda data: self.process_gnss(data))
		self.actor_list.append(self.gnss_sensor)

		self.imu_bp = self.blueprint_library.find("sensor.other.imu")
		self.imu_bp.set_attribute("sensor_tick", "0.001")
		self.imu_bp.set_attribute("noise_accel_stddev_x", "0.00")
		self.imu_bp.set_attribute("noise_accel_stddev_y", "0.00")
		self.imu_bp.set_attribute("noise_accel_stddev_z", "0.00")
		transform = carla.Transform(carla.Location(x=0, y=0, z=0))
		self.imu_sensor = self.world.spawn_actor(self.imu_bp, transform, attach_to=self.vehicle)
		self.imu_sensor.listen(lambda data: self.process_imu(data))
		self.actor_list.append(self.imu_sensor)
		self.imu_meas = None

	def load_perfect_sensors(self):
		self.gnss_bp = self.blueprint_library.find("sensor.other.gnss")
		self.gnss_bp.set_attribute("sensor_tick", "1")
		transform = carla.Transform(carla.Location())
		self.gnss_sensor = self.world.spawn_actor(self.gnss_bp, transform, attach_to=self.vehicle)
		self.gnss_sensor.listen(lambda data: self.process_gnss(data))
		self.actor_list.append(self.gnss_sensor)

		self.imu_bp = self.blueprint_library.find("sensor.other.imu")
		self.imu_bp.set_attribute("sensor_tick", "0.02")
		transform = carla.Transform(carla.Location())
		self.imu_sensor = self.world.spawn_actor(self.imu_bp, transform, attach_to=self.vehicle)
		self.imu_sensor.listen(lambda data: self.process_imu(data))
		self.actor_list.append(self.imu_sensor)

	def process_rgb(self, image):
		i = np.array(image.raw_data)
		i2 = i.reshape((self.im_height, self.im_width, 4))
		i3 = i2[:, :, :3]
		self.image = i3

		# self.handle_sensor()

	def process_gnss(self, data):

		if self.imu_iter > 100:
			self.gnss_meas = data
			self.timestamp = data.timestamp
			self.gnss_ready = True

			self.gnss_x, self.gnss_y, self.gnss_z = GeoLocationToLocation(
				self.gnss_meas.latitude - TOWN03_LAT,
				self.gnss_meas.longitude - TOWN03_LON,
				self.gnss_meas.altitude - TOWN03_ALT,
				scale
			)

			self.handle_sensor()
		else:
			pass

	def process_imu(self, data):

		if self.imu_iter > 100:
			self.imu_meas = data
			self.timestamp = data.timestamp
			self.imu_ready = True
			self.handle_sensor()
		else:
			pass

		self.imu_iter = self.imu_iter + 1

	def handle_sensor(self):

		# yaw2 = np.deg2rad(self.vehicle.get_transform().rotation.yaw)
		# pitch2 = np.deg2rad(self.vehicle.get_transform().rotation.pitch)
		# roll2 = np.deg2rad(self.vehicle.get_transform().rotation.roll)
		# R = transforms3d.euler.euler2mat(roll2, pitch2, yaw2, 'sxyz')
		# qi = transforms3d.quaternions.mat2quat(R)
		# print(qi)

		if self.kf.initialized is False:
			il = self.vehicle.get_location()
			iv = self.vehicle.get_velocity()
			ia = self.vehicle.get_acceleration()
			t = self.vehicle.get_transform()
			r = self.vehicle.get_transform().rotation
			R = transforms3d.euler.euler2mat(
				np.deg2rad(r.roll), np.deg2rad(r.pitch),
				np.deg2rad(r.yaw), 'sxyz'
			)
			qi = transforms3d.quaternions.mat2quat(R)

			x00 = np.array([[il.x, il.y, il.z, iv.x, iv.y, iv.z] + qi.tolist()]).T
			self.qi = np.copy(qi)
			e00 = np.diag(np.ones(10, dtype=float) * 0.0001)
			cov_vn = np.diag(np.ones(3, dtype=float) * 0.000001)
			cov_wn = np.diag(np.ones(3, dtype=float) * 0.0000000000000001)

			self.kf.initialize(x00, e00, self.timestamp, cov_vn, cov_wn)

			self.imu_ready = False
			self.gnss_ready = False

		else:
			if self.imu_ready:
				ax = self.imu_meas.accelerometer.x
				ay = self.imu_meas.accelerometer.y
				az = self.imu_meas.accelerometer.z
				# ax = ax if abs(ax) < 100 else 0
				# ay = ay if abs(ay) < 100 else 0
				# az = 0

				# angle = np.deg2rad(self.imu_meas.transform.rotation.yaw)
				# ax = np.cos(angle) * ax - np.sin(angle) * ay
				# ay = np.sin(angle) * ax + np.cos(angle) * ay
				# vn = np.array([[ax, ay, az]]).T

				roll = np.deg2rad(self.imu_meas.transform.rotation.roll)
				pitch = np.deg2rad(self.imu_meas.transform.rotation.pitch)
				yaw = np.deg2rad(self.imu_meas.transform.rotation.yaw)
				R = transforms3d.euler.euler2mat(roll, pitch, yaw, 'sxyz')
				a_sensor = np.array([[self.imu_meas.accelerometer.x,
									  self.imu_meas.accelerometer.y,
									  self.imu_meas.accelerometer.z,
									  self.imu_meas.gyroscope.x,
									  self.imu_meas.gyroscope.y,
									  self.imu_meas.gyroscope.z]]).T

				# print(self.imu_meas.gyroscope.x, self.imu_meas.gyroscope.x, self.imu_meas.gyroscope.x)
				# a = np.matmul(R, a_sensor)
				# a = a - np.array([[0, 0, 9.8]]).T
				# print(a_sensor[2, 0], a[2, 0])
				vn = a_sensor
				self.kf.predict(vn, self.timestamp)
				self.draw_accel_vector()
				# print(self.imu_meas.transform.rotation.yaw)

			elif self.gnss_ready:

				yn = np.array([[self.gnss_x, self.gnss_y, self.gnss_z]]).T
				self.kf.update(yn, self.timestamp)

				

			else:
				pass

			# if straight_trajectory is True:
			# 	self.vehicle.apply_control(carla.VehicleControl(throttle=1.5, steer=0.0))
			# else:
			# 	self.vehicle.apply_control(carla.VehicleControl(throttle=1.5, steer=-0.16))

			self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=-0.0, brake=100))
			self.log_data()

			self.imu_ready = False
			self.gnss_ready = False


	def log_data(self):

		# To get position truth, get the truth returned with the sensor data to timing issues that come from
		# getting truth positioin of get_location() calls.
		self.truth_l = self.imu_meas.transform.location if self.imu_ready else self.gnss_meas.transform.location
		self.truth_r = self.imu_meas.transform.rotation if self.imu_ready else self.gnss_meas.transform.rotation
		self.truth_geolocation = self.map.transform_to_geolocation(self.truth_l)

		# Get truth v and a from the only reasonable way possible right now, with get_velocity/get_acceleration.
		self.truth_v = self.vehicle.get_velocity()
		self.truth_a = self.vehicle.get_acceleration()
		
		self.truth_x = self.truth_l.x
		self.truth_y = self.truth_l.y
		self.truth_z = self.truth_l.z
		self.truth_vx = self.truth_v.x
		self.truth_vy = self.truth_v.y
		self.truth_vz = self.truth_v.z
		self.truth_ax = self.truth_a.x
		self.truth_ay = self.truth_a.y
		self.truth_az = self.truth_a.z

		# print(self.truth_x, self.truth_y, self.truth_z)


		self.truth_data[self.handle_iter, :] = np.array([[
			self.truth_x, self.truth_y, self.truth_z,
			self.truth_vx, self.truth_vy, self.truth_vz,
			self.truth_ax, self.truth_ay, self.truth_az,
			np.deg2rad(self.truth_r.roll), np.deg2rad(self.truth_r.pitch), np.deg2rad(self.truth_r.yaw),
			self.timestamp]]).T
		self.gnss_data[self.handle_iter, :] = np.array([[self.gnss_x, self.gnss_y, self.gnss_z]]).T
		self.imu_data[self.handle_iter, :] = np.array([[self.imu_meas.accelerometer.x, self.imu_meas.accelerometer.y, self.imu_meas.accelerometer.z]]).T
		self.kf_data[self.handle_iter, 0:10] = self.kf.xn
		self.kf_data[self.handle_iter, 10:-1] = np.array([self.kf.euler]).T
		self.kf_data[self.handle_iter, -1] = self.gnss_ready

		self.handle_iter += 1

	def stop(self):
		self.truth_data = self.truth_data[0:self.handle_iter-1, :]
		self.gnss_data = self.gnss_data[0:self.handle_iter-1, :]
		self.kf_data = self.kf_data[0:self.handle_iter-1, :]
		self.imu_data = self.imu_data[0:self.handle_iter-1, :]
		for actor in self.actor_list:
			actor.destroy()		
		cv2.destroyAllWindows()

	def draw_accel_vector(self):
		if self.imu_meas is not None:
			# start = self.imu_meas.transform.location
			# accel = self.imu_meas.accelerometer
			# angle = np.deg2rad(self.imu_meas.transform.rotation.yaw)
			# x = np.cos(angle) * accel.x - np.sin(angle) * accel.y
			# y = np.sin(angle) * accel.x + np.cos(angle) * accel.y
			# # start.z = start.z + 3

			# roll = np.deg2rad(self.imu_meas.transform.rotation.roll)
			# pitch = np.deg2rad(self.imu_meas.transform.rotation.pitch)
			# yaw = np.deg2rad(self.imu_meas.transform.rotation.yaw)
			# R = transforms3d.euler.euler2mat(roll, pitch, yaw, 'sxyz')
			# a_sensor = np.array([[accel.x, accel.y, accel.z]]).T
			# a = np.matmul(R, a_sensor)
			# # print(a)
			# a = a  - np.array([[0, 0, 9.8]]).T

			# start.z = start.z + 2
			# stop = carla.Location(x=start.x + a[0, 0], y=start.y + a[1, 0], z=start.z + a[2, 0])

			start_o = carla.Location(x=self.imu_meas.transform.location.x+3,
									 y=self.imu_meas.transform.location.y+3,
									 z=self.imu_meas.transform.location.z+2)
			x_o = carla.Location(x=start_o.x+1, y=start_o.y, z=start_o.z)
			y_o = carla.Location(x=start_o.x, y=start_o.y+1, z=start_o.z)
			z_o = carla.Location(x=start_o.x, y=start_o.y, z=start_o.z-1)

			self.world.debug.draw_arrow(start_o, x_o, life_time=0.1, color=carla.Color(255, 0, 0))
			self.world.debug.draw_arrow(start_o, y_o, life_time=0.1, color=carla.Color(0, 255, 0))
			self.world.debug.draw_arrow(start_o, z_o, life_time=0.1, color=carla.Color(0, 0, 255))

			if self.kf.initialized is True:

				# yaw = np.deg2rad(self.imu_meas.transform.rotation.yaw)
				# pitch = np.deg2rad(self.imu_meas.transform.rotation.pitch)
				# roll = np.deg2rad(self.imu_meas.transform.rotation.roll)
				# R = transforms3d.euler.euler2mat(roll, pitch, yaw, 'sxyz')
				# q = transforms3d.quaternions.mat2quat(R)
				# out = np.diag([1.0, 1.0, 1.0])
				# # out = np.matmul(R, out)
				# out[:, 0] = transforms3d.quaternions.rotate_vector(out[:, 0], q)
				# out[:, 1] = transforms3d.quaternions.rotate_vector(out[:, 1], q)
				# out[:, 2] = transforms3d.quaternions.rotate_vector(out[:, 2], q)
				# print(q)

				# # r = self.vehicle.get_transform().rotation
				# r = self.imu_meas.transform.rotation
				# R = transforms3d.euler.euler2mat(r.roll, r.pitch, r.yaw, 'sxyz')
				# qi = transforms3d.quaternions.mat2quat(R)
				# print(qi)


			# 	# print(self.qi)

				q = self.kf.xn[6:10, 0]
				out = np.diag([1.0, 1.0, 1.0])
				out[:, 0] = transforms3d.quaternions.rotate_vector(out[:, 0], q)
				out[:, 1] = transforms3d.quaternions.rotate_vector(out[:, 1], q)
				out[:, 2] = transforms3d.quaternions.rotate_vector(out[:, 2], q)
				# print(q)
	
			# # print(out)

				start_c = carla.Location(x=self.imu_meas.transform.location.x,
										 y=self.imu_meas.transform.location.y,
										 z=self.imu_meas.transform.location.z+2)
				x_c = carla.Location(x=start_c.x+out[0,0], y=start_c.y+out[1,0], z=start_c.z-out[2,0])
				y_c = carla.Location(x=start_c.x+out[0,1], y=start_c.y+out[1,1], z=start_c.z-out[2,1])
				z_c = carla.Location(x=start_c.x+out[0,2], y=start_c.y+out[1,2], z=start_c.z-out[2,2])
				self.world.debug.draw_arrow(start_c, x_c, life_time=0.1, color=carla.Color(255, 0, 0))
				self.world.debug.draw_arrow(start_c, y_c, life_time=0.1, color=carla.Color(0, 255, 0))
				self.world.debug.draw_arrow(start_c, z_c, life_time=0.1, color=carla.Color(0, 0, 255))


def plot_results_location(car):

	gnss_update = car.kf_data[:, -1].astype(bool)[:, 0]

	i = np.arange(np.shape(car.truth_data)[0])

	plt.figure()
	plt.title('x')
	plt.plot(i, car.truth_data[:, 0], 'g+-')
	plt.plot(i, car.kf_data[:, 0], 'r+-')
	plt.plot(i[gnss_update==True], car.kf_data[gnss_update==True, 0], 'ro')

	plt.figure()
	plt.title('y')
	plt.plot(car.truth_data[:, 1], 'g+-')
	plt.plot(car.kf_data[:, 1], 'r+-')
	plt.plot(i[gnss_update==True], car.kf_data[gnss_update==True, 1], 'ro')

	plt.figure()
	plt.title('z')
	plt.plot(car.truth_data[:, 2], 'g+-')
	plt.plot(car.kf_data[:, 2], 'r+-')
	plt.plot(i[gnss_update==True], car.kf_data[gnss_update==True, 2], 'ro')

	plt.figure()
	plt.title('x error')
	plt.plot(car.truth_data[:, 0] - car.kf_data[:, 0], '+-')

	plt.figure()
	plt.title('y error')
	plt.plot(car.truth_data[:, 1] - car.kf_data[:, 1], '+-')

	plt.figure()
	plt.title('z error')
	plt.plot(car.truth_data[:, 2] - car.kf_data[:, 2], '+-')


def plot_results_euler(car):

	gnss_update = car.kf_data[:, -1].astype(bool)[:, 0]

	i = np.arange(np.shape(car.truth_data)[0])

	plt.figure()
	plt.title('x')
	plt.plot(i, car.truth_data[:, 9], 'g+-')
	plt.plot(i, car.kf_data[:, 10], 'r+-')
	# plt.plot(i[gnss_update==True], car.kf_data[gnss_update==True, 0], 'ro')

	plt.figure()
	plt.title('y')
	plt.plot(car.truth_data[:, 10], 'g+-')
	plt.plot(car.kf_data[:, 11], 'r+-')
	# plt.plot(i[gnss_update==True], car.kf_data[gnss_update==True, 1], 'ro')

	plt.figure()
	plt.title('z')
	plt.plot(car.truth_data[:, 11], 'g+-')
	plt.plot(car.kf_data[:, 12], 'r+-')
	# plt.plot(i[gnss_update==True], car.kf_data[gnss_update==True, 2], 'ro')

	# plt.figure()
	# plt.title('x error')
	# plt.plot(car.truth_data[:, 0] - car.kf_data[:, 0], '+-')

	# plt.figure()
	# plt.title('y error')
	# plt.plot(car.truth_data[:, 1] - car.kf_data[:, 1], '+-')

	# plt.figure()
	# plt.title('z error')
	# plt.plot(car.truth_data[:, 2] - car.kf_data[:, 2], '+-')

def plot_results_acceleration(car):

	plt.figure()
	plt.title('ax')
	plt.plot(car.imu_data[:, 0], 'r+-')
	plt.plot(car.truth_data[:, 6], 'g+-')

	plt.figure()
	plt.title('ay')
	plt.plot(car.imu_data[:, 1], 'r+-')
	plt.plot(car.truth_data[:, 7], 'g+-')

	plt.figure()
	plt.title('az')
	plt.plot(car.imu_data[:, 2], 'r+-')
	plt.plot(car.truth_data[:, 8], 'g+-')


if __name__ == '__main__':

	car = CarEnv()
	car.reset()

	try:
		while(True):
			if car.image is not None:
				cv2.imshow('', car.image)
				cv2.waitKey(1)
			time.sleep(0.01)	
	except (KeyboardInterrupt, SystemExit):
		print('Exiting')
	finally:
		car.stop()

	# kf = ImuKF()
	# x = []
	# kf.initialize([])

	IPython.embed()
