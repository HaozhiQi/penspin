import os
import copy
import rospy
import threading

import open3d as o3d
import k4a 

from sensor_msgs.msg import JointState
from std_msgs.msg import Header

from xarm_allegro_model_helper import XarmAllegro_ModelCalculate
from .utils import *
from .utils import _action_hora2allegro,_obs_allegro2hora
# Have to include this part to find the dynamic library
from ctypes import cdll
_lib_dir = os.path.abspath(os.path.join(k4a.__file__, "../_libs"))
print(_lib_dir)
depth_lib = cdll.LoadLibrary(os.path.join(_lib_dir, 'libdepthengine.so.2.0'))

# # ############Add SAM to the real robot ############
# from .ground_sam import *
############Add Tracking_SAM to the real robot ############
from Tracking_SAM import tracking_SAM

class RealRobot:
	def __init__(self, sam=False, replay=False):
		self.sam = sam
		self.replay = replay

		self.dof_lower = np.array([-0.47, -0.196, -0.174, -0.227,
								-0.47,-0.196, -0.174, -0.227, 
								-0.47, -0.196,-0.174, -0.227,
								0.7,0.3, -0.189, -0.162,])
		self.dof_upper = np.array([0.47, 1.61, 1.709, 1.618,
								0.47,1.61, 1.709, 1.618, 
								0.47, 1.61,1.709, 1.618,
								1.396, 1.163, 1.644, 1.719,])

		self.stepcount = 0
		# Cache
		# tf = np.eye(4)
		# rotate = transforms3d.euler.euler2mat(0, np.pi/6, np.pi*195/180)
		# tf[:3, :3] = rotate 
		# tf[:,3]=np.array([ 1, 0.1, 0.6 ,1] )
		camera_adj = np.array([0.5562,-0.0,0.125+0.0548+0.015+0.05])
		#camera pose is cam2endeffect,  need to adjust to the robot base

		#DexPen cam2hand
		# self.cam2hand = np.array([[ 0.0220731,  -0.98910943, -0.14551741, 0.17147789],
        # [ 0.86822792, -0.05319939,  0.49330529, -0.09902794-0.001],
        # [-0.49567435, -0.13723106,  0.85759814, -0.32635988],
        # [ 0.,          0.,          0.,          1.        ]])
		# self.cam2hand = np.array([[ 0.0220731,  -0.98910943, -0.14551741,  0.15410028-0.036],          # adjust along the Blue Z axis
 		# 						[ 0.86822792, -0.05319939,  0.49330529, -0.08798173-0.05],              # adjust along the Green y axis
 		# 						[-0.49567435, -0.13723106,  0.85759814, -0.32635111-0.045],                  # adjust along the Red x axis
 		# 						[ 0.,          0.,          0.,          1.        ]])

		self.cam2hand = np.array([[-0.06086204, -0.85466074, -0.51560728,  0.22040379-0.035],            # adjust along the Blue Z axis
 								[ 0.90264606, -0.26762265,  0.33705821, -0.05263305-0.038],              # adjust along the Green y axis
 								[-0.42605858, -0.44489682,  0.78774421, -0.26671011-0.035],              # adjust along the Red x axis
 								[ 0.,          0.,          0.,          1.,        ]])
		
		gl2cv_homo = np.eye(4)
		self.cam2hand = self.cam2hand @ gl2cv_homo
		
		#DexPen initial
		self.hand_initial = INIT_QPOS

		self.arm_initial = np.array([210, 70, -100, 180, 150, -30])

		if not self.replay:

			self.hand_pose_pub = rospy.Publisher("/cmd_joint_states", JointState, queue_size=100, latch=True)
			self.rate = rospy.Rate(18)

			# RealSense camera
			self.device = k4a.Device.open()
			self.device_config = k4a.DeviceConfiguration(color_format=k4a.EImageFormat.COLOR_BGRA32,
													color_resolution=k4a.EColorResolution.RES_720P,
													depth_mode=k4a.EDepthMode.WFOV_2X2BINNED,
													camera_fps=k4a.EFramesPerSecond.FPS_30,
													synchronized_images_only=True,
													depth_delay_off_color_usec=0,
													wired_sync_mode=k4a.EWiredSyncMode.STANDALONE,
													subordinate_delay_off_master_usec=0,
													disable_streaming_indicator=False)

			self.device.start_cameras(self.device_config)
			self.camera_info = self.device.get_calibration(depth_mode=self.device_config.depth_mode, color_resolution= self.device_config.color_resolution)
			#print(camera_info)
			self.transformation = k4a.Transformation(self.camera_info)

			self.robot_qpos = np.zeros([16])
			self.robot_qvel = np.zeros([16])
			self.has_check_joint_order = False
			self._allegro_state = rospy.wait_for_message("/joint_states_pub", JointState, timeout=1)
			self.thread =threading.Thread(target=self.run_async_thread)
			self._lock = threading.Lock()
			self.thread.start()
			self.allegro_state_obs = self.allegro_state.position
			self.filename ="./assets/round_tip/allegro_hand_right_fsr_round_thin.urdf"
			self.robot = XarmAllegro_ModelCalculate(self.filename)
		
			self.load_mesh()


			#####Running the tactile sensors#####
			self.fsrValue = np.zeros(32)

			PORT = "/dev/ttyACM0"
			BAUD = 9600
			# serDev = serial.Serial(PORT, BAUD, timeout=1)

			self.exitThread = False

			def readThread(self,serDev):
				
				print('serial read thread started')
				while not self.exitThread:

					try:
						if serDev.isOpen():
							line = serDev.readline().decode('utf-8').rstrip()
							self.fsrValue =np.array(eval(line))
						else:
							serDev.open()
					except Exception as ex:
						print(ex)

			# self.serialThread = threading.Thread(target=readThread, args=(self,serDev))
			# self.serialThread.daemon = True
			# self.serialThread.start()

		# Use the following to load Tracking_SAM model
		if self.sam:
			sam_checkpoint = "./Tracking_SAM/pretrained_weights/sam_vit_h_4b8939.pth"
			aot_checkpoint = "./Tracking_SAM/pretrained_weights/AOTT_PRE_YTB_DAV.pth"
			grounding_dino_checkpoint = "./Tracking_SAM/pretrained_weights/groundingdino_swint_ogc.pth"
			self.my_tracking_SAM = tracking_SAM.main_tracker(sam_checkpoint, aot_checkpoint, grounding_dino_checkpoint)
			# Fro pen rotation


	def run_async_thread(self):
		self.sub = rospy.Subscriber("/joint_states_pub", JointState, self.cb_joint_pos_update)
		print("Begin ROS event loop.")
		rospy.spin()
	
	def cb_joint_pos_update(self, joint_state: JointState):

		with self._lock:
			self._allegro_state = joint_state

	@property
	def allegro_state(self):
		with self._lock:
			return self._allegro_state		
	
	# @timeit
	def set_action(self , action):
	
		hand_pose = JointState()
		hand_pose.header = Header()
		hand_pose.header.frame_id = "grasp"
		hand_pose.name = ["grasp1"]
		hand_pose.position = action
		self.hand_pose_pub.publish(hand_pose)
		self.rate.sleep()
		return
	
	def get_observation(self):

		obs  = np.zeros(64)
		allegro_state = np.array(self.allegro_state.position).copy()
		self.allegro_state_obs = allegro_state
		hand_qpos_raw =allegro_state[0:16]
		obs[0:16] = hand_qpos_raw

		return obs
	
	def get_tactile(self):
		visual_fsrvalue=np.zeros(32,dtype=np.int8)
		tactile = np.zeros(32)
		fsr = fsr_align(self.fsrValue)
		for i in range(len(fsr)):
			if fsr[i] > 100:
				tactile[i]=1
				visual_fsrvalue[i]=i+1
		print(np.where(tactile == 1))
		return tactile, visual_fsrvalue
	
	def get_objects_ends(self, replay_rgb_c2d=None, replay_depth=None, replay_pc=None):

		pen_pc = self.get_point_cloud_with_Tracking_SAM(replay_rgb_c2d, replay_depth, replay_pc)
		pen_pc = self.pc_factory(pen_pc)
		
		new_points = pen_pc[:,:3]
		new_points = o3d.utility.Vector3dVector(new_points)
		
		pen_pcd = o3d.geometry.PointCloud()
		pen_pcd.points = new_points
		# voxel_down_pc = pen_pcd.uniform_down_sample(every_k_points=10)
		# cl, ind = pen_pcd.remove_statistical_outlier(nb_neighbors=10,std_ratio=2)
		cl, ind = pen_pcd.remove_radius_outlier(nb_points=16,radius=0.05)
		
		# Randomly Pick a point
		ix = np.random.randint(0,len(ind))
		ix = 1
		point1 = pen_pcd.points[ind[ix]]
		point_end1_id = get_furthest_point(point1,cl.points)
		point_end2_id = get_furthest_point(cl.points[point_end1_id],cl.points)
		obj_ends = np.array([cl.points[point_end1_id],cl.points[point_end2_id]])

		# Compensate
		d = [-0.025/2,-0.025/2] if obj_ends[0][2] > obj_ends[1][2] else [0.025/2,0.025/2]
		obj_ends[0][2] += d[0]
		obj_ends[1][2] += d[1]

		id_list = [i for i in range(len(ind))]
		# Make sure tracking the 2ends in real world
		assert len(id_list) == 2 
		id_list.remove(point_end1_id)
		id_list.remove(point_end2_id)
		remain_points = np.array(cl.points)[id_list]
		remain_points = o3d.utility.Vector3dVector(remain_points)
		cl.points = remain_points

		return cl, obj_ends

	
	def get_point_cloud_with_Tracking_SAM(self, replay_rgb_c2d=None, replay_depth=None, replay_pc=None):

		if not self.replay:
			for x in range(1):
				frames = self.device.get_capture(-1)

			image = frames.color.data
			image = self.transformation.color_image_to_depth_camera(frames.depth,frames.color)
			image_np_rgb = rgba2rgb(image.data)
			image_np_rgb = np.asarray(image_np_rgb).astype(np.uint8)

		else: 
			image_np_rgb = replay_rgb_c2d

		# plt.imshow(image_np_rgb)
		# plt.show()	
		
		if self.my_tracking_SAM.is_tracking():
			# start_cp = time.time()
			masks = self.my_tracking_SAM.propagate_one_frame(image_np_rgb) 

			mask_depth = frames.depth.data.copy() if not self.replay else replay_depth
			mask_depth[masks <= 0] = 0
			# plt.imshow(mask_depth)
			# plt.show()
			if not self.replay:
				pc_image = self.transformation.depth_image_to_point_cloud(frames.depth, k4a.ECalibrationType.DEPTH)
				pc = np.asanyarray(pc_image.data) / 1000
			else:
				pc = replay_pc
			pc[masks<=0] = [0,0,0]
			pc = np.reshape(pc, [-1, 3])
			points = pc
		else:
			print("##########Annotate_init_frame!!!##########")
			self.my_tracking_SAM.reset_engine()
			# self.my_tracking_SAM.annotate_init_frame(image_np_rgb, method='dino', category_name='small white stick on the hand')
			self.my_tracking_SAM.annotate_init_frame(image_np_rgb, method='clicking')
			points = None

		return points
	
	def get_camera_pointcloud(self,debug=False):
		for x in range(1):
			frames = self.device.get_capture(-1)

		pc_image = self.transformation.depth_image_to_point_cloud(frames.depth, k4a.ECalibrationType.DEPTH)

		pc = np.asanyarray(pc_image.data) / 1000
		pc = np.reshape(pc, [-1, 3])
		points = pc

		return points
	
	def get_rgb_d(self):
		for x in range(1):
			frames = self.device.get_capture(-1)

		image = frames.color.data
		depth = frames.depth.data.copy()
		rgb_ori = image.copy()
		image = self.transformation.color_image_to_depth_camera(frames.depth,frames.color)
		image_np_rgb = rgba2rgb(image.data)
		image_np_rgb = np.asarray(image_np_rgb).astype(np.uint8)
		rgb_c2d = image_np_rgb

		pc_image = self.transformation.depth_image_to_point_cloud(frames.depth, k4a.ECalibrationType.DEPTH)
		pc = np.asanyarray(pc_image.data) / 1000

		return rgb_ori, rgb_c2d, depth, pc

	def pc_factory(self, pc):
		points = pc
		small_select_indices = np.zeros([4, 4])
		small_select_indices[0, 0] = 1
		select_indices = np.tile(small_select_indices, (128, 128)).flatten()
		select_indices = np.nonzero(select_indices)[0]
		vertices = points[select_indices]
		points = np.array([list(x) for x in vertices])

		valid_bool = np.linalg.norm(points, axis=1) < 2.0
		points = points[valid_bool]

		new_points = self.process_robot_pc(points, 512)

		return new_points
	
	def process_robot_pc(self, cloud: np.ndarray, num_points: int, segmentation=None) -> np.ndarray:
		"""pc: nxm, camera_pose: 4x4"""
		if segmentation is not None:
			raise NotImplementedError
		
		camera_pose = self.cam2hand
		pc = cloud[..., :3]
		pc = pc @ camera_pose[:3, :3].T + camera_pose[:3, 3]
		thetax = 90/180*math.pi
		thetay = -90/180*math.pi
		thetaz = 0/180*math.pi
		roty = np.array([[math.cos(thetay), 0, math.sin(thetay)],
					[0, 1, 0],
					[-math.sin(thetay), 0, math.cos(thetay)]])
		rotz = np.array([[math.cos(thetaz), -math.sin(thetaz), 0],
					[math.sin(thetaz), math.cos(thetaz), 0],
					[0, 0, 1]])
		rotx = np.array([[1, 0, 0],
					[0, math.cos(thetax), -math.sin(thetax)],
					[0, math.sin(thetax), math.cos(thetax)]])
		pc = (roty @ rotx @ pc.T).T

		bound = [-0.2, 0.09, -0.1, 0.1, -0.05, 0.15]

		# remove robot table
		within_bound_x = (pc[..., 0] > bound[0]) & (pc[..., 0] < bound[1])
		within_bound_y = (pc[..., 1] > bound[2]) & (pc[..., 1] < bound[3])
		within_bound_z = (pc[..., 2] > bound[4]) & (pc[..., 2] < bound[5])
		within_bound = np.nonzero(np.logical_and.reduce((within_bound_x, within_bound_y, within_bound_z)))[0]

		num_index = len(within_bound)
		print(num_index)
		if num_index == 0:
			return np.zeros([num_points, 3])
		if num_index < num_points:
			indices = np.concatenate([within_bound, np.ones(num_points - num_index, dtype=np.int32) * within_bound[0]])
		else:
			indices = within_bound[np.random.permutation(num_index)[:num_points]]
		cloud = pc[indices, :]
		cloud = np.concatenate([cloud, np.zeros((cloud.shape[0], 3))], axis=1)
		cloud[:, 4] = 1
		# print(cloud.shape)
		# cloud = np.concatenate([cloud, np.zeros((cloud.shape[0], 2))], axis=)

		return cloud
	
	def load_mesh(self):
		self.meshes = {}
		for name in STL_FILE_DICT.keys():
			try:
				mesh = o3d.io.read_triangle_mesh(STL_FILE_DICT[name])
				self.meshes[name] = mesh
			except:
				continue
	
	def update_imagnation(self):
		all_pc = []
		meshes = []
		
		qpos = np.array(self.allegro_state.position).copy()
		rb_dict = image_dict
		triggered_contact_name = []
		fsr_pc_tmp = np.zeros((16,8,3))
		fsr_count = 0
		tactile, visual_tactile = self.get_tactile()
		for i in visual_tactile:
			if i != 0:
				triggered_contact_name.append(contact_sensor_names[i-1])
		#print(triggered_contact_name)
		contact_box_size = [0.003011, 0.01, 0.02]

		for name, index in rb_dict.items():
			link_state = self.robot.compute_xarmfk_pose(qpos,index)  # position 0:3, rotation 3:7
			link_tf = link_state.to_transformation_matrix()
			# print(index)
			
			rot1 =link_tf[:3,:3]
			
			# if name in triggered_contact_name:
			# 	link_name = name[:-4]
			# 	#print(link_name)
			# 	mesh = copy.deepcopy(self.meshes[link_name])  # .copy()
			# 	# fsr_sample_pts = np.asarray(mesh.sample_points_uniformly(number_of_points=8).points)
			# 	fsr_sample_pts = np.array([[random.uniform(-contact_box_size[0] / 2, contact_box_size[0] / 2) for _ in range(8)],
            #                             [random.uniform(-contact_box_size[1] / 2, contact_box_size[1] / 2) for _ in range(8)],
            #                             [random.uniform(-contact_box_size[2] / 2, contact_box_size[2] / 2) for _ in range(8)]])
			# 	fsr_sample_pts = np.transpose(fsr_sample_pts)

			# 	fsr_sample_pts = fsr_sample_pts @ rot1.T
			# 	fsr_sample_pts = fsr_sample_pts + link_tf[:3,3]

			# 	#fsr_pc_tmp.append(np.asanyarray(fsr_sample_pts))
			# 	fsr_pc_tmp[fsr_count,:,:] = fsr_sample_pts
			# 	fsr_count = fsr_count + 1
			# 	continue

			if 'fsr' in name or name[:4] == 'palm' or name == 'world' or name == 'wrist':
				continue
			
			mesh = copy.deepcopy(self.meshes[name])  # .copy()
			
			# print(link_tf[:3,3])
			if name[-3:] == 'tip' and self.filename == "assets/variation/allegro_hand_right_fsr.urdf":
				
				mesh.scale(0.001, center=np.array([0, 0, 0]))
				rot0 = np.array([[0, -1, 0],
									[1, 0, 0],
									[0, 0, 1]])
				mesh.rotate(rot0, center=np.array([0, 0, 0.02]))
				mesh.rotate(rot1, center=np.array([0, 0, 0.02]))
				mesh.translate(link_tf[:3,3] + np.array([0, 0, -0.02]))
			else:
				mesh.rotate(rot1, center=np.array([0, 0, 0]))
				mesh.translate(link_tf[:3,3])
			# print(name)
			all_pc.append(np.asarray(mesh.sample_points_uniformly(number_of_points=8).points))
		
		fsr_pc_tmp = np.array(fsr_pc_tmp)
		#print(fsr_pc_tmp)
		all_pc = np.array(all_pc)
		#print(all_pc.shape)
# 		palm_inv_pose = np.array([[-5.9278059e-01,  1.3379070e-03,  8.0536294e-01, -4.7972274e-01],
#  [-3.5226127e-04,  9.9999809e-01, -1.9205232e-03, -8.8860172e-05],
#  [-8.0536395e-01, -1.4221469e-03, -5.9277904e-01, -3.1034407e-01],
#  [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]])
		# palm_inv_pose = np.array([[-8.68842742e-01,  1.11324993e-01, -4.82409615e-01, -4.36478234e-01],
        #                           [ 4.94263108e-01,  1.38811489e-01, -8.58158115e-01,  9.31470505e-04],
        #                           [-2.85704492e-02, -9.84041725e-01, -1.75629192e-01,  3.66090257e-01],
        #                           [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
		one_hot = np.zeros((all_pc.shape[0], all_pc.shape[1], 3))
		one_hot_tac = np.zeros((fsr_pc_tmp.shape[0], fsr_pc_tmp.shape[1], 3))
		one_hot[:, :, 0] = 1
		one_hot_tac[:, :, 2] = 1
		all_pc = np.concatenate([all_pc, one_hot], axis=-1).reshape(-1, 6)
		fsr_pc_tmp = np.concatenate([fsr_pc_tmp, one_hot_tac], axis=-1).reshape(-1, 6)
		# all_pc[:, :3] = all_pc[:, :3] @ palm_inv_pose[:3, :3].T + palm_inv_pose[:3, 3]
		# fsr_pc_tmp[:, :3] = fsr_pc_tmp[:, :3] @ palm_inv_pose[:3, :3].T + palm_inv_pose[:3, 3]
		thetax = 90/180*math.pi
		thetay = -90/180*math.pi
		thetaz = 0/180*math.pi
		roty = np.array([[math.cos(thetay), 0, math.sin(thetay)],
				[0, 1, 0],
				[-math.sin(thetay), 0, math.cos(thetay)]])
		rotz = np.array([[math.cos(thetaz), -math.sin(thetaz), 0],
				[math.sin(thetaz), math.cos(thetaz), 0],
				[0, 0, 1]])
		rotx = np.array([[1, 0, 0],
				[0, math.cos(thetax), -math.sin(thetax)],
				[0, math.sin(thetax), math.cos(thetax)]])
		all_pc[:, :3] = (roty @ rotx @ all_pc[:, :3].T).T
		fsr_pc_tmp[:, :3] = (roty @ rotx @ fsr_pc_tmp[:, :3].T).T


		return all_pc, fsr_pc_tmp

	def xarm_init_set(self):
		# self.arm.clean_error()
		# self.arm.motion_enable(enable=True)
		# self.arm.set_mode(0)
		# self.arm.set_state(state=0)
		# self.arm.set_servo_angle(angle=qpos_arm, speed=0.5, is_radian=False,wait=True)
		now = rospy.Time.now()
		hand_pose = JointState()
		hand_pose.header = Header()
		hand_pose.header.frame_id = "init"
		hand_pose.name = ["init"]
		hand_pose.position = _action_hora2allegro(self.hand_initial)
		self.hand_pose_pub.publish(hand_pose)
		# print(self.arm.get_servo_angle(is_radian=True)[1][0:6])

	def get_tip_pos(self, qpos):
		tip_pos = []
		rb_dict = image_dict
		thetay = -np.pi / 2
		roty = np.array([[math.cos(thetay), 0, math.sin(thetay)],
				[0, 1, 0],
				[-math.sin(thetay), 0, math.cos(thetay)]])
		thetax = np.pi / 2
		rotx = np.array([[1, 0, 0],
				[0, math.cos(thetax), -math.sin(thetax)],
				[0, math.sin(thetax), math.cos(thetax)]])
				
		for name, index in rb_dict.items():
			if name == "palm":
				link_state = self.robot.compute_xarmfk_pose(qpos,index)
				palm_pos = link_state.p
			if name == "link_3.0_tip":
				link_state = self.robot.compute_xarmfk_pose(qpos,index)
				index_pos = link_state.p
			if name == "link_15.0_tip":
				link_state = self.robot.compute_xarmfk_pose(qpos,index)  # position 0:3, rotation 3:7
				thumb_pos  = link_state.p
			if name == "link_7.0_tip":
				link_state = self.robot.compute_xarmfk_pose(qpos,index)  # position 0:3, rotation 3:7
				middle_pos = link_state.p
			if name == "link_11.0_tip":
				link_state = self.robot.compute_xarmfk_pose(qpos,index)  # position 0:3, rotation 3:7
				ring_pos = link_state.p
		
		tip_pos = np.concatenate([[index_pos-palm_pos],[thumb_pos-palm_pos],[middle_pos-palm_pos],[ring_pos-palm_pos]])
		tip_pos = ((roty @ rotx @ tip_pos.T).T).reshape(-1)
		tip_pos =  tip_pos + np.array([0,0.07,0.5]*4)

		return tip_pos