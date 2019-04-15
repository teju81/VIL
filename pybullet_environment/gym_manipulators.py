from gym.envs.pybullet.pybullet_env import PybulletMujocoXmlEnv
import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import os, sys
import math
import datetime
import time

class PybulletReacherEnv(PybulletMujocoXmlEnv):
	def __init__(self):
		PybulletMujocoXmlEnv.__init__(self, 'assets/reacher.xml', 'body0', action_dim=6, obs_dim=9)
		#self._p.loadURDF("table/table.urdf", basePosition=[0, 0, 0.0], baseOrientation=[0, 0, 0, 1])

	TARG_LIMIT = 0.15
	epsilon = 0.001
	def robot_specific_reset(self, initial_state = None):
		#self.jdict["target_x"].reset_current_position(self.np_random.uniform( low=-0.27, high=0.27),0)
		#self.jdict["target_y"].reset_current_position(self.np_random.uniform( low=-0.27, high=0.27),0)
		self.fingertip = self.parts["fingertip"]
		#self.target	= self.parts["target"]
		self.central_joint = self.jdict["joint0"]
		self.elbow_joint   = self.jdict["joint1"]
		if initial_state is None:
			self.central_joint.reset_current_position(self.np_random.uniform( low=-math.pi, high=math.pi),0)
			self.elbow_joint.reset_current_position(self.np_random.uniform( low=-math.pi, high=math.pi),0)
		else:
			self.central_joint.reset_current_position(initial_state[0][0],initial_state[0][1])
			self.elbow_joint.reset_current_position(initial_state[1][0],initial_state[1][1])

		self.origin = np.array([0,0,0.01])


	def apply_action(self, a):
		# assert( np.isfinite(a).all() )
		# self.central_joint.set_motor_torque(a[0])
		# self.elbow_joint.set_motor_torque(a[1])
		control_mode = 0
		forces = [10, 10]
		target_velocities = [0, 0]
		position_gains = [1, 1]
		velocity_gains = [1, 1]
		max_velocities = [10, 10]
		joint_indices = [0, 2]
		pos = a[0:2]
		vel=a[2:4]
		torques = a[4:6]
		if control_mode == 0:
			for j in range(2):
				self._p.setJointMotorControl2(bodyIndex=1, jointIndex=joint_indices[j], controlMode=self._p.POSITION_CONTROL, targetPosition=pos[j], \
												targetVelocity=vel[j], positionGain = position_gains[j], \
												velocityGain = velocity_gains[j], maxVelocity=max_velocities[j], force=forces[j])
		elif control_mode == 1:
			for j in range(2):
				self._p.setJointMotorControl2(bodyIndex=1, jointIndex=joint_indices[j], controlMode=self._p.VELOCITY_CONTROL, targetVelocity=vel[j])
		else:
			for j in range(2):
				self._p.setJointMotorControlArray(bodyIndex=1, jointIndex=joint_indices[j], controlMode=self._p.TORQUE_CONTROL, force=torques[j])

	def calc_state(self):
		self.theta,	 self.theta_dot = self.central_joint.current_relative_position()
		self.gamma, self.gamma_dot = self.elbow_joint.current_relative_position()
		cjx,cjv,cjt = self.central_joint.current_position()
		ejx,ejv,ejt = self.elbow_joint.current_position()
		eep = self.fingertip.pose().xyz()
		#target_x, _,_ = self.jdict["target_x"].current_position()
		#target_y, _,_ = self.jdict["target_y"].current_position()
		self.to_target_vec = np.array(self.fingertip.pose().xyz()) - np.array(self.target_position)
		return np.array([cjx, ejx, cjv, ejv, cjt , ejt, eep[0], eep[1], eep[2]])

	def calc_potential(self):
		return -100 * np.linalg.norm(self.to_target_vec)

	def step_action(self, a):
		self.apply_action(a)
		self._p.stepSimulation()

		state = self.calc_state()  # sets self.to_target_vec

		potential_old = self.potential
		self.potential = self.calc_potential()

		electricity_cost = (
			-0.10*(np.abs(a[0]*self.theta_dot) + np.abs(a[1]*self.gamma_dot))  # work torque*angular_velocity
			-0.01*(np.abs(a[0]) + np.abs(a[1]))								# stall torque require some energy
			)
		stuck_joint_cost = -0.1 if np.abs(np.abs(self.gamma)-1) < 0.01 else 0.0
		self.rewards = [float(self.potential - potential_old), float(electricity_cost), float(stuck_joint_cost)]
		self.HUD(state, a, False)
		return state, sum(self.rewards), False, {} # Obs, reward, done, info

	# def get_target_theta(self):
	# 	if self.target_position[1] == 0 and self.target_position[0] == 0:
	# 		self.target_theta = 0
	# 	elif self.target_position[1] != 0 and self.target_position[0] != 0:
	# 		self.target_theta = math.atan(self.target_position[1] / self.target_position[0])
	# 		if self.target_position[1] > 0 and self.target_position[0] < 0: #Q2
	# 			self.target_theta = self.target_theta + math.pi
	# 		elif self.target_position[1] < 0 and self.target_position[0] < 0: #Q3
	# 			self.target_theta = self.target_theta + math.pi
	# 		elif self.target_position[0] > 0 and self.target_position[1] < 0: #Q4
	# 			self.target_theta = self.target_theta + 2*math.pi
	# 	else:
	# 		if self.target_position[1] > 0 and self.target_position[0] == 0:
	# 			self.target_theta = 0.5*math.pi
	# 		elif self.target_position[1] < 0 and self.target_position[0] == 0:
	# 			self.target_theta = 1.5*math.pi
	# 		elif self.target_position[0] > 0 and self.target_position[1] == 0:
	# 			self.target_theta = 0
	# 		else:
	# 			self.target_theta = math.pi

	def step_action_to_get_to_target(self, a):
		jd=[0.001,0.001]
		joint_poses = self._p.calculateInverseKinematics(self.model[1], 4, self.target_position, jointDamping=jd, maxNumIterations = 1000)
		forces = [10, 10]
		target_velocities = [0, 0]
		position_gains = [1, 1]
		velocity_gains = [1, 1]
		max_velocities = [10, 10]
		joint_indices = [0, 2]
		for j in range(2):
			self._p.setJointMotorControl2(bodyIndex=1, jointIndex=joint_indices[j], controlMode=self._p.POSITION_CONTROL, targetPosition=joint_poses[j], \
											targetVelocity=target_velocities[j], positionGain = position_gains[j], \
											velocityGain = velocity_gains[j], maxVelocity=max_velocities[j], force=forces[j])
		# FASTER than multiple calls as done above: But no control over max velocities
		# self._p.setJointMotorControlArray(bodyIndex=6,jointIndices=joint_indices, controlMode=self._p.POSITION_CONTROL, \
		# 								targetPositions=joint_poses, targetVelocities=target_velocities, positionGains=position_gains,\
		# 								velocityGains=velocity_gains, forces=forces)
		self.potential = self.calc_potential()
		self._p.stepSimulation()
		state = self.calc_state()  # sets self.to_target_vec

		potential_old = self.potential

		electricity_cost = (
			-0.10*(np.abs(a[0]*self.theta_dot) + np.abs(a[1]*self.gamma_dot))  # work torque*angular_velocity
			-0.01*(np.abs(a[0]) + np.abs(a[1]))								# stall torque require some energy
			)
		stuck_joint_cost = -0.1 if np.abs(np.abs(self.gamma)-1) < 0.01 else 0.0
		self.rewards = [float(self.potential - potential_old), float(electricity_cost), float(stuck_joint_cost)]
		self.HUD(state, a, False)
		return state, sum(self.rewards), False, {} # Obs, reward, done, info

	def get_joint_angle_trajectories(self):
		self.theta1 = range(0.0,0.5*math.pi, 0.05*math.pi)
		self.theta2 = range(0.0,0.5*math.pi, 0.05*math.pi)

	def step(self, a):
		if self.training_phase == False:
			Obs, reward, done, info = self.step_action(a)
		else:
			Obs, reward, done, info = self.step_action_to_get_to_target(a)
		return Obs, reward, done, info

class PybulletPusherEnv(PybulletMujocoXmlEnv):
	min_target_placement_radius = 0.5
	max_target_placement_radius = 0.8
	min_object_to_target_distance = 0.1
	max_object_to_target_distance = 0.4
	def __init__(self):
		PybulletMujocoXmlEnv.__init__(self, 'assets/pusher.xml', 'body0', action_dim=7, obs_dim=5)

	def robot_specific_reset(self):
		# parts
		self.fingertip = self.parts["fingertip"]
		self.target = self.parts["target"]
		self.object = self.parts["object"]

		# joints
		self.shoulder_pan_joint = self.jdict["shoulder_pan_joint"]
		self.shoulder_lift_joint = self.jdict["shoulder_lift_joint"]
		self.upper_arm_roll_joint = self.jdict["upper_arm_roll_joint"]
		self.elbow_flex_joint = self.jdict["elbow_flex_joint"]
		self.forearm_roll_joint = self.jdict["forearm_roll_joint"]
		self.wrist_flex_joint = self.jdict["wrist_flex_joint"]
		self.wrist_roll_joint = self.jdict["wrist_roll_joint"]

		self.target_pos = np.concatenate([
			self.np_random.uniform(low=-1, high=1, size=1),
			self.np_random.uniform(low=-1, high=1, size=1)
		])

		# make length of vector between min and max_target_placement_radius
		self.target_pos = self.target_pos \
						  / np.linalg.norm(self.target_pos) \
						  * self.np_random.uniform(low=self.min_target_placement_radius,
												   high=self.max_target_placement_radius, size=1)

		self.object_pos = np.concatenate([
			self.np_random.uniform(low=-1, high=1, size=1),
			self.np_random.uniform(low=-1, high=1, size=1)
		])

		# make length of vector between min and max_object_to_target_distance
		self.object_pos = self.object_pos \
						  / np.linalg.norm(self.object_pos - self.target_pos) \
						  * self.np_random.uniform(low=self.min_object_to_target_distance,
												   high=self.max_object_to_target_distance, size=1)

		# set position of objects
		self.zero_offset = np.array([0.45, 0.55])
		self.jdict["target_x"].reset_current_position(self.target_pos[0] - self.zero_offset[0], 0)
		self.jdict["target_y"].reset_current_position(self.target_pos[1] - self.zero_offset[1], 0)
		self.jdict["object_x"].reset_current_position(self.object_pos[0] - self.zero_offset[0], 0)
		self.jdict["object_y"].reset_current_position(self.object_pos[1] - self.zero_offset[1], 0)

		# randomize all joints TODO: Will this work or do we have to constrain this resetting in some way?
		self.shoulder_pan_joint.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
		self.shoulder_lift_joint.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
		self.upper_arm_roll_joint.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
		self.elbow_flex_joint.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
		self.forearm_roll_joint.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
		self.wrist_flex_joint.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)
		self.wrist_roll_joint.reset_current_position(self.np_random.uniform(low=-3.14, high=3.14), 0)

	def apply_action(self, a):
		assert (np.isfinite(a).all())
		self.shoulder_pan_joint.set_motor_torque(0.05 * float(np.clip(a[0], -2, +2)))
		self.shoulder_lift_joint.set_motor_torque(0.05 * float(np.clip(a[1], -2, +2)))
		self.upper_arm_roll_joint.set_motor_torque(0.05 * float(np.clip(a[2], -2, +2)))
		self.elbow_flex_joint.set_motor_torque(0.05 * float(np.clip(a[3], -2, +2)))
		self.forearm_roll_joint.set_motor_torque(0.05 * float(np.clip(a[4], -2, +2)))
		self.wrist_flex_joint.set_motor_torque(0.05 * float(np.clip(a[5], -2, +2)))
		self.wrist_roll_joint.set_motor_torque(0.05 * float(np.clip(a[6], -2, +2)))

	def calc_state(self):
		self.to_target_vec = self.target_pos - self.object_pos
		return np.concatenate([
			np.array([j.current_position() for j in self.ordered_joints]).flatten(),  # all positions
			np.array([j.current_relative_position() for j in self.ordered_joints]).flatten(),  # all speeds
			self.to_target_vec,
			self.fingertip.pose().xyz(),
			self.object.pose().xyz(),
			self.target.pose().xyz(),
		])

	def step(self, a):
		self.apply_action(a)
		self._p.stepSimulation()		

		state = self.calc_state()  # sets self.to_target_vec

		potential_old = self.potential
		self.potential = self.calc_potential()

		joint_vel = np.array([
			self.shoulder_pan_joint.get_velocity(),
			self.shoulder_lift_joint.get_velocity(),
			self.upper_arm_roll_joint.get_velocity(),
			self.elbow_flex_joint.get_velocity(),
			self.upper_arm_roll_joint.get_velocity(),
			self.wrist_flex_joint.get_velocity(),
			self.wrist_roll_joint.get_velocity()
		])

		action_product = np.matmul(np.abs(a), np.abs(joint_vel))
		action_sum = np.sum(a)

		electricity_cost = (
			-0.10 * action_product  # work torque*angular_velocity
			- 0.01 * action_sum  # stall torque require some energy
		)

		stuck_joint_cost = 0
		for j in self.ordered_joints:
			if np.abs(j.current_relative_position()[0]) - 1 < 0.01:
				stuck_joint_cost += -0.1

		self.rewards = [float(self.potential - potential_old), float(electricity_cost), float(stuck_joint_cost)]
		self.HUD(state, a, False)
		return state, sum(self.rewards), False, {}
