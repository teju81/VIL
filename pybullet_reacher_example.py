import gym
import gym.envs.pybullet.gym_manipulators
from PIL import Image, ImageSequence
import argparse
import random
from reacher_task_color_info import gen_reacher_task_color_info, get_reacher_task_color_info
import pickle
import numpy as np
import imageio as io

parser = argparse.ArgumentParser()
parser.add_argument("--gif_file_name",help='file name of the gif')
parser.add_argument("--task_color_index",help='task color index', type=int)
parser.add_argument("--object_type",help='object type')
parser.add_argument("--object_scale",help='object scale', type=int)
parser.add_argument("--image_height",help='image height', type=int)
parser.add_argument("--image_width",help='image width', type=int)
parser.add_argument("--num_frames",help='number of frames', type=int)
args = parser.parse_args()

# Cartesian product is the color pallete
color_pallete = [(c1,c2,c3) for c1 in range(0,255,20) for c2 in range(0,255,20) for c3 in range(0,255,20)]

command_line_args = False
env = gym.make('PyBulletReacher-v0')
if command_line_args:
	gif_name =  args.gif_file_name
	env.env.object_type = args.object_type
	env.env.object_scale = args.object_scale
	env.env.color_list = get_reacher_task_color_info(task_color_index=args.task_color_index)
	env.env.gif_height = args.image_height
	env.env.gif_width = args.image_width
	env.env.T = args.num_frames
else:
	gif_name = 'sample.gif'
	env.env.object_type = "cross"
	env.env.object_scale = 40
	env.env.color_list = get_reacher_task_color_info(task_color_index=1700)
	env.env.gif_height = 128
	env.env.gif_width = 128
	env.env.T = 24
pos_seq = []
vel_seq = []
action_seq = []
eep_seq = []
initial_state = []
img_seq = []
filenames = []
state = []

env.env.reset_env()
mode = "rgb_array"
i_episode = 0
print("Episode #" + str(i_episode))
observation = env.reset()

joint_angles = observation[0:2]
joint_velocities = observation[2:4]
joint_torques = observation[4:6]
ee_pose = observation[6:]
env.env.training_phase = True # Use Inverse Kinematics to generate actions in order to achieve reaching for the target: Ignores any external actions passed through env.step()
state_seq = np.empty((0,7), dtype=np.float32)
action_seq = np.empty((0,7), dtype=np.float32)
for t in range(env.env.T):
    action = env.action_space.sample()#Ignore the action passed. IK solver generates actions internally
    current_state = np.squeeze(np.concatenate((joint_angles, joint_velocities, ee_pose))) # Not including torques
    state_seq = np.vstack((state_seq, current_state))
    observation, reward, done, info = env.step(action)
    joint_angles = observation[0:2]
    joint_velocities = observation[2:4]
    joint_torques = observation[4:6]
    ee_pose = observation[6:]
    current_action = np.squeeze(np.concatenate((joint_angles, joint_velocities, ee_pose))) # Not including torques
    action_seq = np.vstack((action_seq, current_action))
    rgb_img = env.render()
    img_seq.append(rgb_img)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
print("done")

video = np.array(img_seq)
print('Saving gif sample to :%s' % gif_name)
io.mimwrite(gif_name, video)

# # Debug: Check how many frames are there in the recorded gif
# img = Image.open(gif_name)
# print(img.is_animated)
# print(img.n_frames)
# frames = np.array([np.array(frame.copy().convert('RGB').getdata(),dtype=np.uint8).reshape(frame.size[1],frame.size[0],3) for frame in ImageSequence.Iterator(img)])
# print(frames.shape)


demo={'demoX': state_seq, 'demoU':action_seq}
with open(gif_name[:-3] + 'pkl', 'wb') as f:
	pickle.dump(demo, f, protocol=2)
f.close()

# with open(gif_name[:-3] + 'pkl', 'rb') as f:
# 	demo_dash = pickle.load(f)
# f.close()
# print(demo_dash)

# print("Replay")
# env.env.training_phase = False
# observation = env.env.reset(initial_state=initial_state)
# for t in range(24):
#     action = np.concatenate((p[t+1], v[t+1], a[t+1]))
#     observation, reward, done, info = env.step(action)
#     rgb_img = env.render()
#     if done:
#         print("Episode finished after {} timesteps".format(t+1))
#         break
# print("done")
# env.close()