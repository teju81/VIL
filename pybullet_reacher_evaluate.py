import gym
import gym.envs.pybullet.gym_manipulators
from PIL import Image, ImageSequence
import argparse
import random
from reacher_task_color_info import gen_reacher_task_color_info, get_reacher_task_color_info
import pickle
import numpy as np
import imageio as io

import tensorflow as tf
from tensorflow.python.platform import flags
from data_generator import DataGenerator
from mil import MIL
from utils import load_scale_and_bias

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('experiment', 'vision_reach', 'sim_vision_reach or sim_push')
flags.DEFINE_string('demo_file', '/home/teju/code/python/data/vision_reach/color_data', 'path to the directory where demo files that containing robot states and actions are stored')
flags.DEFINE_string('demo_gif_dir', '/home/teju/code/python/data/vision_reach/color_data/', 'path to the videos of demonstrations')
flags.DEFINE_string('gif_prefix', 'color', 'prefix of the video directory for each task, e.g. object_0 for task 0')
flags.DEFINE_integer('im_width', 128, 'width of the images in the demo videos 256 for vision_reach')
flags.DEFINE_integer('im_height', 128, 'height of the images in the demo videos 256 for vision_reach')
flags.DEFINE_integer('num_channels', 3, 'number of channels of the images in the demo videos')
flags.DEFINE_integer('T', 24, 'time horizon of the demo videos 24 for reach')
flags.DEFINE_bool('hsv', False, 'convert the image to HSV format')
flags.DEFINE_bool('use_noisy_demos', False, 'use noisy demonstrations or not (for domain shift)')
flags.DEFINE_string('noisy_demo_gif_dir', None, 'path to the videos of noisy demonstrations')
flags.DEFINE_string('noisy_demo_file', None, 'path to the directory where noisy demo files that containing robot states and actions are stored')
flags.DEFINE_bool('no_action', False, 'do not include actions in the demonstrations for inner update')
flags.DEFINE_bool('no_state', False, 'do not include states in the demonstrations during training, use with two headed architecture')
flags.DEFINE_bool('no_final_eept', False, 'do not include final ee pos in the demonstrations for inner update')
flags.DEFINE_bool('zero_state', False, 'zero-out states (meta-learn state) in the demonstrations for inner update (used in the paper with video-only demos)')
flags.DEFINE_bool('two_arms', False, 'use two-arm structure when state is zeroed-out')
flags.DEFINE_integer('training_set_size', -1, 'size of the training set, 1500 for vision_reach and \
                                                -1 for all data except those in validation set')
flags.DEFINE_integer('val_set_size', 50, 'size of the training set, 150 for vision_reach')

## Training options
flags.DEFINE_integer('metatrain_iterations', 50000, 'number of metatraining iterations.') # 50k for reaching
flags.DEFINE_integer('meta_batch_size', 5, 'number of tasks sampled per meta-update') # 5 for reaching
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator') # Beta
flags.DEFINE_integer('update_batch_size', 1, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('train_update_lr', 0.01, 'step size alpha for inner gradient update.') # Alpha: 0.001 for reaching
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.') # 5 for placing
flags.DEFINE_bool('clip', True, 'use gradient clipping for fast gradient')
flags.DEFINE_float('clip_max', 20.0, 'maximum clipping value for fast gradient')
flags.DEFINE_float('clip_min', -20.0, 'minimum clipping value for fast gradient')
flags.DEFINE_bool('fc_bt', True, 'use bias transformation for the first fc layer')
flags.DEFINE_bool('all_fc_bt', False, 'use bias transformation for all fc layers')
flags.DEFINE_bool('conv_bt', False, 'use bias transformation for the first conv layer, N/A for using pretraining')
flags.DEFINE_integer('bt_dim', 10, 'the dimension of bias transformation for FC layers')
flags.DEFINE_string('pretrain_weight_path', '/home/teju/code/python/mil/data/vgg19_weights_tf_dim_ordering_tf_kernels.h5', 'path to pretrained weights or N/A')
flags.DEFINE_bool('train_pretrain_conv1', False, 'whether to finetune the pretrained weights')
flags.DEFINE_bool('two_head', False, 'use two-head architecture')
flags.DEFINE_bool('learn_final_eept', False, 'learn an auxiliary loss for predicting final end-effector pose')
flags.DEFINE_bool('learn_final_eept_whole_traj', False, 'learn an auxiliary loss for predicting final end-effector pose \
                                                         by passing the whole trajectory of eepts (used for video-only models)')
flags.DEFINE_bool('stopgrad_final_eept', True, 'stop the gradient when concatenate the predicted final eept with the feature points')
flags.DEFINE_integer('final_eept_min', 6, 'first index of the final eept in the action array')
flags.DEFINE_integer('final_eept_max', 8, 'last index of the final eept in the action array')
flags.DEFINE_float('final_eept_loss_eps', 0.1, 'the coefficient of the auxiliary loss')
flags.DEFINE_float('act_loss_eps', 1.0, 'the coefficient of the action loss')
flags.DEFINE_float('loss_multiplier', 1.0, 'the constant multiplied with the loss value, 100 for reach and 50 for push')
flags.DEFINE_bool('use_l1_l2_loss', False, 'use a loss with combination of l1 and l2')
flags.DEFINE_float('l2_eps', 0.01, 'coeffcient of l2 loss')
flags.DEFINE_bool('shuffle_val', False, 'whether to choose the validation set via shuffling or not')

## Model options
flags.DEFINE_integer('random_seed', 0, 'random seed for training')
flags.DEFINE_bool('fp', True, 'use spatial soft-argmax or not')
flags.DEFINE_string('norm', 'layer_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_bool('dropout', False, 'use dropout for fc layers or not')
flags.DEFINE_float('keep_prob', 0.5, 'keep probability for dropout')
flags.DEFINE_integer('num_filters', 40, 'number of filters for conv nets -- 64 for placing, 16 for pushing, 40 for reaching.')
flags.DEFINE_integer('filter_size', 3, 'filter size for conv nets -- 3 for placing, 5 for pushing, 3 for reaching.')
flags.DEFINE_integer('num_conv_layers', 3, 'number of conv layers -- 5 for placing, 4 for pushing, 3 for reaching.')
flags.DEFINE_integer('num_strides', 3, 'number of conv layers with strided filters -- 3 for placing, 4 for pushing, 3 for reaching.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_integer('num_fc_layers', 4, 'number of fully-connected layers')
flags.DEFINE_integer('layer_size', 200, 'hidden dimension of fully-connected layers')
flags.DEFINE_bool('temporal_conv_2_head', False, 'whether or not to use temporal convolutions for the two-head architecture in video-only setting.')
flags.DEFINE_bool('temporal_conv_2_head_ee', False, 'whether or not to use temporal convolutions for the two-head architecture in video-only setting \
                for predicting the ee pose.')
flags.DEFINE_integer('temporal_filter_size', 5, 'filter size for temporal convolution')
flags.DEFINE_integer('temporal_num_filters', 64, 'number of filters for temporal convolution')
flags.DEFINE_integer('temporal_num_filters_ee', 64, 'number of filters for temporal convolution for ee pose prediction')
flags.DEFINE_integer('temporal_num_layers', 3, 'number of layers for temporal convolution for ee pose prediction')
flags.DEFINE_integer('temporal_num_layers_ee', 3, 'number of layers for temporal convolution for ee pose prediction')
flags.DEFINE_string('init', 'xavier', 'initializer for conv weights. Choose among random, xavier, and he')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_integer('print_interval', 50, 'interval between 2 prints')
flags.DEFINE_integer('test_print_interval', 100, 'interval between 2 test prints')
flags.DEFINE_integer('summary_interval', 50, 'interval between 2 summaries')
flags.DEFINE_integer('save_interval', 500, 'interval between 2 saves')

flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('log_dir', '/home/teju/code/python/mil/logs/visionreach', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', False, 'True to train, False to test.')
flags.DEFINE_integer('restore_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training \
                    (use if you want to test with a different number).')
flags.DEFINE_integer('test_update_batch_size', 1, 'number of demos used during test time')
flags.DEFINE_float('gpu_memory_fraction', 1.0, 'fraction of memory used in gpu')
flags.DEFINE_bool('record_gifs', True, 'record gifs during evaluation')


parser = argparse.ArgumentParser()
parser.add_argument("--gif_file_name",help='file name of the gif')
parser.add_argument("--task_color_index",help='task color index', type=int)
parser.add_argument("--object_type",help='object type')
parser.add_argument("--object_scale",help='object scale', type=int)
parser.add_argument("--image_height",help='image height', type=int)
parser.add_argument("--image_width",help='image width', type=int)
parser.add_argument("--num_frames",help='number of frames', type=int)
args = parser.parse_args()

env = gym.make('PyBulletReacher-v0')

gif_name = 'sample.gif'
env.env.object_type = "cross"
env.env.object_scale = 40
env.env.color_list = get_reacher_task_color_info(task_color_index=1700)
env.env.gif_height = 128
env.env.gif_width = 128
env.env.T = 24
# Disable Inverse Kinematics: Uses external actions passed through env.step() to help reach target
env.env.training_phase = False

# Read Video, State Sequence and Action Sequence from training demo
img_seq = []
reader = io.get_reader(gif_name)
for im in reader:
    img_seq.append(np.array(im[:,:,:-1]))
video = np.array(img_seq)

with open(gif_name[:-3] + 'pkl', 'rb') as f:
    demo = pickle.load(f)
f.close()
state_seq = demo['demoX'][:,0:2]
action_seq = demo['demoU'][:,0:2]
# This forms the Training Demo: Prepare to feed tensorflow graph
T, W, H, C = video.shape
img_dim = W*H*C

training_video = video.reshape(1,T,img_dim)
training_state_seq = state_seq.reshape(1,state_seq.shape[0],state_seq.shape[1])
training_action_seq = action_seq.reshape(1,action_seq.shape[0],action_seq.shape[1])

# Setup tensorflow session and Initialize graph
graph = tf.Graph()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
tf_config = tf.ConfigProto(gpu_options=gpu_options)
sess = tf.Session(graph=graph, config=tf_config)
network_config = {
    'num_filters': [FLAGS.num_filters]*FLAGS.num_conv_layers,
    'strides': [[1, 2, 2, 1]]*FLAGS.num_strides + [[1, 1, 1, 1]]*(FLAGS.num_conv_layers-FLAGS.num_strides),
    'filter_size': FLAGS.filter_size,
    'image_width': FLAGS.im_width,
    'image_height': FLAGS.im_height,
    'image_channels': FLAGS.num_channels,
    'n_layers': FLAGS.num_fc_layers,
    'layer_size': FLAGS.layer_size,
    'initialization': FLAGS.init,
    'temporal_conv_2_head_ee': FLAGS.temporal_conv_2_head_ee,
}
data_generator = DataGenerator()
state_idx = data_generator.state_idx
img_idx = range(len(state_idx), len(state_idx)+FLAGS.im_height*FLAGS.im_width*FLAGS.num_channels)
model = MIL(data_generator._dU, state_idx=state_idx, img_idx=img_idx, network_config=network_config)
model.init_network(graph, prefix='Testing')


exp_string = FLAGS.experiment + '.' + FLAGS.init + '_init.' + str(FLAGS.num_conv_layers) + '_conv' + '.' + str(FLAGS.num_strides) + '_strides' + '.' + str(FLAGS.num_filters) + '_filters' + \
            '.' + str(FLAGS.num_fc_layers) + '_fc' + '.' + str(FLAGS.layer_size) + '_dim' + '.bt_dim_' + str(FLAGS.bt_dim) + '.mbs_'+str(FLAGS.meta_batch_size) + \
            '.ubs_' + str(FLAGS.update_batch_size) + '.numstep_' + str(FLAGS.num_updates) + '.updatelr_' + str(FLAGS.train_update_lr)

if FLAGS.clip:
    exp_string += '.clip_' + str(int(FLAGS.clip_max))
if FLAGS.conv_bt:
    exp_string += '.conv_bt'
if FLAGS.all_fc_bt:
    exp_string += '.all_fc_bt'
if FLAGS.fp:
    exp_string += '.fp'
if FLAGS.learn_final_eept:
    exp_string += '.learn_ee_pos'
if FLAGS.no_action:
    exp_string += '.no_action'
if FLAGS.zero_state:
    exp_string += '.zero_state'
if FLAGS.two_head:
    exp_string += '.two_heads'
if FLAGS.two_arms:
    exp_string += '.two_arms'
if FLAGS.temporal_conv_2_head:
    exp_string += '.1d_conv_act_' + str(FLAGS.temporal_num_layers) + '_' + str(FLAGS.temporal_num_filters)
    if FLAGS.temporal_conv_2_head_ee:
        exp_string += '_ee_' + str(FLAGS.temporal_num_layers_ee) + '_' + str(FLAGS.temporal_num_filters_ee)
    exp_string += '_' + str(FLAGS.temporal_filter_size) + 'x1_filters'
if FLAGS.training_set_size != -1:
    exp_string += '.' + str(FLAGS.training_set_size) + '_trials'
exp_string = 'mean_loss'
log_dir = FLAGS.log_dir + '/' + exp_string
print(log_dir)
print(exp_string)
#assert exp_string == 'vision_reach.xavier_init.3_conv.3_strides.40_filters.4_fc.200_dim.bt_dim_10.mbs_5.ubs_1.numstep_1.updatelr_0.01.clip_20.fp'
assert exp_string == 'mean_loss'
#Load trained model
with graph.as_default():
    # Set up saver.
    saver = tf.train.Saver(max_to_keep=2)
    # Initialize variables.
    init_op = tf.global_variables_initializer()
    sess.run(init_op, feed_dict=None)
print(log_dir)
model_file = tf.train.latest_checkpoint(log_dir) #Load Latest model
print("Restoring model weights from " + model_file)
with graph.as_default():
    saver.restore(sess, model_file)
print('done')
scale, bias = load_scale_and_bias('/home/teju/code/python/mil/data/scale_and_bias_vision_reach.pkl')
env.env.reset_env()
mode = "rgb_array"
observation = env.reset()


joint_angles = observation[0:2]
joint_velocities = observation[2:4]
joint_torques = observation[4:6]
ee_pose = observation[6:]
state = np.squeeze(joint_angles)
state = state.reshape(1, 1, -1)
img_seq = []
rgb_img = env.render()
print(state)
for t in range(env.env.T*10):
    print("Executing Time step: {}".format(t+1))
    img_seq.append(rgb_img)
    rgb_img = rgb_img.reshape(1, 1, -1)
    state = state.reshape(1, 1, -1)
    # Feed dictionary will contain training demo and current state of the test
    feed_dict = {
        model.obsa: training_video,
        model.statea: training_state_seq.dot(scale) + bias,
        model.actiona: training_action_seq,
        model.obsb: rgb_img,
        model.stateb: state.dot(scale) + bias
    }
    with graph.as_default():
        action = sess.run(model.test_act_op, feed_dict=feed_dict)
    action = np.squeeze(action)
    print('Model actions: ' + str(action))
    jd=[0.001,0.001]
    jp = env.env._p.calculateInverseKinematics(env.env.model[1], 4, env.env.target_position, jointDamping=jd, maxNumIterations = 1000)
    print("IK jp: " + str(jp))
    observation, reward, done, info = env.step(action)
    
    joint_angles = observation[0:2]
    joint_velocities = observation[2:4]
    joint_torques = observation[4:6]
    ee_pose = observation[6:]
    state = np.squeeze(joint_angles) # Not including torques
    rgb_img = env.render()
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
print("done")


video = np.array(img_seq)
print('Saving gif sample to test.gif')
io.mimwrite('test.gif', video)

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


# Testing on trained tasak
# filenames = []
# observation = env.env.reset()
# for t in range(24):
#     feed_dict = {
#     model.obsa: selected_demoO,
#     model.statea: selected_demoX.dot(scale) + bias,
#     model.actiona: selected_demoU,
#     model.obsb: obs,
#     model.stateb: state.dot(scale) + bias}
#     with graph.as_default():
#         action = sess.run(model.test_act_op, feed_dict=feed_dict)
#     action = np.concatenate((p[t+1], v[t+1], a[t+1]))
#     observation, reward, done, info = env.step(action)
#     rgb_img = env.render()
#     if done:
#         print("Episode finished after {} timesteps".format(t+1))
#         break

# video = np.array(img_seq)
# print('Saving gif sample to :%s' % gif_name)
# io.mimwrite(gif_name, video)
# print("done")
# env.close()
