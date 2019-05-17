""" This file defines Meta Imitation Learning for Target Localization (MTL). """
from __future__ import division

import numpy as np
import random
import tensorflow as tf
from tensorflow.python.platform import flags
from tf_utils import *
from utils import Timer
from natsort import natsorted

FLAGS = flags.FLAGS

class MTL(object):
    """ Initialize MTL. Need to call init_network to contruct the architecture after init. """
    def __init__(self, dU, dT, state_idx=None, img_idx=None, network_config=None):
        # MTL hyperparams
        self.num_updates = FLAGS.num_updates
        self.update_batch_size = FLAGS.update_batch_size
        self.meta_batch_size = FLAGS.meta_batch_size
        self.meta_lr = FLAGS.meta_lr
        self.activation_fn = tf.nn.relu # by default, we use relu
        self.T = FLAGS.T
        self.network_params = network_config
        self.norm_type = FLAGS.norm
        # List of indices for state (vector) data and image (tensor) data in observation.
        self.state_idx, self.img_idx = state_idx, img_idx
        # Dimension of input and output of the model
        self._dO = len(img_idx) + len(state_idx)
        self._dU = dU
        self._dT = dT

    def init_network(self, graph, input_tensors=None, restore_iter=0, prefix='Training_'):
        """ Helper method to initialize the tf networks used """
        with graph.as_default():
            with Timer('building TF network'):
                result = self.construct_model(input_tensors=input_tensors, prefix=prefix, dim_input=self._dO, dim_output=self._dT,
                                          network_config=self.network_params)
            inputas, inputbs, outputas, outputbs, smaxas, smaxbs, test_output, lossesa, lossesb, flat_img_inputb, gradients_op = result
            if 'Testing' in prefix:
                self.obs_tensor = self.obsa
                self.state_tensor = self.statea
                self.test_act_op = test_output
                self.image_op = flat_img_inputb

            trainable_vars = tf.trainable_variables()
            total_losses1 = [tf.reduce_sum(lossesa[j]) / tf.to_float(self.meta_batch_size) for j in range(self.num_updates)]
            total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(self.meta_batch_size) for j in range(self.num_updates)]

            if 'Training' in prefix:
                self.total_losses1 = total_losses1
                self.total_loss1 = total_losses1[0]
                self.total_losses2 = total_losses2
                self.outputas = outputas
                self.outputbs = outputbs
                self.smaxas = smaxas
                self.smaxbs = smaxbs
            elif 'Validation' in prefix:
                self.val_total_losses1 = total_losses1
                self.val_total_loss1 = total_losses1[0]
                self.val_total_losses2 = total_losses2
                self.val_outputas = outputas
                self.val_outputbs = outputbs
                self.val_smaxas = smaxas
                self.val_smaxbs = smaxbs

            if 'Training' in prefix:
                decay_steps = 50000
                lr_decayed = tf.train.cosine_decay(self.meta_lr, self.global_step, decay_steps)
                #lr_decayed = tf.train.exponential_decay(self.meta_lr, self.global_step, 1000, 0.96, staircase=True)
                #lr_decayed = self.meta_lr
                self.train_op = tf.train.AdamOptimizer(lr_decayed).minimize(self.total_losses2[self.num_updates - 1],global_step=self.global_step)

                # Add summaries
                summ = [tf.summary.scalar(prefix + 'Pre-update_loss', self.total_loss1)]
                for k,v in self.weights.items():
                  summ.append(tf.summary.histogram('Weights_of_%s' % (k), v))
                for j in range(self.num_updates):
                    for task_id in range(smaxas[j].shape[0]):
                        imga = inputas[j][task_id,23:,:,:,:]
                        imgb = inputbs[j][task_id,23:,:,:,:]
                        summ.append(tf.summary.image('Task_%d_IMG_A_Step_%d' % (task_id,j),imga, 1))
                        summ.append(tf.summary.image('Task_%d_IMG_B_Step_%d' % (task_id,j),imgb, 1))
                        for filt_id in range(smaxas[j].shape[-1]):
                          filta = smaxas[j][task_id,:,:,:,filt_id:filt_id+1]
                          filtb = smaxbs[j][task_id,:,:,:,filt_id:filt_id+1]
                          summ.append(tf.summary.image('Task_%d_Spatial_Softmax_A_%d_Step_%d' % (task_id,filt_id,j),filta, 1))
                          summ.append(tf.summary.image('Task_%d_Spatial_Softmax_B_%d_Step_%d' % (task_id,filt_id,j),filtb, 1))

                    summ.append(tf.summary.scalar(prefix + 'Post-update_loss_step_%d' % j, self.total_losses2[j]))
                    for k in range(len(self.sorted_weight_keys)):
                        summ.append(tf.summary.histogram('Gradient_of_%s_step_%d' % (self.sorted_weight_keys[k], j), gradients_op[j][k]))

                self.train_summ_op = tf.summary.merge(summ)
            elif 'Validation' in prefix:
                # Add summaries
                summ = [tf.summary.scalar(prefix + 'Pre-update_loss', self.val_total_loss1)]
                for j in range(self.num_updates):
                    for task_id in range(smaxas[j].shape[0]):
                      imga = inputas[j][task_id,:,:,:]
                      imgb = inputbs[j][task_id,:,:,:]
                      summ.append(tf.summary.image('Val_Task_%d_IMG_A_Step_%d' % (task_id,j),imga, 1))
                      summ.append(tf.summary.image('Val_Task_%d_IMG_B_Step_%d' % (task_id,j),imgb, 1))
                      for filt_id in range(smaxas[j].shape[-1]):
                        filta = smaxas[j][task_id,:,:,:,filt_id:filt_id+1]
                        filtb = smaxbs[j][task_id,:,:,:,filt_id:filt_id+1]
                        summ.append(tf.summary.image('Val_Task_%d_Spatial_Softmax_A_%d_Step_%d' % (task_id,filt_id,j),filta, 1))
                        summ.append(tf.summary.image('Val_Task_%d_Spatial_Softmax_B_%d_Step_%d' % (task_id,filt_id,j),filtb, 1))

                    summ.append(tf.summary.scalar(prefix + 'Post-update_loss_step_%d' % j, self.val_total_losses2[j]))
                self.val_summ_op = tf.summary.merge(summ)

    def construct_image_input(self, nn_input, state_idx, img_idx, network_config=None):
        """ Preprocess images. """
        state_input = nn_input[:, 0:state_idx[-1]+1]
        flat_image_input = nn_input[:, state_idx[-1]+1:img_idx[-1]+1]

        # image goes through 3 convnet layers
        num_filters = network_config['num_filters']

        im_height = network_config['image_height']
        im_width = network_config['image_width']
        num_channels = network_config['image_channels']
        image_input = tf.reshape(flat_image_input, [-1, im_width, im_height, num_channels]) # For mujoco swap 1 and 3 dimensions in reshape
        #image_input = tf.transpose(image_input, perm=[0,3,2,1]) # Mujoco
        if FLAGS.pretrain_weight_path != 'N/A':
            image_input = image_input * 255.0 - tf.convert_to_tensor(np.array([103.939, 116.779, 123.68], np.float32))
            # 'RGB'->'BGR'
            image_input = image_input[:, :, :, ::-1]
        return image_input, flat_image_input, state_input

    def construct_weights(self, dim_input, dim_output, network_config=None):
        """ Construct weights for the network. """
        weights = {}
        num_filters = network_config['num_filters']
        strides = network_config.get('strides', [[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1]])
        filter_sizes = network_config.get('filter_size', [3]*len(strides)) # used to be 2
        if type(filter_sizes) is not list:
            filter_sizes = len(strides)*[filter_sizes]
        im_height = network_config['image_height']
        im_width = network_config['image_width']
        num_channels = network_config['image_channels']
        is_dilated = network_config.get('is_dilated', False)
        use_fp = FLAGS.fp
        pretrain = FLAGS.pretrain_weight_path != 'N/A'
        train_pretrain_conv1 = FLAGS.train_pretrain_conv1
        initialization = network_config.get('initialization', 'random')
        if pretrain:
            num_filters[0] = 64
        pretrain_weight_path = FLAGS.pretrain_weight_path
        n_conv_layers = len(num_filters)
        downsample_factor = 1
        for stride in strides:
            downsample_factor *= stride[1]
        if use_fp:
            self.conv_out_size = int(num_filters[-1]*2)
        else:
            self.conv_out_size = int(np.ceil(im_width/(downsample_factor)))*int(np.ceil(im_height/(downsample_factor)))*num_filters[-1]

        # conv weights
        fan_in = num_channels
        if FLAGS.conv_bt:
            fan_in += num_channels
        if FLAGS.conv_bt:
            weights['img_context'] = safe_get('img_context', initializer=tf.zeros([im_height, im_width, num_channels], dtype=tf.float32))
            weights['img_context'] = tf.clip_by_value(weights['img_context'], 0., 1.)
        for i in range(n_conv_layers):
            if not pretrain or i != 0:
                if self.norm_type == 'selu':
                    weights['wc%d' % (i+1)] = init_conv_weights_snn([filter_sizes[i], filter_sizes[i], fan_in, num_filters[i]], name='wc%d' % (i+1)) # 5x5 conv, 1 input, 32 outputs
                elif initialization == 'xavier':
                    weights['wc%d' % (i+1)] = init_conv_weights_xavier([filter_sizes[i], filter_sizes[i], fan_in, num_filters[i]], name='wc%d' % (i+1)) # 5x5 conv, 1 input, 32 outputs
                elif initialization == 'random':
                    weights['wc%d' % (i+1)] = init_weights([filter_sizes[i], filter_sizes[i], fan_in, num_filters[i]], name='wc%d' % (i+1)) # 5x5 conv, 1 input, 32 outputs
                else:
                    raise NotImplementedError
                weights['bc%d' % (i+1)] = init_bias([num_filters[i]], name='bc%d' % (i+1))
                fan_in = num_filters[i]
            else:
                import h5py

                assert num_filters[i] == 64
                vgg_filter_size = 3
                weights['wc%d' % (i+1)] = safe_get('wc%d' % (i+1), [vgg_filter_size, vgg_filter_size, fan_in, num_filters[i]], dtype=tf.float32, trainable=train_pretrain_conv1)
                weights['bc%d' % (i+1)] = safe_get('bc%d' % (i+1), [num_filters[i]], dtype=tf.float32, trainable=train_pretrain_conv1)
                pretrain_weight = h5py.File(pretrain_weight_path, 'r')
                conv_weight = pretrain_weight['block1_conv%d' % (i+1)]['block1_conv%d_W_1:0' % (i+1)][...]
                conv_bias = pretrain_weight['block1_conv%d' % (i+1)]['block1_conv%d_b_1:0' % (i+1)][...]
                weights['wc%d' % (i+1)].assign(conv_weight)
                weights['bc%d' % (i+1)].assign(conv_bias)
                fan_in = conv_weight.shape[-1]

        # fc weights
        in_shape = self.conv_out_size
        if not FLAGS.no_state:
            in_shape += len(self.state_idx)
        if FLAGS.fc_bt:
            in_shape += FLAGS.bt_dim
        if FLAGS.fc_bt:
            weights['context'] = safe_get('context', initializer=tf.zeros([FLAGS.bt_dim], dtype=tf.float32))
        self.conv_out_size_final = in_shape

        if self.norm_type == 'selu':
            weights['w_fc'] = init_fc_weights_snn([in_shape, dim_output], name='w_fc')
        else:
            weights['w_fc'] = init_weights([in_shape, dim_output], name='w_fc')
        weights['b_fc'] = init_bias([dim_output], name='b_fc')

        return weights
      
    def construct_lr_weights(self, weight_keys, inner_lr, num_updates):
        lr_weight_dict = {}
        for key in weight_keys:
          name = 'ld_' + key
          lr_weight_dict[name] = safe_get(name, initializer=tf.ones([1], dtype=tf.float32))
          for j in range(num_updates):
            name = 'lr_' + key + 'update%d'%j
            lr_weight_dict[name] = safe_get(name, initializer=inner_lr*tf.ones([1], dtype=tf.float32))
          
        return lr_weight_dict

    def forward(self, image_input, state_input, weights, meta_testing=False, is_training=True, testing=False, network_config=None):
        """ Perform the forward pass. """
        if FLAGS.fc_bt:
            im_height = network_config['image_height']
            im_width = network_config['image_width']
            num_channels = network_config['image_channels']
            flatten_image = tf.reshape(image_input, [-1, im_height*im_width*num_channels])
            context = tf.transpose(tf.gather(tf.transpose(tf.zeros_like(flatten_image)), list(range(FLAGS.bt_dim))))
            context += weights['context']
        norm_type = self.norm_type
        decay = network_config.get('decay', 0.9)
        strides = network_config.get('strides', [[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1]])
        downsample_factor = strides[0][1]
        n_strides = len(strides)
        n_conv_layers = len(strides)
        use_dropout = FLAGS.dropout
        prob = FLAGS.keep_prob
        is_dilated = network_config.get('is_dilated', False)
        im_height = network_config['image_height']
        im_width = network_config['image_width']
        num_channels = network_config['image_channels']
        conv_layer = image_input
        if FLAGS.conv_bt:
            img_context = tf.zeros_like(conv_layer)
            img_context += weights['img_context']
            conv_layer = tf.concat(axis=3, values=[conv_layer, img_context])
        for i in range(n_conv_layers):
            if not use_dropout:
                conv_layer = norm(conv2d(img=conv_layer, w=weights['wc%d' % (i+1)], b=weights['bc%d' % (i+1)], strides=strides[i], is_dilated=is_dilated), \
                                norm_type=norm_type, decay=decay, id=i, is_training=is_training, activation_fn=self.activation_fn)
            else:
                conv_layer = dropout(norm(conv2d(img=conv_layer, w=weights['wc%d' % (i+1)], b=weights['bc%d' % (i+1)], strides=strides[i], is_dilated=is_dilated), \
                                norm_type=norm_type, decay=decay, id=i, is_training=is_training, activation_fn=self.activation_fn), keep_prob=prob, is_training=is_training, name='dropout_%d' % (i+1))

        #self.sp_smax_in = conv_layer3
        if FLAGS.fp:
            _, num_rows, num_cols, num_fp = conv_layer.get_shape()
            if is_dilated:
                num_rows = int(np.ceil(im_width/(downsample_factor**n_strides)))
                num_cols = int(np.ceil(im_height/(downsample_factor**n_strides)))
            num_rows, num_cols, num_fp = [int(x) for x in [num_rows, num_cols, num_fp]]
            x_map = np.empty([num_rows, num_cols], np.float32)
            y_map = np.empty([num_rows, num_cols], np.float32)

            # Teju Comment: 
            # Spatial softmax = Expected pixel position per channel using softmax of activation of each channel as prob distribution

            # Teju comment: pixel position is transformed to vary between (-0.5, 0.5)
            for i in range(num_rows):
                for j in range(num_cols):
                    x_map[i, j] = (i - num_rows / 2.0) / num_rows
                    y_map[i, j] = (j - num_cols / 2.0) / num_cols

            x_map = tf.convert_to_tensor(x_map)
            y_map = tf.convert_to_tensor(y_map)

            x_map = tf.reshape(x_map, [num_rows * num_cols])
            y_map = tf.reshape(y_map, [num_rows * num_cols])

            # rearrange features to be [batch_size, num_fp, num_rows, num_cols]
            features = tf.reshape(tf.transpose(conv_layer, [0,3,1,2]),
                                  [-1, num_rows*num_cols])
            softmax = tf.nn.softmax(features)

            fp_x = tf.reduce_sum(tf.multiply(x_map, softmax), [1], keep_dims=True)
            fp_y = tf.reduce_sum(tf.multiply(y_map, softmax), [1], keep_dims=True)

            conv_out_flat = tf.reshape(tf.concat(axis=1, values=[fp_x, fp_y]), [-1, num_fp*2])
        else:
            conv_out_flat = tf.reshape(conv_layer, [-1, self.conv_out_size])
        fc_input = conv_out_flat #tf.add(conv_out_flat, 0)

        smax = tf.nn.softmax(conv_layer) # Return activations for debugging

        if FLAGS.fc_bt:
            fc_input = tf.concat(axis=1, values=[fc_input, context])
        #Fully connected layers
        fc_output = fc_input #tf.add(fc_input, 0)
        if state_input is not None:
            fc_output = tf.concat(axis=1, values=[fc_output, state_input])
        fc_output = tf.matmul(fc_output, weights['w_fc']) + weights['b_fc']

        #Sigmoid to keep output between 0 - 1
        fc_output = tf.nn.sigmoid(fc_output)

        return smax, fc_output

    def construct_model(self, input_tensors=None, prefix='Training_', dim_input=27, dim_output=2, network_config=None):
        """
        Construct the meta-learning graph.
        Args:
            input_tensors: tensors of input videos
            prefix: indicate whether we are building training, validation or testing graph.
            dim_input: Dimensionality of input.
            dim_output: Dimensionality of the output.
            network_config: dictionary of network structure parameters
        Returns:
            a tuple of output tensors.
        """
        if input_tensors is None:
            self.obsa = obsa = tf.placeholder(tf.float32, name='obsa') # meta_batch_size x update_batch_size x dim_input
            self.obsb = obsb = tf.placeholder(tf.float32, name='obsb')
        else:
            self.obsa = obsa = input_tensors['inputa'] # meta_batch_size x update_batch_size x dim_input
            self.obsb = obsb = input_tensors['inputb']

        if not hasattr(self, 'statea'):
            self.statea = statea = tf.placeholder(tf.float32, name='statea')
            self.stateb = stateb = tf.placeholder(tf.float32, name='stateb')
            self.actiona = actiona = tf.placeholder(tf.float32, name='actiona')
            self.actionb = actionb = tf.placeholder(tf.float32, name='actionb')
            self.targeta = targeta = tf.placeholder(tf.float32, name='targeta')
            self.targetb = targetb = tf.placeholder(tf.float32, name='targetb')
        else:
            statea = self.statea
            stateb = self.stateb
            actiona = self.actiona
            actionb = self.actionb
            targeta = self.targeta
            targetb = self.targetb

        inputa = tf.concat(axis=2, values=[statea, obsa])
        inputb = tf.concat(axis=2, values=[stateb, obsb])

        with tf.variable_scope('model', reuse=None) as training_scope:
            self.global_step = tf.Variable(0,trainable=False,name='global_step')
            # Construct layers weight & bias
            if 'weights' not in dir(self):
                self.weights = weights = self.construct_weights(dim_input, dim_output, network_config=network_config)
                self.sorted_weight_keys = natsorted(self.weights.keys())
                #Learn the inner learning rates
                self.lr_weights = lr_weights = self.construct_lr_weights(self.sorted_weight_keys, FLAGS.train_update_lr, FLAGS.num_updates)
            else:
                training_scope.reuse_variables()
                weights = self.weights
                lr_weights = self.lr_weights
                print('Reusing weight names defined in the graph')

            self.step_size = FLAGS.train_update_lr
            loss_multiplier = FLAGS.loss_multiplier

            num_updates = self.num_updates
            lossesa = [[] for _ in range(num_updates)]
            outputsa = [[] for _ in range(num_updates)]
            lossesb = [[] for _ in range(num_updates)]
            outputsb = [[] for _ in range(num_updates)]

            def batch_metalearn(inp):
                inputa, inputb, targeta, targetb = inp
                inputa = tf.reshape(inputa, [-1, dim_input])
                inputb = tf.reshape(inputb, [-1, dim_input])
                targeta = tf.reshape(targeta, [-1, dim_output])
                targetb = tf.reshape(targetb, [-1, dim_output])
                gradients_summ = []
                testing = 'Testing' in prefix

                local_outputas, local_lossesa, local_outputbs, local_lossesb = [], [], [], []
                smaxas, smaxbs = [], []
                # Assume fixed data for each update
                targetas = [targeta]*num_updates
                targetbs = [targetb]*num_updates

                # Convert to image dims
                inputa, _, state_inputa = self.construct_image_input(inputa, self.state_idx, self.img_idx, network_config=network_config)
                inputb, flat_img_inputb, state_inputb = self.construct_image_input(inputb, self.state_idx, self.img_idx, network_config=network_config)
                inputas = [inputa]*num_updates
                inputbs = [inputb]*num_updates
                if FLAGS.zero_state:
                    state_inputa = tf.zeros_like(state_inputa)
                state_inputas = [state_inputa]*num_updates
                if FLAGS.no_state:
                    state_inputa = None

                # Pre-update
                if 'Training' in prefix:
                    smaxa, local_outputa = self.forward(inputa, state_inputa, weights, network_config=network_config)
                else:
                    smaxa, local_outputa = self.forward(inputa, state_inputa, weights, is_training=False, network_config=network_config)
                local_outputas.append(local_outputa)
                smaxas.append(smaxa)

                local_lossa = euclidean_loss_layer(local_outputa, targeta, multiplier=loss_multiplier, use_l1=FLAGS.use_l1_l2_loss)
                local_lossesa.append(local_lossa)

                # Compute fast gradients
                #gradients = {}
                #for k,v in weights.items():
                    #gradients[k] = tf.gradients(local_lossa, v)

                # Compute fast gradients
                grads = tf.gradients(local_lossa, list(weights.values()))
                gradients = dict(zip(weights.keys(), grads))
                
                # make fast gradient zero for weights with gradient None
                for key in gradients.keys():
                    if gradients[key] is None:
                        gradients[key] = tf.zeros_like(weights[key])
                if FLAGS.stop_grad:
                    gradients = {key:tf.stop_gradient(gradients[key]) for key in gradients.keys()}
                if FLAGS.clip:
                    clip_min = FLAGS.clip_min
                    clip_max = FLAGS.clip_max
                    for key in gradients.keys():
                        gradients[key] = tf.clip_by_value(gradients[key], clip_min, clip_max)
                if FLAGS.pretrain_weight_path != 'N/A':
                    gradients['wc1'] = tf.zeros_like(gradients['wc1'])
                    gradients['bc1'] = tf.zeros_like(gradients['bc1'])
                gradients_summ.append([gradients[key] for key in self.sorted_weight_keys])
                
                fast_weights = {}
                for key in weights.keys():
                  self.step_size = lr_weights['lr_' + key + 'update0']
                  #self.update_dir = tf.sign(lr_weights['ld_' + key])
                  fast_weights[key] = weights[key] - self.step_size*gradients[key]

                # Post-update
                if FLAGS.no_state:
                    state_inputb = None
                if 'Training' in prefix:
                    smaxb, outputb = self.forward(inputb, state_inputb, fast_weights, meta_testing=True, network_config=network_config)
                else:
                    smaxb, outputb = self.forward(inputb, state_inputb, fast_weights, meta_testing=True, is_training=False, testing=testing, network_config=network_config)
                local_outputbs.append(outputb)
                smaxbs.append(smaxb)
                local_lossb = euclidean_loss_layer(outputb, targetb, multiplier=loss_multiplier, use_l1=FLAGS.use_l1_l2_loss)
                local_lossesb.append(local_lossb)

                for j in range(num_updates - 1):
                    # Pre-update
                    state_inputa_new = state_inputas[j+1]
                    if FLAGS.no_state:
                        state_inputa_new = None
                    if 'Training' in prefix:
                        smaxa, outputa = self.forward(inputas[j+1], state_inputa_new, fast_weights, network_config=network_config)
                    else:
                        smaxa, outputa = self.forward(inputas[j+1], state_inputa_new, fast_weights, is_training=False, testing=testing, network_config=network_config)
                    local_outputas.append(outputa)
                    smaxas.append(smaxa)
                    loss = euclidean_loss_layer(outputa, targetas[j+1], multiplier=loss_multiplier, use_l1=FLAGS.use_l1_l2_loss)
                    local_lossesa.append(loss)

                    # Compute fast gradients
                    #gradients = {}
                    #for k,v in fast_weights.items():
                        #gradients[k] = tf.gradients(local_lossa, v)

                    # Compute fast gradients
                    grads = tf.gradients(local_lossa, list(weights.values()))
                    gradients = dict(zip(weights.keys(), grads))

                    # make fast gradient zero for weights with gradient None
                    for key in gradients.keys():
                        if gradients[key] is None:
                            gradients[key] = tf.zeros_like(fast_weights[key])
                    if FLAGS.stop_grad:
                        gradients = {key:tf.stop_gradient(gradients[key]) for key in gradients.keys()}
                    if FLAGS.clip:
                        clip_min = FLAGS.clip_min
                        clip_max = FLAGS.clip_max
                        for key in gradients.keys():
                            gradients[key] = tf.clip_by_value(gradients[key], clip_min, clip_max)
                    if FLAGS.pretrain_weight_path != 'N/A':
                        gradients['wc1'] = tf.zeros_like(gradients['wc1'])
                        gradients['bc1'] = tf.zeros_like(gradients['bc1'])
                    gradients_summ.append([gradients[key] for key in self.sorted_weight_keys])
                    
                    for key in fast_weights.keys():
                      self.step_size = lr_weights['lr_' + key + 'update%d'%(j+1)]
                      #self.update_dir = tf.sign(lr_weights['ld_' + key])
                      fast_weights[key] = fast_weights[key] - self.step_size*gradients[key]

                    #fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.step_size*gradients[key] for key in fast_weights.keys()]))

                    # Post-update
                    if FLAGS.no_state:
                        state_inputb = None
                    if 'Training' in prefix:
                        smaxb, output = self.forward(inputbs[j+1], state_inputb, fast_weights, meta_testing=True, network_config=network_config)
                    else:
                        smaxb, output = self.forward(inputbs[j+1], state_inputb, fast_weights, meta_testing=True, is_training=False, testing=testing, network_config=network_config)
                    local_outputbs.append(output)
                    smaxbs.append(smaxb)
                    lossb = euclidean_loss_layer(output, targetbs[j+1], multiplier=loss_multiplier, use_l1=FLAGS.use_l1_l2_loss)
                    local_lossesb.append(lossb)

                local_fn_output = [inputas, inputbs, local_outputas, local_outputbs, smaxas, smaxbs, local_outputbs[-1], local_lossesa, local_lossesb, flat_img_inputb, gradients_summ]
                return local_fn_output

        if self.norm_type:
            # initialize batch norm vars.
            unused = batch_metalearn((inputa[0], inputb[0], targeta[0], targetb[0]))

        out_dtype = [[tf.float32]*num_updates, [tf.float32]*num_updates, [tf.float32]*num_updates, [tf.float32]*num_updates, [tf.float32]*num_updates, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates, [tf.float32]*num_updates, tf.float32, [[tf.float32]*len(self.weights.keys())]*num_updates]
        result = tf.map_fn(batch_metalearn, elems=(inputa, inputb, targeta, targetb), dtype=out_dtype)
        print('Done with map.')
        return result
