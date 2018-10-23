import tensorflow as tf
import numpy as np
import os
import sys 
sys.path.append('utils')
import cls_provider as provider
import colorsys
from datetime import datetime

def _variable_on_cpu(name, shape, initializer, use_fp16=False):
    """Helper to create a Variable stored on CPU memory.
    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    use_xavier: bool, whether to use xavier initializer

    Returns:
    Variable Tensor
    """
    if use_xavier:
        initializer = tf.contrib.layers.xavier_initializer()
    else:
        initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = _variable_on_cpu(name, shape, initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
    """ 2D convolution with non-linear operation.

    Args:
        inputs: 4-D tensor variable BxHxWxC
        num_output_channels: int
        kernel_size: a list of 2 ints
        scope: string
        stride: a list of 2 ints
        padding: 'SAME' or 'VALID'
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        weight_decay: float
        activation_fn: function
        bn: bool, whether to use batch norm
        bn_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

    Returns:
        Variable tensor
    """
    
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w,
                        num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                            shape=kernel_shape,
                                            use_xavier=use_xavier,
                                            stddev=stddev,
                                            wd=weight_decay)
        stride_h, stride_w = stride
        outputs = tf.nn.conv2d(inputs, kernel,
                                [1, stride_h, stride_w, 1],
                                padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv2d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs

def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
  """ Batch normalization on convolutional maps and beyond...
  Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
  
  Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      is_training:   boolean tf.Varialbe, true indicates training phase
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
  Return:
      normed:        batch-normalized maps
  """
  with tf.variable_scope(scope) as sc:
    num_channels = inputs.get_shape()[-1].value
    beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                       name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                        name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
    decay = bn_decay if bn_decay is not None else 0.9
    ema = tf.train.ExponentialMovingAverage(decay=decay)
    # Operator that maintains moving averages of variables.
    ema_apply_op = tf.cond(is_training,
                           lambda: ema.apply([batch_mean, batch_var]),
                           lambda: tf.no_op())
    
    # Update moving average and return current batch's avg and var.
    def mean_var_with_update():
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)
    
    # ema.average returns the Variable holding the average of var.
    mean, var = tf.cond(is_training,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
  return normed
  
def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope):
    """ Batch normalization on 2D convolutional maps.
    
    Args:
        inputs:      Tensor, 4D BHWC input maps
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0,1,2], bn_decay)

def batch_norm_for_fc(inputs, is_training, bn_decay, scope):
    """ Batch normalization on FC data.
    
    Args:
        inputs:      Tensor, 2D BxC input
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0,], bn_decay)

def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.0,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    is_training=None):
    """ Fully connected layer with non-linear operation.
    
    Args:
        inputs: 2-D tensor BxN
        num_outputs: int
    
    Returns:
        Variable tensor of size B x num_outputs.
    """
    with tf.variable_scope(scope) as sc:
        num_input_units = inputs.get_shape()[-1].value
        weights = _variable_with_weight_decay('weights',
                                            shape=[num_input_units, num_outputs],
                                            use_xavier=use_xavier,
                                            stddev=stddev,
                                            wd=weight_decay)
        outputs = tf.matmul(inputs, weights)
        biases = _variable_on_cpu('biases', [num_outputs],
                                tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)
        
        if bn:
            outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs

class Pointnet:

    def __init__(
            self,
                mode,
                nb_class,
                input_channel,
                batch_size          = 32,
                nb_pt               = 1024,
                base_laerning_rate  = 0.001,
                gpu_idx             = 0,
                momentum            = 0.9,
                decay_step          = 200000,
                decay_rate          = 0.7,
                log_dir             = 'logs',
                bn_init_decay       = 0.5,
                bn_decay_decay_rate = 0.5,
                bn_decay_decay_step = 200000,
                bn_decay_clip       = 0.9,
                is_training         = True,
        ):
        """
            mode, string classification | segmentation
            nb_class, int 
            input_channel, int
            batch_size, int default 32
            nb_pt, int default 1024
            base_laerning_rate, float default 0.001
            gpu_idx, int default 0,
            momentum, float default 0.9
            decay_step, int 200000
            decay_rate, float 0.7
            log_dir, string default logs
            bn_init_decay, float default 0.5
            bn_decay_decay_rate, float 0.5
            bn_decay_decay_step, float 200000
            bn_decay_clip, float 0.9
            is_training, bool True
        """
        opts = {
            'mode': mode,
            'nb_class': nb_class,
            'input_channel': input_channel,
            'batch_size': batch_size,
            'nb_pt': nb_pt,
            'base_laerning_rate': base_laerning_rate,
            'gpu_idx': gpu_idx,
            'momentum': momentum,
            'decay_step': decay_step,
            'decay_rate': decay_rate,
            'log_dir': log_dir,
            'bn_init_decay': bn_init_decay,
            'bn_decay_decay_rate': bn_decay_decay_rate,
            'bn_decay_decay_step': bn_decay_decay_step,
            'bn_decay_clip': bn_decay_clip
        }

        self.opts = opts
        self.create_grpah()
        self.create_loss()

        self.saver = tf.train.Saver()
        
        hex_color, rgb_color = get_N_color(nb_class)
        self.color_scale_rgb = rgb_color
        self.color_scale_hex = hex_color


    def create_grpah(self):
        opts = self.opts

        end_points = {}

        # label placeholder
        pts_pl = tf.placeholder(tf.float32, shape=(opts['batch_size'], opts['nb_pt'], opts['input_channel']))
        end_points['pts_pl'] = pts_pl

        # label placeholder

        if self.opts['mode'] == "classification":
            labels_pl = tf.placeholder(tf.int32, shape=(opts['batch_size']))
            end_points['labels_pl'] = labels_pl
        else:
            labels_pl = tf.placeholder(tf.int32, shape=(opts['batch_size'], opts['nb_pt']))
            end_points['labels_pl'] = labels_pl

        # batch var 
        batch = tf.Variable(0)
        end_points['batch'] = batch

        # is training_pl
        is_training_pl = tf.placeholder(tf.bool, shape=())
        end_points['is_training_pl'] = is_training_pl        

        # learning rate
        learning_rate = tf.train.exponential_decay(
                            opts['base_laerning_rate'],  # Base learning rate.
                            batch * opts['batch_size'],  # Current index into the dataset.
                            opts['decay_step'],          # Decay step.
                            opts['decay_rate'],          # Decay rate.
                            staircase=True)
        learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
        end_points['learning_rate'] = learning_rate 

        # bn_decay
        bn_momentum = tf.train.exponential_decay(
                        opts['bn_init_decay'],
                        batch * opts['batch_size'],  # Current index into the dataset.
                        opts['bn_decay_decay_step'],
                        opts['bn_decay_decay_rate'],
                        staircase=True)
        bn_decay = tf.minimum(opts['bn_decay_clip'], 1 - bn_momentum)

        # T-net 1
        with tf.variable_scope('t_net_1') as sc:
            net = tf.expand_dims(pts_pl, -1)
            

            for idx, size in enumerate([64, 128, 1024]):
                kernel_size = [1,1]
                if(idx == 0):
                    kernel_size = [1, self.opts['input_channel']]
                net  = conv2d(net, size, kernel_size,
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training_pl,
                            scope='t1conv'+str(idx+1), bn_decay=bn_decay)

            net = tf.nn.max_pool(net, 
                                ksize=[1,opts['nb_pt'],1,1],
                                strides=[1,2,2,1],
                                padding='VALID')
            net = tf.reshape(net, [opts['batch_size'], -1])
            net = fully_connected(net, 512, bn=True, is_training=is_training_pl,
                                  scope='tfc1', bn_decay=bn_decay)
            net = fully_connected(net, 256, bn=True, is_training=is_training_pl,
                                  scope='tfc2', bn_decay=bn_decay)
            
            with tf.variable_scope('transform_input') as sc:
                weights = tf.get_variable('weights', [256, opts['input_channel']*opts['input_channel']],
                                        initializer=tf.constant_initializer(0.0),
                                        dtype=tf.float32)
                biases = tf.get_variable('biases', [opts['input_channel']*opts['input_channel']],
                                        initializer=tf.constant_initializer(0.0),
                                        dtype=tf.float32)
                biases += tf.constant(np.eye(opts['input_channel']).flatten(), dtype=tf.float32)
                transform = tf.matmul(net, weights)
                transform = tf.nn.bias_add(transform, biases)
            
            transform_matrix = tf.reshape(transform, [opts['batch_size'], opts['input_channel'], opts['input_channel']])

        transformed_pts = tf.matmul(pts_pl, transform_matrix)
        
        # Mlp 1
        with tf.variable_scope('mlp1') as sc:
            net = tf.expand_dims(transformed_pts, -1)
            for idx, size in enumerate([64, 64]):
                kernel_size = [1,1]
                if(idx == 0):
                    kernel_size = [1, self.opts['input_channel']]
                net  = conv2d(net, size, kernel_size,
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training_pl,
                            scope='ml1_conv'+str(idx+1), bn_decay=bn_decay)
        ml1_out = net
        end_points['ml1_out'] = ml1_out
            
        # T-net 2
        with tf.variable_scope('t_net_2') as sc:
            for idx, size in enumerate([64, 128, 1024]):
                net  = conv2d(net, size, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training_pl,
                            scope='t2conv'+str(idx+1), bn_decay=bn_decay)

            net = tf.nn.max_pool(net, 
                                ksize=[1,opts['nb_pt'],1,1],
                                strides=[1,2,2,1],
                                padding='VALID')
            net = tf.reshape(net, [opts['batch_size'], -1])
            net = fully_connected(net, 512, bn=True, is_training=is_training_pl,
                                  scope='tfc1', bn_decay=bn_decay)
            net = fully_connected(net, 256, bn=True, is_training=is_training_pl,
                                  scope='tfc2', bn_decay=bn_decay)

            feature_size = 64
            with tf.variable_scope('transform_feature') as sc:
                weights = tf.get_variable('weights', [256, feature_size*feature_size],
                                        initializer=tf.constant_initializer(0.0),
                                        dtype=tf.float32)
                biases = tf.get_variable('biases', [feature_size*feature_size],
                                        initializer=tf.constant_initializer(0.0),
                                        dtype=tf.float32)
                biases += tf.constant(np.eye(feature_size).flatten(), dtype=tf.float32)
                transform = tf.matmul(net, weights)
                transform = tf.nn.bias_add(transform, biases)
            
            transform_matrix = tf.reshape(transform, [opts['batch_size'], feature_size, feature_size])
        end_points['transform_feature_matrix'] = transform_matrix
        net_transformed = tf.matmul(tf.squeeze(ml1_out, axis=[2]), transform_matrix)
        net = tf.expand_dims(net_transformed, [2])
        pt_feature = net

        # Mlp 2
        with tf.variable_scope('mlp2') as sc:
            for idx, size in enumerate([64, 128, 1024]):
                net  = conv2d(net, size, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training_pl,
                            scope='ml2_conv'+str(idx+1), bn_decay=bn_decay)
        ml2_out = net
        end_points['ml2_out'] = ml2_out

        net = tf.nn.max_pool(net, 
                            ksize=[1,opts['nb_pt'],1,1],
                            strides=[1,2,2,1],
                            padding='VALID')

        if self.opts['mode'] == "classification":
            # Mlp 3
            with tf.variable_scope('cls_mlp3') as sc:
                net = tf.reshape(net, [opts['batch_size'], -1])
                net = fully_connected(net, 512, bn=True, is_training=is_training_pl,
                                        scope='fc1', bn_decay=bn_decay)
                net = tf.cond(is_training_pl,
                                lambda: tf.nn.dropout(net, 0.7),
                                lambda: net)
                net = fully_connected(net, 256, bn=True, is_training=is_training_pl,
                                        scope='fc2', bn_decay=bn_decay)
                net = tf.cond(is_training_pl,
                                lambda: tf.nn.dropout(net, 0.7),
                                lambda: net)
                net = fully_connected(net, opts['nb_class'], bn=True, is_training=is_training_pl,
                                        scope='fc3', bn_decay=bn_decay)
        else:
            global_feature = net 
            global_feature_exp = tf.tile(tf.reshape(global_feature, [self.opts['batch_size'], 1, 1, -1]), [1, self.opts['nb_pt'], 1, 1])
            print(global_feature_exp.get_shape())
            concated_feature = tf.concat([pt_feature, global_feature_exp],3)
            print(concated_feature.get_shape())
            net = concated_feature
            
            with tf.variable_scope('seg_mlp_3') as sc:
                for idx, size in enumerate([512,256,128]):
                    net  = conv2d(net, size, [1,1],
                                padding='VALID', stride=[1,1],
                                bn=True, is_training=is_training_pl,
                                scope='ml3_conv'+str(idx+1), bn_decay=bn_decay)
            
            net  = conv2d(net, self.opts['nb_class'], [1,1],
                        padding='VALID', stride=[1,1],
                        bn=True, is_training=is_training_pl,
                        activation_fn=None,
                        scope='ml3_conv4', bn_decay=bn_decay)
            net = tf.squeeze(net, [2]) # BxNxC
        
        end_points['output'] = net
        self.end_points = end_points

    def create_loss(self, reg_weight=0.001):
        
        print(self.end_points['output'].get_shape(), self.end_points['labels_pl'].get_shape())
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.end_points['output'], labels=self.end_points['labels_pl'])
        classify_loss = tf.reduce_mean(loss)
        tf.summary.scalar('classify_loss', classify_loss)

        if self.opts['mode'] == "classification":
            # Enforce the transformation as orthogonal matrix
            transform = self.end_points['transform_feature_matrix'] # BxKxK
            K = transform.get_shape()[1].value
            mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
            mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
            mat_diff_loss = tf.nn.l2_loss(mat_diff) 
            tf.summary.scalar('mat_loss', mat_diff_loss)

            self.end_points['loss'] = classify_loss + mat_diff_loss * reg_weight
            tf.summary.scalar('loss', self.end_points['loss'])
            correct = tf.equal(tf.argmax(self.end_points['output'], 1), tf.to_int64(self.end_points['labels_pl']))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(self.opts['batch_size'])
            tf.summary.scalar('accuracy', accuracy)
        else:
            self.end_points['loss'] = classify_loss
            loss = classify_loss           
            correct = tf.equal(tf.argmax(self.end_points['output'], 2), tf.to_int64(self.end_points['labels_pl']))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(self.opts['batch_size']*self.opts['nb_pt'])
            tf.summary.scalar('accuracy', accuracy)



        tf.summary.scalar('learning_rate', self.end_points['learning_rate'])
        optimizer = tf.train.AdamOptimizer(self.end_points['learning_rate'])
        train_op = optimizer.minimize(loss, global_step=self.end_points['batch'])
        self.end_points['train_op'] = train_op
        


    def train(self, sess, files):
        """Train one epoch. """
        is_training = True
        
        # Shuffle train files
        train_file_idxs = np.arange(0, len(files))
        np.random.shuffle(train_file_idxs)
    
        for fn in range(len(files)):
            current_data, current_label = provider.loadDataFile(files[train_file_idxs[fn]])
            current_data = current_data[:,0:self.opts['nb_pt'],:]
            current_label = current_label[:, 0:self.opts['nb_pt']]
            current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
            current_label = np.squeeze(current_label)

            file_size = current_data.shape[0]
            num_batches = file_size // self.opts['batch_size']

            total_correct = 0
            total_seen = 0
            loss_sum = 0

            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.opts['batch_size']
                end_idx = (batch_idx+1) * self.opts['batch_size']

                if self.opts['mode'] == "classification":
                    # Augment batched point clouds by rotation and jittering
                    rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
                    data = provider.jitter_point_cloud(rotated_data)
                    feed_dict = {self.end_points['pts_pl']: data,
                                self.end_points['labels_pl']: current_label[start_idx:end_idx],
                                self.end_points['is_training_pl']: is_training,}
                else:
                    # Augment batched point clouds by rotation and jittering
                    data = current_data[start_idx:end_idx, :, :]
                    feed_dict = {self.end_points['pts_pl']: data,
                                self.end_points['labels_pl']: current_label[start_idx:end_idx],
                                self.end_points['is_training_pl']: is_training,}
                summary, step, _, loss_val, pred_val = sess.run([   self.end_points['merged'], 
                                                                    self.end_points['batch'],
                                                                    self.end_points['train_op'], 
                                                                    self.end_points['loss'], 
                                                                    self.end_points['output']
                                                                ], 
                                                                feed_dict=feed_dict)
                self.end_points['train_writer'].add_summary(summary, step)
                if len(current_label.shape) == 1:
                    pred_val = np.argmax(pred_val, 1)
                    correct = np.sum(pred_val == current_label[start_idx:end_idx])
                    total_seen += self.opts['batch_size']
                    total_correct += correct
                    loss_sum += loss_val
                else:
                    pred_val = np.argmax(pred_val, 2)
                    correct = np.sum(pred_val == current_label[start_idx:end_idx])
                    total_correct += correct
                    total_seen += (self.opts['batch_size']*self.opts['nb_pt'])
                    loss_sum += loss_val
                    

        self.log_string('mean loss: %f' % (loss_sum / float(num_batches)))
        self.log_string('accuracy: %f' % (total_correct / float(total_seen)))
    
    def log_string(self, out_str):
        self.LOG_FOUT.write(str(out_str)+'\n')
        self.LOG_FOUT.flush()
        print(out_str)

    def create_log(self, sess):
        if not os.path.exists(self.opts['log_dir']): 
            os.mkdir(self.opts['log_dir'])
        else:
            import shutil
            new_log_path = datetime.now().strftime(self.opts['log_dir']+'%H_%M_%d_%m_%Y')
            shutil.move(self.opts['log_dir'], new_log_path)
            os.mkdir(self.opts['log_dir'])
        self.LOG_FOUT = open(os.path.join(self.opts['log_dir'], 'log_train.txt'), 'w')

        merged = tf.summary.merge_all()
        self.end_points['merged'] = merged
        train_writer = tf.summary.FileWriter(os.path.join(self.opts['log_dir'], 'train'), sess.graph)
        self.end_points['train_writer'] = train_writer
        
        
        test_writer = tf.summary.FileWriter(os.path.join(self.opts['log_dir'], 'test'))
        self.end_points['test_writer'] = test_writer


    def test(self, sess, files, save_test_result=False):
        """
        save_test_result: bool it define will programe save the result, ground truth of segmentation 
        Test one epoch. 
        """
        is_training = False
        total_correct = 0.0
        total_seen = 0.0
        loss_sum = 0.0
        total_seen_class = [0 for _ in range(self.opts['nb_class'])]
        total_correct_class = [0 for _ in range(self.opts['nb_class'])]
        
        
        gt_classes = [0 for _ in range(self.opts['nb_class'])]
        positive_classes = [0 for _ in range(self.opts['nb_class'])]
        true_positive_classes = [0 for _ in range(self.opts['nb_class'])]
        
        for fn in range(len(files)):
#             self.log_string('----' + str(fn) + '-----')
            current_data, current_label = provider.loadDataFile(files[fn])
            current_data = current_data[:,0:self.opts['nb_pt'],:]
            current_label = np.squeeze(current_label)
            
            file_size = current_data.shape[0]
            num_batches = file_size // self.opts['batch_size']
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.opts['batch_size']
                end_idx = (batch_idx+1) * self.opts['batch_size']

                if self.opts['mode'] == "classification":
                    feed_dict = {self.end_points['pts_pl']: current_data[start_idx:end_idx, 0:self.opts['nb_pt'], :],
                                self.end_points['labels_pl']: current_label[start_idx:end_idx],
                                self.end_points['is_training_pl']: is_training,}
                else:                

                    feed_dict = {self.end_points['pts_pl']: current_data[start_idx:end_idx, 0:self.opts['nb_pt'], :],
                                self.end_points['labels_pl']: current_label[start_idx:end_idx, 0:self.opts['nb_pt']],
                                self.end_points['is_training_pl']: is_training,}
                
                summary, step, loss_val, pred_val = sess.run([   self.end_points['merged'], 
                                                                    self.end_points['batch'],
                                                                    self.end_points['loss'], 
                                                                    self.end_points['output']
                                                                ], 
                                                                feed_dict=feed_dict)
                self.end_points['test_writer'].add_summary(summary, step)
                
                ## save test result
                if save_test_result and self.opts['mode'] == "segmentation":
                    log_path = self.opts['log_dir']+'/seg_result'
                    if not os.path.exists(log_path): 
                        os.mkdir(log_path)
                        
                    pred_cls = np.argmax(pred_val, 2)
                    for i in range(start_idx, end_idx):
                        mpred_cls = pred_cls[i-start_idx, 0:self.opts['nb_pt']]
                        in_pts = current_data[i, 0:self.opts['nb_pt']]
                        gt_cls = current_label[i, 0:self.opts['nb_pt']]
                        
                        name = datetime.now().strftime('%S_%M_%H_T_%Y_%m_%d.xyzrgb')
                        
                        raw_path = log_path+'/raw_'+ name
                        gt_path = log_path+'/gt_'+ name
                        pred_path = log_path+'/pred_'+ name
                        
                        self.saveSegmentationResult(raw_path, current_data[i], current_label[i])
                        self.saveSegmentationResult(gt_path, in_pts, gt_cls)
                        self.saveSegmentationResult(pred_path, in_pts, mpred_cls)
                        
#                         self.saveSegmentationResult(gt_path, pred_path, current_data[i, 0:self.opts['nb_pt']], mpred_cls, gt_cls)
                                       
                
                
                if self.opts['mode'] == "classification":
                    pred_val = np.argmax(pred_val, 1)
                    correct = np.sum(pred_val == current_label[start_idx:end_idx])
                    total_correct += correct
                    total_seen += self.opts['batch_size']
                    loss_sum += (loss_val*self.opts['batch_size'])
                    for i in range(start_idx, end_idx):
                        l = current_label[i]
                        total_seen_class[l] += 1
                        total_correct_class[l] += (pred_val[i-start_idx] == l)
                else:                    
                    pred_val = np.argmax(pred_val, 2)
                    correct = np.sum(pred_val == current_label[start_idx:end_idx])
                    total_seen += (self.opts['batch_size']*self.opts['nb_pt'])
                    loss_sum += (loss_val*self.opts['batch_size'])
                    for i in range(start_idx, end_idx):
                        for j in range(self.opts['nb_pt']):
                            l = current_label[i, j]
                            total_seen_class[l] += 1
                            total_correct_class[l] += (pred_val[i-start_idx, j] == l)
                            total_correct += (pred_val[i-start_idx, j] == l)
                            
                        
                    for i in range(start_idx, end_idx):
                        pred = pred_val[i-start_idx,:self.opts['nb_pt']]
                        label = current_label[i, :self.opts['nb_pt']]
                        for j in xrange(self.opts['nb_pt']):
                            gt_l = pred[j]
                            pred_l = label[j]
                            gt_classes[gt_l] += 1
                            positive_classes[pred_l] += 1
                            true_positive_classes[gt_l] += int(gt_l==pred_l)          
         
        if self.opts['mode'] == "classification":
            pass
        else:
            print 'IoU:'
            iou_list = []
            for i in range(13):
                iou = true_positive_classes[i]/float(gt_classes[i]+positive_classes[i]-true_positive_classes[i]) 
                print(iou)
                iou_list.append(iou)
            self.log_string("Mean iou: ")
            self.log_string(sum(iou_list)/13.0)
        
        self.log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
        self.log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
        self.log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))

    def saveSegmentationResult(self, filename, points, mcls):
        
        with open(filename, 'w') as file:
            for row_idx, row in enumerate(points):
                for col in row:
                    file.write(str(col)+' ')
                # write the color 
                for c in self.color_scale_rgb[mcls[row_idx]] :
                    file.write(str(c)+' ')
                file.write('\n')
        os.chmod(filename, 0o777) 

def get_N_color(N=5):
    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in xrange(N)]
    hex_out = []
    rgb_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x*255),colorsys.hsv_to_rgb(*rgb))
        rgb_out.append(rgb)
        hex_out.append("".join(map(lambda x: chr(x).encode('hex'),rgb)))
    return hex_out, rgb_out

    

def savePoints(filename, points):
    """
        filename: string asb path
        points: npArray nbPts:n
    """    
    with open(filename, 'w') as file:
        for row in points:
            for col in row:
                file.write(str(col)+' ')
            file.write('\n')
    os.chmod(filename, 0o777)
        
# def confusion_matrix(eval_segm, gt_segm, **kwargs):
#     merged_maps = np.bitwise_or(np.left_shift(gt_segm.astype('uint16'), 8), eval_segm.astype('uint16'))
#     hist = np.bincount(np.ravel(merged_maps))
#     nonzero = np.nonzero(hist)[0]
#     pred, label = np.bitwise_and(255, nonzero), np.right_shift(nonzero, 8)
#     class_count = np.array([pred, label]).max() + 1
#     conf_matrix = np.zeros([class_count, class_count], dtype='uint64')
#     conf_matrix.put(pred * class_count + label, hist[nonzero])
#     return conf_matrix

# def computeIoU(y_pred_batch, y_true_batch, N_CLASSES_PASCAL):
#     return np.mean(np.asarray([pixelAccuracy(y_pred_batch[i], y_true_batch[i], N_CLASSES_PASCAL) for i in range(len(y_true_batch))])) 

# def pixelAccuracy(y_pred, y_true, N_CLASSES_PASCAL):
#     y_pred = np.argmax(np.reshape(y_pred,[N_CLASSES_PASCAL,img_rows,img_cols]),axis=0)
#     y_true = np.argmax(np.reshape(y_true,[N_CLASSES_PASCAL,img_rows,img_cols]),axis=0)
#     y_pred = y_pred * (y_true>0)

#     return 1.0 * np.sum((y_pred==y_true)*(y_true>0)) /  np.sum(y_true>0)