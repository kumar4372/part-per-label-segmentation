import tensorflow as tf
import numpy as np
import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util


def get_transform_K(inputs, is_training, bn_decay=None, K = 3):
	""" Transform Net, input is BxNx1xK gray image
		Return:
			Transformation matrix of size KxK """
	batch_size = inputs.get_shape()[0].value
	num_point = inputs.get_shape()[1].value
	
	net = tf_util.conv2d(inputs, 256, [1,1], padding='VALID', stride=[1,1],
						 bn=True, is_training=is_training, scope='tconv1', bn_decay=bn_decay)
	net = tf_util.conv2d(net, 1024, [1,1], padding='VALID', stride=[1,1],
						 bn=True, is_training=is_training, scope='tconv2', bn_decay=bn_decay)
	net = tf_util.max_pool2d(net, [num_point,1], padding='VALID', scope='tmaxpool')
# (32, 1, 1, 1024)
	net = tf.reshape(net, [batch_size, -1])
	
	net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='tfc1', bn_decay=bn_decay)
	
	net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='tfc2', bn_decay=bn_decay)

	with tf.variable_scope('transform_feat') as sc:
		weights = tf.get_variable('weights', [256, K*K], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		biases = tf.get_variable('biases', [K*K], initializer=tf.constant_initializer(0.0), dtype=tf.float32) + tf.constant(np.eye(K).flatten(), dtype=tf.float32)
		transform = tf.matmul(net, weights)
		transform = tf.nn.bias_add(transform, biases)

	#transform = tf_util.fully_connected(net, 3*K, activation_fn=None, scope='tfc3')
	transform = tf.reshape(transform, [batch_size, K, K])
	return transform





def get_transform(point_cloud, is_training, bn_decay=None, K = 3):
	""" Transform Net, input is BxNx3 gray image
		Return:
			Transformation matrix of size 3xK """
	batch_size = point_cloud.get_shape()[0].value
	num_point = point_cloud.get_shape()[1].value

	input_image = tf.expand_dims(point_cloud, -1)
	net = tf_util.conv2d(input_image, 64, [1,3], padding='VALID', stride=[1,1],
						 bn=True, is_training=is_training, scope='tconv1', bn_decay=bn_decay)
	net = tf_util.conv2d(net, 128, [1,1], padding='VALID', stride=[1,1],
						 bn=True, is_training=is_training, scope='tconv3', bn_decay=bn_decay)
	net = tf_util.conv2d(net, 1024, [1,1], padding='VALID', stride=[1,1],
						 bn=True, is_training=is_training, scope='tconv4', bn_decay=bn_decay)
	net = tf_util.max_pool2d(net, [num_point,1], padding='VALID', scope='tmaxpool')

	net = tf.reshape(net, [batch_size, -1])
	net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='tfc1', bn_decay=bn_decay)
	net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='tfc2', bn_decay=bn_decay)

	with tf.variable_scope('transform_XYZ') as sc:
		assert(K==3)
		weights = tf.get_variable('weights', [128, 3*K], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		biases = tf.get_variable('biases', [3*K], initializer=tf.constant_initializer(0.0), dtype=tf.float32) + tf.constant([1,0,0,0,1,0,0,0,1], dtype=tf.float32)
		transform = tf.matmul(net, weights)
		transform = tf.nn.bias_add(transform, biases)

	#transform = tf_util.fully_connected(net, 3*K, activation_fn=None, scope='tfc3')
	transform = tf.reshape(transform, [batch_size, 3, K])
	return transform


def get_model(point_cloud,input_label, is_training, cat_num, \
		batch_size, num_point, weight_decay, bn_decay=None):
	""" ConvNet baseline, input is BxNx3 gray image """
	end_points = {}

	with tf.variable_scope('transform_net1') as sc:
		K = 3
		transform = get_transform(point_cloud, is_training, bn_decay, K = 3)
	point_cloud_transformed = tf.matmul(point_cloud, transform)

	input_image = tf.expand_dims(point_cloud_transformed, -1)
	out1 = tf_util.conv2d(input_image, 64, [1,K], padding='VALID', stride=[1,1],
						 bn=True, is_training=is_training, scope='conv1', bn_decay=bn_decay)
	# (32, 2048, 1, 64)

	out2 = tf_util.conv2d(out1, 128, [1,1], padding='VALID', stride=[1,1],
						 bn=True, is_training=is_training, scope='conv2', bn_decay=bn_decay)
	# (32, 2048, 1, 128)

	out3 = tf_util.conv2d(out2, 128, [1,1], padding='VALID', stride=[1,1],
						 bn=True, is_training=is_training, scope='conv3', bn_decay=bn_decay)
	# (32, 2048, 1, 128)


	with tf.variable_scope('transform_net2') as sc:
		K = 128
		transform = get_transform_K(out3, is_training, bn_decay, K)

	end_points['transform'] = transform
	# (32, 128, 128)


	
	squeezed_out3 = tf.reshape(out3, [batch_size, num_point, 128])
	# (32, 2048, 128)

	
	net_transformed = tf.matmul(squeezed_out3, transform)
	# (32, 2048, 128)

	
	net_transformed = tf.expand_dims(net_transformed, [2])
   # (32, 2048, 1, 128)

	out4 = tf_util.conv2d(net_transformed, 512, [1,1], padding='VALID', stride=[1,1],
						 bn=True, is_training=is_training, scope='conv4', bn_decay=bn_decay)
   # (32, 2048, 1, 512)

	out5 = tf_util.conv2d(out4, 3000, [1,1], padding='VALID', stride=[1,1],
						 bn=True, is_training=is_training, scope='conv5', bn_decay=bn_decay)
	# (32, 2048, 1, 2048)


	out_max = tf_util.max_pool2d(out5, [num_point,1], padding='VALID', scope='maxpool')
	# (32, 1, 1, 2048)


	# classification network
	# net = tf.reshape(out_max, [batch_size, -1])
	# net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='cla/fc1', bn_decay=bn_decay)
	# net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='cla/fc2', bn_decay=bn_decay)
	# net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='cla/dp1')
	# net = tf_util.fully_connected(net, cat_num, activation_fn=None, scope='cla/fc3')

	# segmentation network
	one_hot_label_expand = tf.reshape(input_label, [batch_size, 1, 1, cat_num])
# (32, 1, 1, 16)
	out_max = tf.concat(axis=3, values=[out_max, one_hot_label_expand])
# (32, 1, 1, 2064)
	expand = tf.tile(out_max, [1, num_point, 1, 1])
# (32, 2048, 1, 2064)
	
	concat = tf.concat(axis=3, values=[expand, out1, out2, out3, out4, out5])
# (32, 2048, 1, 4944)

	net2 = tf_util.conv2d(concat, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
						bn=True, is_training=is_training, scope='seg/conv1', weight_decay=weight_decay)
# (32, 2048, 1, 256)

	net2 = tf_util.dropout(net2, keep_prob=0.8, is_training=is_training, scope='seg/dp1')
# (32, 2048, 1, 256)
	
	net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
						bn=True, is_training=is_training, scope='seg/conv2', weight_decay=weight_decay)
# (32, 2048, 1, 256)
	
	net2 = tf_util.dropout(net2, keep_prob=0.8, is_training=is_training, scope='seg/dp2')
# (32, 2048, 1, 256)
	
	net2 = tf_util.conv2d(net2, 128, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
						bn=True, is_training=is_training, scope='seg/conv3', weight_decay=weight_decay)
# (32, 2048, 1, 128)
	
	net2 = tf_util.conv2d(net2,50, [1,1], padding='VALID', stride=[1,1], activation_fn=None, 
						bn=False, scope='seg/conv4', weight_decay=weight_decay)

# (32, 2048, 1, 50)
	
	net2 = tf.reshape(net2, [batch_size, num_point, 50])
	# (32, 2048,50)

	return net2, end_points

def get_loss( seg_pred, seg, end_points):
	# per_instance_label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=l_pred, labels=label)
	# label_loss = tf.reduce_mean(per_instance_label_loss)

	# size of seg_pred is batch_size x point_num x part_cat_num
	# size of seg is batch_size x point_num
	per_instance_seg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_pred, labels=seg), axis=1)
	# print(per_instance_seg_loss.get_shape())
	seg_loss = tf.reduce_mean(per_instance_seg_loss)
	# print(seg_loss.get_shape())

	per_instance_seg_pred_res = tf.argmax(seg_pred, 2)
	
	# Enforce the transformation as orthogonal matrix
	transform = end_points['transform'] # BxKxK
	K = transform.get_shape()[1].value
	mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1])) - tf.constant(np.eye(K), dtype=tf.float32)
	mat_diff_loss = tf.nn.l2_loss(mat_diff) 
	

	total_loss = seg_loss  + mat_diff_loss * 1e-3

	return total_loss,seg_loss, per_instance_seg_loss, per_instance_seg_pred_res

