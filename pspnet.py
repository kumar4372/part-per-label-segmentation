import tensorflow as tf
import numpy as np
import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
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


def get_model(point_cloud, input_label, is_training, cat_num, \
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
	# (10, 3000, 1, 3000)

	out_max1 = tf_util.max_pool2d(out5, [num_point,1], padding='VALID', scope='maxpool1')
# (10, 1, 1, 3000)
	out_max2 = tf_util.max_pool2d(out5, [1999,1], padding='VALID', scope='maxpool2')
	# (10, 501, 1, 3000)
	out_max3 = tf_util.max_pool2d(out5, [1499,1], padding='VALID', scope='maxpool3')
	# (10, 751, 1, 3000)
	out_max4 = tf_util.max_pool2d(out5, [999,1], padding='VALID', scope='maxpool4')
	# (10, 1001, 1, 3000)
	out_max5 = tf_util.max_pool2d(out5, [499,1], padding='VALID', scope='maxpool5')
	# (10, 1251, 1, 3000)

# Autoencoder for out_max1
	# Autoencoder for net_transformed
	out40 = tf_util.conv2d(out_max1, 512, [1,1], padding='VALID', stride=[1,1],
						 bn=True, is_training=is_training, scope='conv40', bn_decay=bn_decay)
   # (10, 1, 1, 512)
	out50 = tf_util.conv2d(out40, num_point, [1,1], padding='VALID', stride=[1,1],
						 bn=True, is_training=is_training, scope='conv50', bn_decay=bn_decay)
	# (10, 1, 1, 3000)
	out50 = tf_util.conv2d_transpose(out50, num_point, [1,1], padding='VALID', stride=[1,1],
						 bn=True, is_training=is_training, scope='conv51', bn_decay=bn_decay)
	# (10, 1, 1, 3000)
	print(out5.get_shape())
	out50 = tf_util.conv2d_transpose(out50, 512, [1,1], padding='VALID', stride=[1,1],
						 bn=True, is_training=is_training, scope='conv511', bn_decay=bn_decay)
	# (10, 1, 1, 512)
	out_max00 = tf_util.max_pool2d(out50, [1,1], padding='VALID', scope='maxpool50')

# Autoencoder for out_max2
	out41 = tf_util.conv2d(out_max2, 512, [1,1], padding='VALID', stride=[1,1],
						 bn=True, is_training=is_training, scope='conv41', bn_decay=bn_decay)
   # (10, 501, 1, 512)
	out51 = tf_util.conv2d(out41, num_point, [1,1], padding='VALID', stride=[1,1],
						 bn=True, is_training=is_training, scope='conssv51', bn_decay=bn_decay)
	# (10, 501, 1, 2048)
	out51 = tf_util.conv2d_transpose(out51, num_point, [1,1], padding='VALID', stride=[1,1],
						 bn=True, is_training=is_training, scope='conssv512', bn_decay=bn_decay)
	# (10, 501, 1, 2048)
	out51= tf_util.conv2d_transpose(out51, 512, [1,1], padding='VALID', stride=[1,1],
						 bn=True, is_training=is_training, scope='conv5211', bn_decay=bn_decay)
	# (10, 501, 1, 2048)
	out_max11 = tf_util.max_pool2d(out51, [501,1], padding='VALID', scope='maxpool51')
	# (10, 1, 1, 2048)

# Autoencoder for out_max3
	out42 = tf_util.conv2d(out_max3, 512, [1,1], padding='VALID', stride=[1,1],
						 bn=True, is_training=is_training, scope='conv42', bn_decay=bn_decay)
   # (32, 2048, 1, 512)
	out52 = tf_util.conv2d(out42, num_point, [1,1], padding='VALID', stride=[1,1],
						 bn=True, is_training=is_training, scope='conv52', bn_decay=bn_decay)
	# (32, 2048, 1, 2048)
	out52 = tf_util.conv2d_transpose(out52, num_point, [1,1], padding='VALID', stride=[1,1],
						 bn=True, is_training=is_training, scope='conv514', bn_decay=bn_decay)
	# (32, 2048, 1, 2048)
	out52 = tf_util.conv2d_transpose(out52, 512, [1,1], padding='VALID', stride=[1,1],
						 bn=True, is_training=is_training, scope='conv5151', bn_decay=bn_decay)
	# (32, 2048, 1, 2048)
	out_max22 = tf_util.max_pool2d(out52, [751,1], padding='VALID', scope='maxpool52')
	# (32, 1, 1, 2048)

# Autoencoder for out_max4

	out43 = tf_util.conv2d(out_max4, 512, [1,1], padding='VALID', stride=[1,1],
						 bn=True, is_training=is_training, scope='conv43', bn_decay=bn_decay)
   # (32, 2048, 1, 512)

	out53 = tf_util.conv2d(out43, num_point, [1,1], padding='VALID', stride=[1,1],
						 bn=True, is_training=is_training, scope='conv53', bn_decay=bn_decay)
	# (32, 2048, 1, 2048)

	out53 = tf_util.conv2d_transpose(out53, num_point, [1,1], padding='VALID', stride=[1,1],
						 bn=True, is_training=is_training, scope='condv51', bn_decay=bn_decay)
	# (32, 2048, 1, 2048)
	out53 = tf_util.conv2d_transpose(out53, 512, [1,1], padding='VALID', stride=[1,1],
						 bn=True, is_training=is_training, scope='codnv511', bn_decay=bn_decay)
	# (32, 2048, 1, 2048)
	out_max33 = tf_util.max_pool2d(out53, [1001,1], padding='VALID', scope='maxpool53')
	# (32, 1, 1, 2048)

# Autoencoder for out_max5

	out44 = tf_util.conv2d(out_max5, 512, [1,1], padding='VALID', stride=[1,1],
						 bn=True, is_training=is_training, scope='conv44', bn_decay=bn_decay)
   # (32, 2048, 1, 512)

	out54 = tf_util.conv2d(out44, num_point, [1,1], padding='VALID', stride=[1,1],
						 bn=True, is_training=is_training, scope='conv54', bn_decay=bn_decay)
	# (32, 2048, 1, 2048)
	out54 = tf_util.conv2d_transpose(out54, num_point, [1,1], padding='VALID', stride=[1,1],
						 bn=True, is_training=is_training, scope='conefv51', bn_decay=bn_decay)
	# (32, 2048, 1, 2048)
	out54 = tf_util.conv2d_transpose(out54, 512, [1,1], padding='VALID', stride=[1,1],
						 bn=True, is_training=is_training, scope='coadfnv511', bn_decay=bn_decay)
	# (32, 2048, 1, 2048)

	out_max44 = tf_util.max_pool2d(out54, [1251,1], padding='VALID', scope='maxpool54')
	# (32, 1, 1, 2048)


	print('hi')
	# print(out_max11.get_shape())
	# print(out_max22.get_shape())
	# print(out_max33.get_shape())
	# print(out_max44.get_shape())
# 	print(out5.get_shape())
# 	print(out51.get_shape())
# 	print(out52.get_shape())
# 	print(out53.get_shape())
# 	print(out54.get_shape())
# # (10, 3000, 1, 512)
# (10, 501, 1, 512)
# (10, 751, 1, 512)
# (10, 1001, 1, 512)
# (10, 1251, 1, 512)


	# segmentation network
	one_hot_label_expand = tf.reshape(input_label, [batch_size, 1, 1, cat_num])
	one_hot_label_expand = tf.tile(one_hot_label_expand,[1,1,1,5])
# (32, 1, 1, 16)
	out_max = tf.concat(axis=3, values=[out_max00, out_max11,out_max22,out_max33,out_max44, one_hot_label_expand])
# (32, 1, 1, 2064)
	expand = tf.tile(out_max, [1, num_point, 1, 1])
# (10, 3000, 1, 3080)
	
	concat = tf.concat(axis=3, values=[expand, out1, out2, out3, out4, out5])
# (10, 3000, 1, 6472)
	print(concat.get_shape())
	net2 = tf_util.conv2d(concat, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
						bn=True, is_training=is_training, scope='seg/conv1', weight_decay=weight_decay)
# (10, 3000, 1, 256)
	print(net2.get_shape())

	net2 = tf_util.dropout(net2, keep_prob=0.8, is_training=is_training, scope='seg/dp1')
# (32, 2048, 1, 256)
	
	net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
						bn=True, is_training=is_training, scope='seg/conv2', weight_decay=weight_decay)
# (32, 2048, 1, 256)
	
	net2 = tf_util.dropout(net2, keep_prob=0.8, is_training=is_training, scope='seg/dp2')
# (32, 2048, 1, 256)
	
	net2 = tf_util.conv2d(net2, 128, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
						bn=True, is_training=is_training, scope='seg/conv3', weight_decay=weight_decay)
# (10, 3000, 1, 128)
	# net2=tf.concat(axis=3, values=[net2,out3])
	net2 = tf_util.conv2d(net2,50, [1,1], padding='VALID', stride=[1,1], activation_fn=None, 
						bn=False, scope='seg/conv4', weight_decay=weight_decay)

# (32, 2048, 1, 50)
	
	net2 = tf.reshape(net2, [batch_size, num_point, 50])
	# (10, 3000,50)

	return net2, end_points

def get_loss( seg_pred, seg, end_points):
	# per_instance_label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=l_pred, labels=label)
	# label_loss = tf.reduce_mean(per_instance_label_loss)
	print(seg_pred.get_shape())
	print("hhhh")
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

