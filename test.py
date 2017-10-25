import argparse
import tensorflow as tf
import json
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import provider
import pointnet_part_seg as model

DATA_DIR = os.path.join(BASE_DIR, 'data')
path_data = DATA_DIR+'/test_data'
path_label = DATA_DIR+'/test_label'

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='train_results/trained_models/epoch_80.ckpt', help='Model checkpoint path')
FLAGS = parser.parse_args()


# DEFAULT SETTINGS
pretrained_model_path = FLAGS.model_path # os.path.join(BASE_DIR, './pretrained_model/model.ckpt')
print(pretrained_model_path)

gpu_to_use = 0
output_dir = os.path.join(BASE_DIR, './test_results')
output_verbose = True   # If true, output all color-coded part segmentation obj files

# MAIN SCRIPT
point_num = 3000            # the max number of points in the all testing data shapes
batch_size = 1


NUM_OBJ_CATS = 16
NUM_PART_CATS = 50

def placeholder_inputs():
    pointclouds_ph = tf.placeholder(tf.float32, shape=(batch_size, point_num, 3))
    input_label_ph = tf.placeholder(tf.float32, shape=(batch_size, NUM_OBJ_CATS))
    return pointclouds_ph, input_label_ph

num_test_file = 16
list_dirs = os.listdir(path_data)

def predict():
    is_training = False
    
    with tf.device('/gpu:'+str(gpu_to_use)):
        pointclouds_ph,input_label_ph = placeholder_inputs()
        is_training_ph = tf.placeholder(tf.bool, shape=())

        # simple model
        seg_pred, end_points = model.get_model(pointclouds_ph, input_label_ph, \
                cat_num=NUM_OBJ_CATS, is_training=is_training_ph, \
                batch_size=batch_size, num_point=point_num, weight_decay=0.0, bn_decay=None)
        
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)


        # Restore variables from disk.
        print ('Loading model %s' % pretrained_model_path)
        saver.restore(sess, pretrained_model_path)
        print( 'Model restored.')
        
        for i in range(num_test_file):
            o=list_dirs[i]
            mypath1 = os.path.join(path_data,o)
            for (dirpath, dirnames, filenames) in os.walk(mypath1):
                for item in filenames:
                    mypath1 = mypath1 + '/item'
                    cur_data = provider.load_data(mypath1)
                    print(i)
                    ori_point_num = cur_data.shape[0]

                    seg_pred_res = sess.run([seg_pred], feed_dict={
                                pointclouds_ph: batch_data,
                                input_label_ph: cur_label_one_hot, 
                                is_training_ph: is_training,
                            })
                    
                    # seg_pred_res = seg_pred_res[0, ...]

                    # iou_oids = object2setofoid[objcats[cur_gt_label]]
                    # non_cat_labels = list(set(np.arange(NUM_PART_CATS)).difference(set(iou_oids)))

                    # mini = np.min(seg_pred_res)
                    # seg_pred_res[:, non_cat_labels] = mini - 1000

                    seg_pred_val = np.argmax(seg_pred_res, axis=1)[:ori_point_num]
                    print(seg_pred_val)
                    break
                # break
            break


                
with tf.Graph().as_default():
    predict()
