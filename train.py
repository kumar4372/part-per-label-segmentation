Fimport argparse
import subprocess
import tensorflow as tf
import numpy as np
from datetime import datetime
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import provider
import pointnet_part_seg as model

# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--epoch', type=int, default=200, help='Epoch to run [default: 50]')
parser.add_argument('--point_num', type=int, default=3000, help='Point Number [256/512/1024/2048]')
parser.add_argument('--output_dir', type=str, default='train_results', help='Directory that stores all training logs and trained models')
parser.add_argument('--wd', type=float, default=0, help='Weight Decay [Default: 0.0]')
FLAGS = parser.parse_args()

data_dir = os.path.join(BASE_DIR, 'data')

# MAIN SCRIPT
point_num = FLAGS.point_num
batch_size = FLAGS.batch
output_dir = FLAGS.output_dir

if not os.path.exists(output_dir):
    os.mkdir(output_dir)


# all_obj_cats_file = os.path.join(hdf5_data_dir, 'all_object_categories.txt')
# fin = open(all_obj_cats_file, 'r')
# lines = [line.rstrip() for line in fin.readlines()]
# all_obj_cats = [(line.split()[0], line.split()[1]) for line in lines]
# fin.close()

# all_cats = json.load(open(os.path.join(hdf5_data_dir, 'overallid_to_catid_partid.json'), 'r'))
NUM_CATEGORIES = 16
# NUM_PART_CATS = len(all_cats)

print('#### Batch Size: {0}'.format(batch_size))
print('#### Point Number: {0}'.format(point_num))
print('#### Training using GPU: {0}'.format(FLAGS.gpu))

DECAY_STEP = 16881 * 20
DECAY_RATE = 0.5

LEARNING_RATE_CLIP = 1e-5

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_CLIP = 0.99

BASE_LEARNING_RATE = 0.001
MOMENTUM = 0.9
TRAINING_EPOCHES = FLAGS.epoch
print('### Training epoch: {0}'.format(TRAINING_EPOCHES))

MODEL_STORAGE_PATH = os.path.join(output_dir, 'trained_models')
if not os.path.exists(MODEL_STORAGE_PATH):
    os.mkdir(MODEL_STORAGE_PATH)

LOG_STORAGE_PATH = os.path.join(output_dir, 'logs')
if not os.path.exists(LOG_STORAGE_PATH):
    os.mkdir(LOG_STORAGE_PATH)

SUMMARIES_FOLDER =  os.path.join(output_dir, 'summaries')
if not os.path.exists(SUMMARIES_FOLDER):
    os.mkdir(SUMMARIES_FOLDER)

def printout(flog, data):
	print(data)
	flog.write(data + '\n')

def placeholder_inputs():
    pointclouds_ph = tf.placeholder(tf.float32, shape=(batch_size, point_num, 3))
    input_label_ph = tf.placeholder(tf.float32, shape=(batch_size, NUM_CATEGORIES))
    # labels_ph = tf.placeholder(tf.int32, shape=(batch_size))
    seg_ph = tf.placeholder(tf.int32, shape=(batch_size, point_num))
    return pointclouds_ph, input_label_ph, seg_ph
    # return seg_ph

def convert_label_to_one_hot(shape,i):
    label_one_hot = np.zeros((shape, NUM_CATEGORIES))
    for idx in range(0,shape):
        label_one_hot[idx, i] = 1
    return label_one_hot

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(FLAGS.gpu)):
            pointclouds_ph,input_label_ph,seg_ph = placeholder_inputs()
            is_training_ph = tf.placeholder(tf.bool, shape=())

            batch = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(
                            BASE_LEARNING_RATE,     # base learning rate
                            batch * batch_size,     # global_var indicating the number of steps
                            DECAY_STEP,             # step size
                            DECAY_RATE,             # decay rate
                            staircase=True          # Stair-case or continuous decreasing
                            )
            learning_rate = tf.maximum(learning_rate, LEARNING_RATE_CLIP)
        
            bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*batch_size,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
            bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)

            lr_op = tf.summary.scalar('learning_rate', learning_rate)
            batch_op = tf.summary.scalar('batch_number', batch)
            bn_decay_op = tf.summary.scalar('bn_decay', bn_decay)
 
            seg_pred, end_points = model.get_model(pointclouds_ph,input_label_ph, \
                    is_training=is_training_ph, bn_decay=bn_decay, cat_num=NUM_CATEGORIES, \
                    batch_size=batch_size, num_point=point_num, weight_decay=FLAGS.wd)
            # seg_pred1, end_points1 = model.get_modeld(pointclouds_ph,input_label_ph, \
                    # is_training=is_training_ph, bn_decay=bn_decay, cat_num=NUM_CATEGORIES, \
                    # batch_size=batch_size, num_point=point_num, weight_decay=FLAGS.wd)
            # seg_pred=tf.add(tf.divide(seg_pred1,2),tf.divide(seg_pred2,2))
            # print(seg_pred.get_shape())
            # model.py defines both classification net and segmentation net, which share the common global feature extractor network.
            # In model.get_loss, we define the total loss to be weighted sum of the classification and segmentation losses.
            # Here, we only train for segmentation network. Thus, we set weight to be 1.0.
            loss, seg_loss, per_instance_seg_loss, per_instance_seg_pred_res  \
                = model.get_loss( seg_pred, seg_ph,  end_points)

            total_training_loss_ph = tf.placeholder(tf.float32, shape=())
            total_testing_loss_ph = tf.placeholder(tf.float32, shape=())

            # label_training_loss_ph = tf.placeholder(tf.float32, shape=())
            # label_testing_loss_ph = tf.placeholder(tf.float32, shape=())

            seg_training_loss_ph = tf.placeholder(tf.float32, shape=())
            seg_testing_loss_ph = tf.placeholder(tf.float32, shape=())

            # label_training_acc_ph = tf.placeholder(tf.float32, shape=())
            # label_testing_acc_ph = tf.placeholder(tf.float32, shape=())
            # label_testing_acc_avg_cat_ph = tf.placeholder(tf.float32, shape=())

            seg_training_acc_ph = tf.placeholder(tf.float32, shape=())
            seg_testing_acc_ph = tf.placeholder(tf.float32, shape=())
            seg_testing_acc_avg_cat_ph = tf.placeholder(tf.float32, shape=())

            total_train_loss_sum_op = tf.summary.scalar('total_training_loss', total_training_loss_ph)
            total_test_loss_sum_op = tf.summary.scalar('total_testing_loss', total_testing_loss_ph)

            # label_train_loss_sum_op = tf.summary.scalar('label_training_loss', label_training_loss_ph)
            # label_test_loss_sum_op = tf.summary.scalar('label_testing_loss', label_testing_loss_ph)

            seg_train_loss_sum_op = tf.summary.scalar('seg_training_loss', seg_training_loss_ph)
            seg_test_loss_sum_op = tf.summary.scalar('seg_testing_loss', seg_testing_loss_ph)

            # label_train_acc_sum_op = tf.summary.scalar('label_training_acc', label_training_acc_ph)
            # label_test_acc_sum_op = tf.summary.scalar('label_testing_acc', label_testing_acc_ph)
            # label_test_acc_avg_cat_op = tf.summary.scalar('label_testing_acc_avg_cat', label_testing_acc_avg_cat_ph)

            seg_train_acc_sum_op = tf.summary.scalar('seg_training_acc', seg_training_acc_ph)
            seg_test_acc_sum_op = tf.summary.scalar('seg_testing_acc', seg_testing_acc_ph)
            seg_test_acc_avg_cat_op = tf.summary.scalar('seg_testing_acc_avg_cat', seg_testing_acc_avg_cat_ph)

            train_variables = tf.trainable_variables()

            trainer = tf.train.AdamOptimizer(learning_rate)
            train_op = trainer.minimize(loss, var_list=train_variables, global_step=batch)

        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        
        init = tf.global_variables_initializer()
        sess.run(init)

        train_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/test')

        # train_file_list = provider.getDataFiles(TRAINING_FILE_LIST)
        num_train_file = 16
        # test_file_list = provider.getDataFiles(TESTING_FILE_LIST)
        num_test_file = 16

        fcmd = open(os.path.join(LOG_STORAGE_PATH, 'cmd.txt'), 'w')
        fcmd.write(str(FLAGS))
        fcmd.close()

        # write logs to the disk
        flog = open(os.path.join(LOG_STORAGE_PATH, 'log.txt'), 'w')

        def train_one_epoch(epoch_num):
            is_training = True
            path_data = os.path.join(data_dir,'train_data')
            path_seg = os.path.join(data_dir,'train_label')

            list_dirs = os.listdir(path_data)
            for i in range(num_train_file):
                
                o=list_dirs[i]
                mypath1 = os.path.join(path_data,o)
                mypath2=os.path.join(path_seg,o)
                cur_data,cur_seg=provider.load(mypath1,mypath2)
                print(i)
                
                cur_labels_one_hot = convert_label_to_one_hot(cur_seg.shape[0],i)

                # printout(flog, 'Loading train file ' + cur_train_filename)
                p=cur_seg.shape[0]

                # cur_data, cur_labels, cur_seg = provider.loadDataFile_with_seg(cur_train_filename)
                cur_data, idx= provider.shuffle_data(cur_data, p)
                cur_seg = cur_seg[idx, ...]

                # cur_labels_one_hot = convert_label_to_one_hot(cur_labels)

                num_data = cur_seg.shape[0]
                num_batch = num_data // batch_size
                print(num_data)
                print(num_batch)

                total_loss = 0.0
                total_label_loss = 0.0
                total_seg_loss = 0.0
                total_label_acc = 0.0
                total_seg_acc = 0.0

                for j in range(num_batch):
                    begidx = j * batch_size
                    endidx = (j + 1) * batch_size

                    feed_dict = {
                            pointclouds_ph: cur_data[begidx: endidx], 
                            input_label_ph: cur_labels_one_hot[begidx: endidx], 
                            seg_ph: cur_seg[begidx: endidx],
                            is_training_ph: is_training, 
                            }

                    _, loss_val,  seg_loss_val, \
                            per_instance_seg_loss_val,  seg_pred_val, pred_seg_res \
                            = sess.run([train_op, loss,seg_loss,  \
                            per_instance_seg_loss, seg_pred, per_instance_seg_pred_res], \
                            feed_dict=feed_dict)

                    per_instance_part_acc = np.mean(pred_seg_res == cur_seg[begidx: endidx], axis=1)
                    average_part_acc = np.mean(per_instance_part_acc)

                    total_loss += loss_val
                    total_seg_loss += seg_loss_val
                    
                    total_seg_acc += average_part_acc

                total_loss = total_loss * 1.0 / num_batch
                total_seg_loss = total_seg_loss * 1.0 / num_batch
                total_seg_acc = total_seg_acc * 1.0 / num_batch

                lr_sum, bn_decay_sum, batch_sum, train_loss_sum,  \
                        train_seg_loss_sum, train_seg_acc_sum = sess.run(\
                        [lr_op, bn_decay_op, batch_op, total_train_loss_sum_op, \
                        seg_train_loss_sum_op, seg_train_acc_sum_op], \
                        feed_dict={total_training_loss_ph: total_loss, \
                        seg_training_loss_ph: total_seg_loss, \
                        seg_training_acc_ph: total_seg_acc})

                train_writer.add_summary(train_loss_sum, i + epoch_num * num_train_file)
                train_writer.add_summary(train_seg_loss_sum, i + epoch_num * num_train_file)
                train_writer.add_summary(lr_sum, i + epoch_num * num_train_file)
                train_writer.add_summary(bn_decay_sum, i + epoch_num * num_train_file)
                train_writer.add_summary(train_seg_acc_sum, i + epoch_num * num_train_file)
                train_writer.add_summary(batch_sum, i + epoch_num * num_train_file)

                printout(flog, '\tTraining Total Mean_loss: %f' % total_loss)
                printout(flog, '\t\tTraining Seg Mean_loss: %f' % total_seg_loss)
                printout(flog, '\t\tTraining Seg Accuracy: %f' % total_seg_acc)

        def eval_one_epoch(epoch_num):
            is_training = False

            total_loss = 0.0
            total_seg_loss = 0.0
            total_seg_acc = 0.0
            total_seen = 0

            total_seg_acc_per_cat = np.zeros((NUM_CATEGORIES)).astype(np.float32)
            total_seen_per_cat = np.zeros((NUM_CATEGORIES)).astype(np.int32)
            path_data = os.path.join(data_dir,'val_data')
            path_seg = os.path.join(data_dir,'val_label')

            list_dirs = os.listdir(path_data)
            for i in range(num_test_file):

               
                o=list_dirs[i]
                mypath1 = os.path.join(path_data,o)
                mypath2=os.path.join(path_seg,o)
                cur_data,cur_seg = provider.load(mypath1,mypath2)
                
                cur_labels_one_hot = convert_label_to_one_hot(cur_seg.shape[0],i)

                num_data = cur_seg.shape[0]
                num_batch = num_data // batch_size

                for j in range(num_batch):
                    begidx = j * batch_size
                    endidx = (j + 1) * batch_size
                    feed_dict = {
                            pointclouds_ph: cur_data[begidx: endidx], 
                            input_label_ph: cur_labels_one_hot[begidx: endidx], 
                            seg_ph: cur_seg[begidx: endidx],
                            is_training_ph: is_training, 
                            }

                    loss_val, seg_loss_val, \
                            per_instance_seg_loss_val,  seg_pred_val, pred_seg_res \
                            = sess.run([loss, seg_loss,\
                            per_instance_seg_loss,  seg_pred, per_instance_seg_pred_res], \
                            feed_dict=feed_dict)



                    per_instance_part_acc = np.mean(pred_seg_res == cur_seg[begidx: endidx], axis=1)
                    average_part_acc = np.mean(per_instance_part_acc)

                    total_seen += 1
                    total_loss += loss_val
                    total_seg_loss += seg_loss_val
                    
                    total_seg_acc += average_part_acc

                    # for shape_idx in range(begidx, endidx):
                    #     total_seen_per_cat[cur_labels[shape_idx]] += 1
                    #     total_label_acc_per_cat[cur_labels[shape_idx]] += np.int32(per_instance_label_pred[shape_idx-begidx] == cur_labels[shape_idx])
                    #     total_seg_acc_per_cat[cur_labels[shape_idx]] += per_instance_part_acc[shape_idx - begidx]

            total_loss = total_loss * 1.0 / total_seen
            total_seg_loss = total_seg_loss * 1.0 / total_seen
            total_seg_acc = total_seg_acc * 1.0 / total_seen

            test_loss_sum,  test_seg_loss_sum, test_seg_acc_sum = sess.run(\
                    [total_test_loss_sum_op, seg_test_loss_sum_op, seg_test_acc_sum_op], \
                    feed_dict={total_testing_loss_ph: total_loss, \
                    seg_testing_loss_ph: total_seg_loss,  seg_testing_acc_ph: total_seg_acc})

            test_writer.add_summary(test_loss_sum, (epoch_num+1) * num_train_file-1)
            test_writer.add_summary(test_seg_loss_sum, (epoch_num+1) * num_train_file-1)
            test_writer.add_summary(test_seg_acc_sum, (epoch_num+1) * num_train_file-1)

            printout(flog, '\tTesting Total Mean_loss: %f' % total_loss)
            printout(flog, '\t\tTesting Seg Mean_loss: %f' % total_seg_loss)
            printout(flog, '\t\tTesting Seg Accuracy: %f' % total_seg_acc)

            # for cat_idx in range(NUM_CATEGORIES):
            #     if total_seen_per_cat[cat_idx] > 0:
            #         printout(flog, '\n\t\tCategory %s Object Number: %d' % (all_obj_cats[cat_idx][0], total_seen_per_cat[cat_idx]))
            #         printout(flog, '\t\tCategory %s Label Accuracy: %f' % (all_obj_cats[cat_idx][0], total_label_acc_per_cat[cat_idx]/total_seen_per_cat[cat_idx]))
            #         printout(flog, '\t\tCategory %s Seg Accuracy: %f' % (all_obj_cats[cat_idx][0], total_seg_acc_per_cat[cat_idx]/total_seen_per_cat[cat_idx]))

        if not os.path.exists(MODEL_STORAGE_PATH):
            os.mkdir(MODEL_STORAGE_PATH)

        for epoch in range(TRAINING_EPOCHES):
            printout(flog, '\n<<< Testing on the test dataset ...')
            eval_one_epoch(epoch)

            printout(flog, '\n>>> Training for the epoch %d/%d ...' % (epoch, TRAINING_EPOCHES))

            # train_file_idx = np.arange(0, len(train_file_list))
            # np.random.shuffle(train_file_idx)

            train_one_epoch(epoch)

            if (epoch+1) % 10 == 0:
                cp_filename = saver.save(sess, os.path.join(MODEL_STORAGE_PATH, 'epoch_' + str(epoch+1)+'.ckpt'))
                printout(flog, 'Successfully store the checkpoint model into ' + cp_filename)

            flog.flush()

        flog.close()

if __name__=='__main__':
    train()
