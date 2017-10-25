import os
import sys
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

if not os.path.exists(os.path.join(DATA_DIR, 'train_data')):
    www = 'https://shapenet.cs.stanford.edu/iccv17/partseg/train_data.zip'
    zipfile = os.path.basename(www)
    os.system('wget %s; unzip %s' % (www, zipfile))
    os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    os.system('rm %s' % (zipfile))

if not os.path.exists(os.path.join(DATA_DIR, 'train_label')):
    www = 'https://shapenet.cs.stanford.edu/iccv17/partseg/train_label.zip'
    zipfile = os.path.basename(www)
    os.system('wget %s; unzip %s' % (www, zipfile))
    os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    os.system('rm %s' % (zipfile))

if not os.path.exists(os.path.join(DATA_DIR, 'val_data')):
    www = 'https://shapenet.cs.stanford.edu/iccv17/partseg/val_data.zip'
    zipfile = os.path.basename(www)
    os.system('wget %s; unzip %s' % (www, zipfile))
    os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    os.system('rm %s' % (zipfile))

if not os.path.exists(os.path.join(DATA_DIR, 'val_label')):
    www = 'https://shapenet.cs.stanford.edu/iccv17/partseg/val_label.zip'
    zipfile = os.path.basename(www)
    os.system('wget %s; unzip %s' % (www, zipfile))
    os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    os.system('rm %s' % (zipfile))

if not os.path.exists(os.path.join(DATA_DIR, 'test_data')):
    www = 'https://shapenet.cs.stanford.edu/iccv17/partseg/test_data.zip'
    zipfile = os.path.basename(www)
    os.system('wget %s; unzip %s' % (www, zipfile))
    os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    os.system('rm %s' % (zipfile))

path_data = DATA_DIR+'/train_data'
path_label = DATA_DIR+'/train_label'
list_dirs = os.listdir(path_data)

num_points = 3000
def load_data(filepath):
    data = np.zeros((num_points,3))
    i = 0
    with open(filepath,'r') as f:
        while (i<num_points):
            a = f.readline()
            if (a==''):
                break
            else:
                a = a.split()
                b = []
                for item in a:
                    b.append(float(item))
                data[i] = b
                i = i+1
    f.close()
    return data

def load_label(filepath):
    data = np.zeros((num_points))
    i = 0
    with open(filepath,'r') as f:
        while (i<num_points):
            a = f.readline()
            if (a==''):
                break
            else:
                data[i] = int(a)
            i = i+1
    f.close()
    return data

def load(mypath1,mypath2):
    # we are considering here path of train data, label etc.
    # for n = 1, path will be 0 to 32.
    # dic = {}
    for (dirpath, dirnames, filenames) in os.walk(mypath1):
        # print(filenames)
        # dic[filenames] = path_label+'/'+o;
        data = np.zeros((len(filenames),num_points,3))
        label = np.zeros((len(filenames),num_points))
        for k in range(len(filenames)):
            path1 = os.path.join(mypath1,filenames[k])
            # print(path1)
            name = filenames[k][0:-3]+'seg'
            path2 = os.path.join(mypath2,name)
            data[k] = load_data(path1)
            label[k] = load_label(path2)
            # dic[path1] = path2
    # print(dic)
    return data,label

# train_nthbatch(32,17)    
# path1 = '/home/gautam/Desktop/project/shapenet challenge/pointnet-master/part_seg/data/train_data/02691156'
# path2 = '/home/gautam/Desktop/project/shapenet challenge/pointnet-master/part_seg/data/train_label/02691156'
# data,label = load(path1,path2)
# print(data[0])
# print(label.shape)

def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(labels)
    np.random.shuffle(idx)
    return data[idx, ...], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)
