# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import functions_multi as func
import matplotlib.pyplot as plt
import scipy.io as sio
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, Conv3D, MaxPooling3D, ZeroPadding3D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, Input
from keras.utils.np_utils import to_categorical
from sklearn.decomposition import PCA
from keras.optimizers import Adam, SGD, Adadelta, RMSprop, Nadam
import keras.callbacks as kcallbacks
from keras.regularizers import l2
import time
from Utils import zeroPadding, normalization, doPCA, modelStatsRecord, averageAccuracy, ssrn_SS_UP

import collections
from sklearn import metrics, preprocessing
import argparse
import os

import Normalization as NL

#from CPL_NET import res4_model_ss
from FIX_RES_NET_ATTENTION import res4_model_ss

tf.reset_default_graph()

def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca

def indexToAssignment(Row_index, Col_index, pad_length):
    new_assign = {}
    for counter in range(Row_index.shape[0]):
        assign_0 = Row_index[counter] + pad_length
        assign_1 = Col_index[counter] + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len,pos_row+ex_len+1), :]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch

def do_eval(sess, eval_correct, images, labels, test_x, test_y):
    true_count = 0.0
    total_true_count=0.0
    pred_labels = []
    test_num = test_y.shape[0]
    batch_size = FLAGS.batch_size
    batch_num = test_num // batch_size if test_num % batch_size == 0 else test_num // batch_size + 1

    for i in range(batch_num):
        batch_x = test_x[i*batch_size:(i+1)*batch_size]
        batch_y = test_y[i*batch_size:(i+1)*batch_size]
        test_prediction, true_count = sess.run(eval_correct, feed_dict={images:batch_x, labels:batch_y})
        test_prediction.ravel()
#        test_prediction = int(test_prediction)
#        test_prediction=test_prediction.tolist();
        total_true_count =  total_true_count +true_count
        pred_labels=np.concatenate((pred_labels, test_prediction),axis=0)       
    return pred_labels, total_true_count / test_num

#gt_IN = set_zeros(gt_IN, [1,4,7,9,13,15,16])
numComponents=10
PATCH_LENGTH = 5
                  
window_size = PATCH_LENGTH*2+1           #27, 27

epoch_num = 50
ITER = 1
# 10%:10%:80% data for training, validation and testing
#TRAINING_SPLIT = 0.95                     # 10% for training and %90 for validation and testing

def run_training():

# load the data
    print (150*'*')
    uPavia = sio.loadmat('/home/amax/xibobo/HybridSN-sample/data/UP/PaviaU.mat')
    gt_uPavia = sio.loadmat('/home/amax/xibobo/HybridSN-sample/data/UP/PaviaU_combinedGt.mat')
    data_IN = uPavia['paviaU']
    gt_IN = gt_uPavia['combinedGt']
#    uPavia = sio.loadmat('/home/amax/xibobo/SSRN-master/datasets/IN/Indian_pines_corrected.mat')
#    gt_uPavia = sio.loadmat('/home/amax/xibobo/SSRN-master/datasets/IN/Indian_pines_gt.mat')
#    data_IN = uPavia['indian_pines_corrected']
#    gt_IN = gt_uPavia['indian_pines_gt']

    print (data_IN.shape)
    data = data_IN.reshape(np.prod(data_IN.shape[:2]),np.prod(data_IN.shape[2:]))
    gt = gt_IN.reshape(np.prod(gt_IN.shape[:2]),)
    
    #[2]:
#    trainingIndexf = '/home/amax/xibobo/HybridSN-sample/data/DisjointUPtrainingIndexRandomTrain20.mat'
#    train_indices = sio.loadmat(trainingIndexf)['trainingIndexRandomtrain']
#    train_indices_rows = sio.loadmat(trainingIndexf)['trainingIndexRandomtrain_rows']
#    train_indices_cols = sio.loadmat(trainingIndexf)['trainingIndexRandomtrain_cols']
#    
#    testingIndexf = '/home/amax/xibobo/HybridSN-sample/data/DisjointUPtestingIndexMix.mat'
#    test_indices = sio.loadmat(testingIndexf)['testingIndexMix']  
#    test_indices_rows = sio.loadmat(testingIndexf)['testingIndexMix_rows']  
#    test_indices_cols = sio.loadmat(testingIndexf)['testingIndexMix_cols'] 
  
    trainingIndexf = '/home/amax/xibobo/HybridSN-sample/data/DisjointUPtrainingIndexMix.mat'
    train_indices = sio.loadmat(trainingIndexf)['trainingIndexMix']
    train_indices_rows = sio.loadmat(trainingIndexf)['trainingIndexMix_rows']
    train_indices_cols = sio.loadmat(trainingIndexf)['trainingIndexMix_cols']
    testingIndexf = '/home/amax/xibobo/HybridSN-sample/data/DisjointUPtestingIndexMix.mat'
    test_indices = sio.loadmat(testingIndexf)['testingIndexMix']  
    test_indices_rows = sio.loadmat(testingIndexf)['testingIndexMix_rows']  
    test_indices_cols = sio.loadmat(testingIndexf)['testingIndexMix_cols'] 
    
    train_indices = np.squeeze(train_indices-1)
    test_indices = np.squeeze(test_indices-1)
    height = gt_IN.shape[0]
    width = gt_IN.shape[1]   
    Y=gt_IN.T
    Y = Y.reshape(height*width,)
    
    train_y = Y[train_indices]-1
    y_train = to_categorical(np.asarray(train_y))
    
    
    test_y = Y[test_indices] - 1
    y_test = to_categorical(np.asarray(test_y))
    TRAIN_SIZE = train_indices.shape[0]
    TEST_SIZE = test_indices.shape[0]
    classes_num = np.max(gt)
    
    data = preprocessing.scale(data)
    whole_data = data.reshape(data_IN.shape[0], data_IN.shape[1], data_IN.shape[2])
#    whole_data = NL.NormalizationEachBand(data_IN, unit=False)
    
    # scaler = preprocessing.MaxAbsScaler()
    # data = scaler.fit_transform(data)
    
#    whole_data = data.reshape(data_IN.shape[0], data_IN.shape[1], data_IN.shape[2])
    whole_data,pca = applyPCA(whole_data, numComponents = numComponents)
    img_channels= whole_data.shape[2]
#    temp_data = whole_data.reshape(whole_data.shape[0]*whole_data.shape[1], whole_data.shape[2])
#    temp_data = preprocessing.scale(temp_data)
#    whole_data = temp_data.reshape(whole_data.shape[0], whole_data.shape[1], whole_data.shape[2])
    
    
    padded_data = zeroPadding.zeroPadding_3D(whole_data, PATCH_LENGTH)
    
    train_data = np.zeros((TRAIN_SIZE, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, img_channels))
    test_data = np.zeros((TEST_SIZE, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, img_channels))
    
    train_assign = indexToAssignment(np.squeeze(train_indices_rows-1), np.squeeze(train_indices_cols-1), PATCH_LENGTH)
    for i in range(len(train_assign)):
        train_data[i] = selectNeighboringPatch(padded_data,train_assign[i][0],train_assign[i][1],PATCH_LENGTH)
    #
    test_assign = indexToAssignment(np.squeeze(test_indices_rows-1), np.squeeze(test_indices_cols-1), PATCH_LENGTH)
    for i in range(len(test_assign)):
        test_data[i] = selectNeighboringPatch(padded_data,test_assign[i][0],test_assign[i][1],PATCH_LENGTH)
    
    Xtrain = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2],img_channels)
    Xtest = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], img_channels)
    train_x = Xtrain.reshape(-1,window_size,window_size,img_channels,1)   
    test_x = Xtest.reshape(-1, window_size,window_size,img_channels,1)      
    train_num = train_x.shape[0]
    test_num = test_x.shape[0]

    # construct the computation graph
    images = tf.placeholder(tf.float32, shape=[None,window_size,window_size,img_channels,1])
    labels = tf.placeholder(tf.int32, shape=[None])
    lr= tf.placeholder(tf.float32)

    features,_ = res4_model_ss(images,classes_num,[1],[1])
    centers = func.construct_center(features, classes_num, 1)
    
    loss1 = func.dce_loss(features, labels, centers, FLAGS.temp)
#    loss2 = func.mcl_loss(features, labels, centers, 0.9)
#    loss2 = func.gmcl_loss(features, labels, centers, 0.9)
    loss2 = func.pl_loss(features, labels, centers)

#    loss=loss1
    loss = loss1 + FLAGS.weight_pl * loss2
#    
    eval_correct = func.evaluation(features, labels, centers)
    train_op = func.training(loss, lr)
    
    #counts = tf.get_variable('counts', [FLAGS.classes_num], dtype=tf.int32,
    #    initializer=tf.constant_initializer(0), trainable=False)
    #add_op, count_op, average_op = net.init_centers(features, labels, centers, counts)
    init = tf.global_variables_initializer()
    # initialize the variables
    sess = tf.Session()
    sess.run(init)
    #compute_centers(sess, add_op, count_op, average_op, images, labels, train_x, train_y)

    # run the computation graph (train and test process)
    epoch = 1
    index = list(range(train_num))
    np.random.shuffle(index)
    batch_size = FLAGS.batch_size
    batch_num = train_num//batch_size if train_num % batch_size==0 else train_num//batch_size+1
    #saver = tf.train.Saver(max_to_keep=1)
    train_start= time.time()
    # train the framework with the training data
#    while stopping<FLAGS.stop:
    while epoch<epoch_num:
        time1 = time.time()
        loss_now = 0.0
        score_now = 0.0
           
        for i in range(batch_num):
            batch_x = train_x[index[i*batch_size:(i+1)*batch_size]]
            batch_y = train_y[index[i*batch_size:(i+1)*batch_size]]
            result = sess.run([train_op, loss, eval_correct], feed_dict={images:batch_x,
                labels:batch_y, lr:FLAGS.learning_rate})
#            init_logits_value = sess.run(logits)
            loss_now += result[1]
            score_now += result[2][1]
        score_now /= train_num

        print ('epoch {}: training: loss --> {:.3f}, acc --> {:.3f}%'.format(epoch, loss_now, score_now*100))
        #print sess.run(centers)

        #checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
        #saver.save(sess, checkpoint_file, global_step=epoch)
#        FLAGS.learning_rate*=FLAGS.decay
        FLAGS.learning_rate-=FLAGS.decay
#        FLAGS.learning_rate*= (1. / (1. + FLAGS.decay * epoch))
        epoch += 1
        np.random.shuffle(index)

        time2 = time.time()
        print ('time for this epoch: {:.3f} minutes'.format((time2-time1)/60.0))
    print()
    print('time for the whole training phase: '+str(time.time()-train_start)+' s')   
    # test the framework with the test data
    init_centers_value = sess.run(centers)
    test_start= time.time()
    pred_labels, test_score = do_eval(sess, eval_correct, images, labels, test_x, test_y)
    print('time for the whole testing phase: '+str(time.time()-test_start)+' s')
    sess.close()    
    pred_labels = np.int8(pred_labels)  
    test_y = np.int8(test_y) 
#    confusion matrix
    matrix = np.zeros((classes_num, classes_num))
    with open('prediction_DPN_HRA.txt', 'w') as f:
        for i in range(test_num):
            pre_label = pred_labels[i]
            f.write(str(pre_label)+'\n')
            matrix[pre_label, test_y[i]] += 1
    f.closed  
    print()
    print('The confusion matrix is:')
    print(np.int_(matrix))
#     overall accuracy    
    OA=np.sum(np.trace(matrix)) / float(test_num)
#    print('OA = '+str(OA)+'\n')
    
#     average accuracy
#    print('ua =')
    ua = np.diag(matrix)/np.sum(matrix, axis=1)
    
#     precision
#    print('precision =')
    precision = np.diag(matrix)/np.sum(matrix, axis=0)  
#     Kappa
    matrix = np.mat(matrix);
    Po = OA;
    xsum = np.sum(matrix, axis=1);
    ysum = np.sum(matrix,axis = 0);
    Pe = float(ysum*xsum)/(np.sum(matrix)**2);
    Kappa = float((Po-Pe)/(1-Pe));
    
    for i in ua:
         print(i)
    print(str(np.sum(ua)/matrix.shape[0]))
    print(str(OA))
    print(str(Kappa));
    print()
    for i in precision:
         print(i)  
    print(str(np.sum(precision)/matrix.shape[0]))
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    #parser.add_argument('--log_dir', type=str, default='data/', help='directory to save the data')
    parser.add_argument('--stop', type=int, default=5, help='stopping number')
    parser.add_argument('--decay', type=float, default=1e-6, help='the value to decay the learning rate')
    parser.add_argument('--temp', type=float, default=1, help='the temperature used for calculating the loss')
    parser.add_argument('--weight_pl', type=float, default=0.001, help='the weight for the prototype loss (PL)')
    parser.add_argument('--gpu', type=int, default=1, help='the gpu id for use')

    FLAGS, unparsed = parser.parse_known_args()
    print (150*'*')
    print( 'Configuration of the training:')
    print ('learning rate:', FLAGS.learning_rate)
    print ('batch size:', FLAGS.batch_size)
    print( 'stopping:', FLAGS.stop)
    print( 'learning rate decay:', FLAGS.decay)
    print( 'value of the temperature:', FLAGS.temp)
    print ('prototype loss weight:', FLAGS.weight_pl)
    print( 'GPU id:', FLAGS.gpu)
    #print 'path to save the model:', FLAGS.log_dir

    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

    run_training()    
    
    

