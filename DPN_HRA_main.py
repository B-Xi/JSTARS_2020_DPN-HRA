# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import functions as func
import scipy.io as sio
from sklearn.decomposition import PCA
import time
from Utils import zeroPadding
from sklearn import preprocessing
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from DPN_HRA_MODEL import res4_model_ss

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
        total_true_count = total_true_count +true_count
        pred_labels=np.concatenate((pred_labels, test_prediction),axis=0)       
    return pred_labels, total_true_count / test_num

def run_training():
# load the data
    print (150*'*')
    HU2012 = sio.loadmat('./data/HU2012/2012_Houston.mat')
    data_IN = HU2012['spectraldata']
    gt_IN = HU2012['gt_2012']
    print (data_IN.shape)
    data = data_IN.reshape(np.prod(data_IN.shape[:2]),np.prod(data_IN.shape[2:]))
    gt = gt_IN.reshape(np.prod(gt_IN.shape[:2]),)

    trainingIndexf = './data/Houston2012trainingIndex.mat'
    train_indices = sio.loadmat(trainingIndexf)['trainingIndex']
    train_indices_rows = sio.loadmat(trainingIndexf)['trainingIndex_rows']
    train_indices_cols = sio.loadmat(trainingIndexf)['trainingIndex_cols']
    testingIndexf = './data/Houston2012testingIndex.mat'
    test_indices = sio.loadmat(testingIndexf)['testingIndex']  
    test_indices_rows = sio.loadmat(testingIndexf)['testingIndex_rows']  
    test_indices_cols = sio.loadmat(testingIndexf)['testingIndex_cols'] 

    train_indices = np.squeeze(train_indices-1)
    test_indices = np.squeeze(test_indices-1)
    height, width = gt_IN.shape

    Y=gt_IN.T
    Y = Y.reshape(height*width,)
    train_y = Y[train_indices]-1
    test_y = Y[test_indices] - 1

    classes_num = np.max(gt)
    
    data = preprocessing.scale(data)
    whole_data = data.reshape(data_IN.shape[0], data_IN.shape[1], data_IN.shape[2])

    whole_data, pca = applyPCA(whole_data, numComponents = FLAGS.numComponents)
    img_channels = whole_data.shape[2]
    PATCH_LENGTH = int((FLAGS.window_size-1)/2)
    padded_data = zeroPadding.zeroPadding_3D(whole_data, PATCH_LENGTH)
    train_data = np.zeros((train_indices.shape[0], FLAGS.window_size, FLAGS.window_size, img_channels))
    test_data = np.zeros((test_indices.shape[0], FLAGS.window_size, FLAGS.window_size, img_channels))
    
    train_assign = indexToAssignment(np.squeeze(train_indices_rows-1), np.squeeze(train_indices_cols-1), PATCH_LENGTH)
    for i in range(len(train_assign)):
        train_data[i] = selectNeighboringPatch(padded_data,train_assign[i][0],train_assign[i][1],PATCH_LENGTH)

    test_assign = indexToAssignment(np.squeeze(test_indices_rows-1), np.squeeze(test_indices_cols-1), PATCH_LENGTH)
    for i in range(len(test_assign)):
        test_data[i] = selectNeighboringPatch(padded_data,test_assign[i][0],test_assign[i][1],PATCH_LENGTH)
    
    Xtrain = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2],img_channels)
    Xtest = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], img_channels)
    train_x = Xtrain.reshape(-1,train_data.shape[1], train_data.shape[2],img_channels,1)
    test_x = Xtest.reshape(-1, test_data.shape[1], test_data.shape[2],img_channels,1)
    train_num = train_x.shape[0]
    test_num = test_x.shape[0]

    # construct the computation graph
    images = tf.placeholder(tf.float32, shape=[None,FLAGS.window_size,FLAGS.window_size,img_channels,1])
    labels = tf.placeholder(tf.int32, shape=[None])
    lr= tf.placeholder(tf.float32)

    features = res4_model_ss(images,[1],[1])
    prototypes = func.construct_center(features, classes_num, 1)
    
    loss1 = func.dce_loss(features, labels, prototypes, FLAGS.temp)
    loss2 = func.dis_loss(features, labels, prototypes)
    loss = loss1 + FLAGS.weight_dis * loss2

    eval_correct = func.evaluation(features, labels, prototypes)
    train_op = func.training(loss, lr)

    # initialize the variables
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # run the computation graph (train and test process)
    epoch = 1
    index = list(range(train_num))
    np.random.shuffle(index)
    batch_size = FLAGS.batch_size
    batch_num = train_num//batch_size if train_num % batch_size ==0 else train_num//batch_size+1
    train_start= time.time()

    # train the framework with the training data
    while epoch<FLAGS.epoch_num:
        time1 = time.time()
        loss_now = 0.0
        score_now = 0.0
        for i in range(batch_num):
            batch_x = train_x[index[i*batch_size:(i+1)*batch_size]]
            batch_y = train_y[index[i*batch_size:(i+1)*batch_size]]
            result = sess.run([train_op, loss, eval_correct], feed_dict={images:batch_x,
                labels:batch_y, lr:FLAGS.learning_rate})
            loss_now += result[1]
            score_now += result[2][1]
        score_now /= train_num
        print ('epoch {}: training: loss --> {:.3f}, acc --> {:.3f}%'.format(epoch, loss_now, score_now*100))
        FLAGS.learning_rate-=FLAGS.decay
        epoch += 1
        np.random.shuffle(index)
        time2 = time.time()
        print ('time for this epoch: {:.3f} minutes'.format((time2-time1)/60.0))
    print()
    print('time for the whole training phase: '+str(time.time()-train_start)+' s')   
    # test the framework with the test data
    # init_prototypes_value = sess.run(prototypes) # get the variable of prototypes
    test_start= time.time()
    pred_labels, test_score = do_eval(sess, eval_correct, images, labels, test_x, test_y)
    print('time for the whole testing phase: '+str(time.time()-test_start)+' s')
    sess.close()    
    pred_labels = np.int8(pred_labels)  
    test_y = np.int8(test_y) 
#    confusion matrix
    matrix = np.zeros((classes_num, classes_num))
    with open('prediction.txt', 'w') as f:
        for i in range(test_num):
            pre_label = pred_labels[i]
            f.write(str(pre_label)+'\n')
            matrix[pre_label, test_y[i]] += 1
    f.closed  
    print()
    print('The confusion matrix is:')
    print(np.int_(matrix))

#   calculate the overall accuracy
    OA = np.sum(np.trace(matrix)) / float(test_num)
#    print('OA = '+str(OA)+'\n')
#   calculate the per-class accuracy
#    print('ua =')
    ua = np.diag(matrix)/np.sum(matrix, axis=0)
#   calculate the precision
#    print('precision =')
    precision = np.diag(matrix)/np.sum(matrix, axis=1)
#   calculate the Kappa coefficient
    matrix = np.mat(matrix)
    Po = OA
    xsum = np.sum(matrix, axis=1)
    ysum = np.sum(matrix, axis=0)
    Pe = float(ysum*xsum)/(np.sum(matrix)**2)
    Kappa = float((Po-Pe)/(1-Pe))
    ## print the classification result
    for i in ua:
         print(i)
    print(str(np.sum(ua)/matrix.shape[0]))
    print(str(OA))
    print(str(Kappa))
    print()
    for i in precision:
         print(i)  
    print(str(np.sum(precision)/matrix.shape[0]))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--decay', type=float, default=1e-6, help='the value to decay the learning rate')
    parser.add_argument('--temp', type=float, default=1, help='the temperature used for calculating the loss')
    parser.add_argument('--weight_dis', type=float, default=0.01, help='the weight for the discriminative loss (DIS)')
    parser.add_argument('--numComponents', type=int, default=30, help='the number of the principal components')
    parser.add_argument('--window_size', type=int, default=11, help='the window size of the 3D samples')
    parser.add_argument('--epoch_num', type=int, default=50, help='epoch number of the iterations')

    FLAGS, unparsed = parser.parse_known_args()
    print (150*'*')
    print('Configuration of the training:')
    print('learning rate:', FLAGS.learning_rate)
    print('batch size:', FLAGS.batch_size)
    print('learning rate decay:', FLAGS.decay)
    print('value of the temperature:', FLAGS.temp)
    print('discriminative loss weight:', FLAGS.weight_dis)
    print('number of the principal components:', FLAGS.numComponents)
    print('window size of the 3D samples:', FLAGS.window_size)
    print('epoch number of the iterations:', FLAGS.epoch_num)

    run_training()


