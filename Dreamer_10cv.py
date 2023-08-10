#%%
'''
10cv benchmark for the subject dependent experiment
'''
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from scipy.io import loadmat
from utils import Embed_DREAMER
from Models import MSBAM, EEGNet, HST_model, KAMNet
from Probability_helper import *
from Losses import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import zscore
from scipy.stats import zscore
from keras.utils import to_categorical
from utils import sample_generator, val_gen, data_split_10, label_encoding, label_noise, make_segs
from sklearn.metrics import accuracy_score, confusion_matrix, top_k_accuracy_score, f1_score
from Metrics import count_off_neibor, adj_score
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# gpus = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(gpus[1], 'GPU')
# %%
'''
Preprocess and generator

with 1-s randomly cropped  signals
'''
def label_smoothing(Y):
    # onehot = to_categorical(Y)
    # assert len(x.shape)==2
    # grid = np.arange(len( np.unique(Y))) # Y label has to consective
    grid = np.arange( 5 )
    spread = 0.5
    temp = np.exp(-0.5*(grid[None,:]-Y[:,None])**2/spread**2)
    return temp/np.sum(temp,axis=1, keepdims=True)

def to_binary(Y):
    temp = np.where(Y>1.5, 1, 0)
    return temp

def onehot(Y):
    return to_categorical(Y, 5)


def Embed_X(X):
    return Embed_DREAMER(X.transpose(0,2,1))[...,None]


#%%
'''
https://ieeexplore.ieee.org/document/7887697
Dreamer:
   23 participants,
   18 videos
   128Hz sampling rate
   with baseline 
   unbalanced
   scores runs from 1-5
'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--Subject', help='subject index')
parser.add_argument('--ModelChoice', help='choice of classificatoin model')
parser.add_argument('--LabelMode', help='choice of labels for training')
parser.add_argument('--ValMode', default='random', help='mode of validation, random or fix')
parser.add_argument('--LabelChoice', help='choice of label: V, A or D')

args = parser.parse_args()

Subject = int(args.Subject)
Model_Choice = args.ModelChoice # 'MSBAM', 'EEGNet'
Label_Code = args.LabelChoice # 'V', 'A', 'D' 
LabelMode = args.LabelMode # Mixture, Gaussian, Regression OneHot_b or OneHot
val_mode = args.ValMode

# Model_Choice = 'EEGNet' # 'MSBAM', 'EEGNet'
# Label_Code = 'V' # 'V', 'A', 'D' 
# LabelMode = 'OneHot'  # Mixture, Gaussian, Regression OneHot_b or OneHot
# val_mode = 'random'

discrete_mixture = False
seg_len = 128
num_channels = 14
num_sessions = 18
num_subjects = 23
Total_cls = 5
batchsize = 256
val_batchsize = 2000
seg_stride = seg_len
test_batchsize = 1000
Epochs = 50
lr = 1e-3

X = loadmat('/mnt/HDD/Datasets/DREAMER/S{:02d}_1min.mat'.format(Subject))['X'].transpose((1,0,2))
Y = loadmat('/mnt/HDD/Datasets/DREAMER/Labels.mat')['Y'][Subject-1] #careful here for the indexing

'''
Some subject may not have ALL the scores, for example S02, Arousal
'''
if Label_Code == 'V':
    Label = Y[:,0] - 1  #range [0, 1, 2, 3, 4]
elif Label_Code == 'A': 
    Label = Y[:,1] - 1  #range [0, 1, 2, 3, 4]
elif Label_Code == 'D':
    label = Y[:,2] - 1  #range [0, 1, 2, 3, 4]
else:
    raise('Not valid Label!')

# train_win = (0, 128*48)
# # test_win = (128*54, 128*60)
# test_win = (128*48, 128*54)
# # val_win = (128*48, 128*54)
# val_win = (128*54, 128*60)
# test_win = (6000, data.shape[1])
# Xtest = make_segs(X[:,40*128:,:], seg_len=seg_len, stride=seg_stride)  #Xtest may not be balanced this way

# X_prep = lambda x: zscore(x, axis=1)
X_prep = lambda x: x/np.max(np.abs(x), axis=1, keepdims=True)
# X_prep = lambda x: -1+2*(x - np.min(x, axis=1, keepdims=True) )/ (np.max(x, axis=1, keepdims=True) - np.min(x, axis=1, keepdims=True) )
# X_prep = lambda x: (x - np.min(x, axis=1, keepdims=True) )/ (np.max(x, axis=1, keepdims=True) - np.min(x, axis=1, keepdims=True) )
X = X_prep(X)

X_trans = Embed_X if Model_Choice == 'MSBAM' else None

#%%
'''
Create Model
'''
#%%
if LabelMode == 'Mixture':
    # prob_model = lambda x: mixture_logistic(x, discrete_mixture, 3)
    prob_model = lambda x: mixture_gaussian(x, discrete_mixture, 3)
elif LabelMode == 'Gaussian':
    prob_model = lambda x: Single_Gaussian(x, discrete_mixture)
elif LabelMode == 'Gaussian_2':
    prob_model = lambda x: Two_Gaussians(x)
else:
    prob_model = None

if Model_Choice == 'EEGNet':
    model = EEGNet(prob_model, nb_classes = 5, LabelMode=LabelMode, activation='elu',
                    Chans = num_channels, Samples = seg_len,
                    dropoutRate = 0.75, kernLength = 5, F1 = 8, 
                    D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout')
elif Model_Choice == 'KAMNet':
    model = KAMNet(prob_model, nb_classes = 5, LabelMode=LabelMode, activation='elu',
                    Chans = num_channels, Samples = seg_len,
                    dropoutRate = 0.75, kernLength = 5, F1 = 8, 
                    D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout')
elif Model_Choice == 'MSBAM':
    model = MSBAM(input_shape = (9,9,128,1), LabelMode=LabelMode,
                Nclass = 5, Prob_model = prob_model, Droprate=0.75)
elif Model_Choice == 'HST':
    def create_adj():
        A1 = np.zeros((4,4))
        A2 = np.zeros((3,3))
        # A3 = np.zeros(3,3)
        # A4 = np.zeros(4,4)
        A_R = np.zeros((4,4))

        A1[0,2] = 1.0
        A1[2,0] = 1.0
        A1[1,2] = 1.0
        A1[2,1] = 1.0
        A1[1,3] = 1.0
        A1[3,1] = 1.0
        A1[2,3] = 1.0
        A1[3,2] = 1.0

        A2[0,1] = 1.0
        A2[1,0] = 1.0
        A2[1,2] = 1.0
        A2[2,1] = 1.0

        A3 = A2.copy()
        A4 = A1[::-1,::-1]

        A_R[0,1] = 1.0
        A_R[1,0] = 1.0
        A_R[1,2] = 1.0
        A_R[2,1] = 1.0
        A_R[2,3] = 1.0
        A_R[3,2] = 1.0    
        A_R[0,3] = 1.0
        A_R[3,0] = 1.0

        return [A1, A2, A3, A4], A_R

    adjN, adjR = create_adj()
    model = HST_model(input_shape = (128,14), LabelMode=LabelMode, 
                      nb_classes=5, Prob_model=prob_model, mergeMode='dense',
                      loc=[0,4,7,10,14], adj_list=adjN, 
                      hop_list=[2,2,2,2], 
                      region_adjM=adjR, 
                      region_hop=2, poly_type='chebyshev',
                      activation='elu', droprate=0.5, F1=8, ksize=64)

if LabelMode in ['Mixture', 'Gaussian_2']:
    model.compile( 
                loss=NLL_loss,
                # loss=hybrid_loss,
                #   metrics=['mean_absolute_error'],
                optimizer=Adam(lr )
    )
elif LabelMode == 'Gaussian':
    model.compile( 
                # loss = KL_loss,
                # loss = hybrid_loss,
                loss = NLL_loss,
                #   metrics=['mean_absolute_error'],
                optimizer=Adam(lr )
    )
elif LabelMode in ['OneHot', 'OneHot_N', 'OneHot_b']:
    model.compile( 
                loss='categorical_crossentropy',
                metrics=['accuracy'],
                optimizer=Adam(lr )
    )
elif LabelMode == 'Regression':
    model.compile( 
                loss='mean_absolute_error',
                # loss = Noise_loss(),
                #   metrics=['mean_absolute_error'],
                optimizer=Adam(lr )
    )
model.summary()
model.save_weights('/mnt/HDD/Benchmarks/DREAMER/ini_weights.h5')

ckpt_path = '/mnt/HDD/Benchmarks/DREAMER/ckpt/DREAMER_S{:02d}_{}_{}_{}'.format(Subject, Model_Choice, Label_Code, LabelMode)
ckpt = ModelCheckpoint(filepath=ckpt_path,
                        save_weights_only=True,
                        monitor='val_loss',
                        mode='min',
                        save_best_only=True)
def scheduler(epoch, lr):
    if epoch < 30:
      return lr
    else:
    #   return lr * tf.math.exp(-0.1)
      return 5e-4
lrs = LearningRateScheduler(scheduler)
#%%
summary = {
    'top_1_acc': [],
    'top_2_acc': [],
    'F1':[],
    'CM':[],
    'off_neibor':[],
    'closeness':[]
}

for i in range(10):
    keras.backend.clear_session()
    train_win, val_win, test_win = data_split_10(i, X.shape[1])

    if LabelMode in ['Mixture', 'Gaussian', 'Gaussian_2']:
        # train_gen = sample_generator(X, Label, X_trans, None, seg_len, train_win, batchsize=batchsize)
        train_gen = sample_generator(X, Label, X_trans, label_noise, seg_len, train_win, batchsize=batchsize)
        # test_gen = sample_generator(data_N, label_V, seg_len, test_win, batchsize=batchsize)
        Xtest, Ytest = val_gen(X, Label, X_trans, None, seg_len, test_win, test_batchsize )
        # Ytest = np.repeat(label_V, Xtest.shape[0]//num_sessions, axis=0)
        # Ytest = label_encoding(Ytest)

        if val_mode == 'random':
            Val = val_gen(X, Label, X_trans, None, seg_len, val_win, val_batchsize )
            # Val = val_gen(X, label_V, label_noise, seg_len, val_win, val_batchsize )
        elif val_mode == 'fix':
            Val_x = make_segs(X[:,val_win[0]:val_win[1],:], seg_len, seg_stride)
            Val_y = np.repeat(Label, Val_x.shape[0]//num_sessions, axis=0)
            Val = (Val_x, Val_y)

    elif LabelMode in ['OneHot', 'OneHot_N']:
        if LabelMode == 'OneHot':
            train_gen = sample_generator(X, Label, X_trans, label_smoothing, seg_len, train_win, batchsize=batchsize)
        elif LabelMode == 'OneHot_N':
            train_gen = sample_generator(X, Label, X_trans, onehot, seg_len, train_win, batchsize=batchsize)

        Xtest, Ytest = val_gen(X, Label, X_trans, None, seg_len, test_win, test_batchsize )

        if val_mode == 'random':
            Val = val_gen(X, Label, X_trans, onehot, seg_len, val_win, val_batchsize )
        elif val_mode == 'fix':
            Val_x = make_segs(X[:,val_win[0]:val_win[1],:], seg_len, seg_stride)
            Val_y = np.repeat(Label, Val_x.shape[0]//num_sessions, axis=0)
            Val = (Val_x, to_categorical(Val_y))

    elif LabelMode == 'OneHot_b':
        train_gen = sample_generator(X, to_binary(Label), X_trans, to_categorical, seg_len, train_win, batchsize=batchsize)
        Xtest, Ytest = val_gen(X, to_binary(Label), X_trans, None, seg_len, test_win, test_batchsize )

        if val_mode == 'random':
            Val = val_gen(X, to_binary(Label), X_trans, to_categorical, seg_len, val_win, val_batchsize )
        elif val_mode == 'fix':
            Val_x = make_segs(X[:,val_win[0]:val_win[1],:], seg_len, seg_stride)
            Val_y = np.repeat(to_binary(Label), Val_x.shape[0]//num_sessions, axis=0)
            Val = (Val_x, to_categorical(Val_y) )

    elif LabelMode == 'Regression':
        train_gen = sample_generator(X, Label, X_trans, label_encoding, seg_len, train_win, batchsize=batchsize)
        Xtest, Ytest = val_gen(X, Label, X_trans, None, seg_len, test_win, test_batchsize )

        if val_mode == 'random':
            Val = val_gen(X, Label, X_trans, label_encoding, seg_len, val_win, val_batchsize )
            # Val = val_gen(X, label_V, None, seg_len, val_win, val_batchsize )
        elif val_mode == 'fix':
            Val_x = make_segs(X[:,val_win[0]:val_win[1],:], seg_len, seg_stride)
            Val_y = np.repeat(Label, Val_x.shape[0]//num_sessions, axis=0)
            Val = (Val_x, label_encoding(Val_y) )


    '''
    training
    '''
    model.load_weights('/mnt/HDD/Benchmarks/DREAMER/ini_weights.h5') 
    steps = (train_win[1] - train_win[0]-seg_len)*num_sessions//batchsize if len(train_win) == 2 else len(train_win)*num_sessions//batchsize                
    hist = model.fit(train_gen,
                    epochs=Epochs, 
                    steps_per_epoch=steps,
                    validation_data = Val,
                    #  validation_steps= 2,
                    #  validation_split=0.2,
                    verbose=0,
                    callbacks=[ckpt, lrs],
                    #  shuffle = True,
        #                  class_weight = weight_dict
                        )

    model.load_weights(ckpt_path)

    '''
    Evaluation
    '''

    Ytest_label = np.unique(Ytest).astype(np.int8)  # Ytest should not be transformed for the following to work

    if LabelMode == 'Mixture':
        num_samples = 1024
        pred_model = model(Xtest)
        pred_samples = pred_model.sample(num_samples ).numpy()
        if not discrete_mixture:
            temp = pred_samples[None,...] - np.array([0,1,2,3,4]).reshape([5,1,1])

            pred_samples = np.argmin(np.abs(temp), axis=0)

        pred_hist = np.array([np.sum(pred_samples==i, axis=0) for i in range(5)])
        pred_hist_score = pred_hist.T/num_samples  
        top1acc =  100*top_k_accuracy_score(Ytest, pred_hist_score[:, Ytest_label], k=1)
        top2acc =  100*top_k_accuracy_score(Ytest, pred_hist_score[:, Ytest_label], k=2)
        f1 = 100*f1_score(Ytest, np.argmax(pred_hist_score, axis=1), average='weighted')
        # print('Top-1 acc: {:.02f}'.format(100*top_k_accuracy_score(Ytest, pred_hist_score[:, Ytest_label], k=1)))
        # print('Top-2 acc: {:.02f}'.format(100*top_k_accuracy_score(Ytest, pred_hist_score[:, Ytest_label], k=2)))
        CM = confusion_matrix(Ytest, np.argmax(pred_hist_score, axis=1))

    if LabelMode in ['Gaussian','Gaussian_2']:
        num_samples = 1024
        pred_model = model(Xtest)

        pred_samples = pred_model.sample(num_samples ).numpy().squeeze()
        temp = pred_samples[None,...] - np.array([0,1,2,3,4]).reshape([5,1,1])
        pred_samples = np.argmin(np.abs(temp), axis=0)
        pred_hist = np.array([np.sum(pred_samples==i, axis=0) for i in range(5)])
        pred_hist_score = pred_hist.T/num_samples

        top1acc =  100*top_k_accuracy_score(Ytest, pred_hist_score[:, Ytest_label], k=1)
        top2acc =  100*top_k_accuracy_score(Ytest, pred_hist_score[:, Ytest_label], k=2)
        f1 = 100*f1_score(Ytest, np.argmax(pred_hist_score, axis=1), average='weighted')
        # print('Top-1 acc: {:.02f}'.format(100*top_k_accuracy_score(Ytest, pred_hist_score[:, Ytest_label], k=1)))
        # print('Top-2 acc: {:.02f}'.format(100*top_k_accuracy_score(Ytest, pred_hist_score[:, Ytest_label], k=2)))
        CM = confusion_matrix(Ytest, np.argmax(pred_hist_score, axis=1))    

    elif LabelMode in ['OneHot','OneHot_N']:
        pred = model.predict(Xtest)
        top1acc =  100*top_k_accuracy_score(Ytest, pred[:, Ytest_label], k=1)
        top2acc =  100*top_k_accuracy_score(Ytest, pred[:, Ytest_label], k=2)
        f1 = 100*f1_score(Ytest, np.argmax(pred, axis=1), average='weighted')
        # print('Top-1 acc: {:.02f}'.format(100*top_k_accuracy_score(Ytest, pred[:, Ytest_label], k=1)))
        # print('Top-2 acc: {:.02f}'.format(100*top_k_accuracy_score(Ytest, pred[:, Ytest_label], k=2)))
        CM = confusion_matrix(Ytest, np.argmax(pred, axis=1))

    elif LabelMode == 'OneHot_b':
        pred = model.predict(Xtest)
        top1acc = 100*top_k_accuracy_score(Ytest,  np.argmax(pred,axis=1), k=1)
        top2acc = 100
        f1 = 100*f1_score(Ytest, np.argmax(pred, axis=1), average='binary')
        # print('Top-1 acc: {:.02f}'.format(100*top_k_accuracy_score(Ytest,  np.argmax(pred,axis=1), k=1)))
        # print('Top-2 acc: {:.02f}'.format(100*top_k_accuracy_score(Ytest_b,  pred, k=2)))
        CM = confusion_matrix(Ytest, np.argmax(pred, axis=1))    

    elif LabelMode == 'Regression':
        pred = model.predict(Xtest)
        distant = np.abs(pred - np.array([-1, -0.5, 0, 0.5, 1]))
        pred_label = np.argsort(distant, axis=1)
        pred_hist_score = np.exp(-distant)/np.sum( np.exp(-distant), axis=1, keepdims=True)
        # acc = accuracy_score(Ytest, pred_label[:,0])
        # print('acc: {:.2f}'.format(100*acc))
        top1acc =  100*top_k_accuracy_score(Ytest, pred_hist_score[:, Ytest_label], k=1)
        top2acc =  100*top_k_accuracy_score(Ytest, pred_hist_score[:, Ytest_label], k=2)
        f1 = 100*f1_score(Ytest, np.argmax(pred_hist_score, axis=1), average='weighted')
        # print('Top-1 acc: {:.02f}'.format(100*top_k_accuracy_score(Ytest, pred_hist_score[:, Ytest_label], k=1)))
        # print('Top-2 acc: {:.02f}'.format(100*top_k_accuracy_score(Ytest, pred_hist_score[:, Ytest_label], k=2)))
        CM = confusion_matrix(Ytest, pred_label[:,0])

    if LabelMode != 'OneHot_b':
        _, off_s = count_off_neibor(CM)
        # print(off_s)
        if LabelMode in ['OneHot', 'OneHot_N']:
            adj_a, adj_b, adj_c = adj_score(pred, Ytest)
        else:
            adj_a, adj_b, adj_c = adj_score(pred_hist_score, Ytest)
        # print("adj:{}, adj_in_true:{}, adj_in_true_2nd:{}".format(
        #         len(adj_a)/len(Ytest), len(adj_b)/len(Ytest), len(adj_c)/len(Ytest)) )

    '''
    collect performance into summary dictionary
    '''
    summary['top_1_acc'].append(top1acc)
    summary['top_2_acc'].append(top2acc)
    summary['F1'].append(f1)
    summary['CM'].append(CM)
    summary['off_neibor'].append(off_s)
    summary['closeness'].append([len(adj_a)/len(Ytest), len(adj_b)/len(Ytest), len(adj_c)/len(Ytest)])

'''
Write summary to file
'''
for _k, _v in summary.items():
    summary[_k] = np.array(_v)

with open('/mnt/HDD/Benchmarks/DREAMER/Summary/DREAMER_S{:02d}_{}_{}_{}.pkl'.format(
                Subject, Model_Choice, Label_Code, LabelMode), 'wb') as pkl:
    pickle.dump(summary, pkl)

entries = ['top_1_acc', 'top_2_acc', 'F1', 'off_neibor', 'closeness']
with open('/mnt/HDD/Benchmarks/DREAMER/DREAMER_{}_{}_{}.txt'.format(
                            Model_Choice, Label_Code, LabelMode), 'a') as file:
    file.write('\n' + '='*60 + '\n')
    file.write('{}\'s performance on Subject {}: \n'.format(Model_Choice, Subject))

    file.write('\n')
    file.write('mean(W):  ')
    file.writelines(['{:.02f}    '.format(np.mean(summary[s])) for s in entries])
    file.write('\n')
    file.write('std(W):  ')
    file.writelines(['{:.02f}    '.format(np.std(summary[s])) for s in entries])
    file.write('\n')