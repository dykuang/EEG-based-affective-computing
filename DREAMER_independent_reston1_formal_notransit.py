'''
Formal test script for gathering benchmark stats

try subject independent case:
    train on A
    fine tune on B's small subset (first 10 second)
    test on B's rest (the rest 50 second)
'''
#%%
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"  # place at top to take effect
# gpus = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(gpus[1], 'GPU')
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from scipy.io import loadmat
from utils import Embed_DREAMER
from Models import MSBAM, EEGNet, HST_model, HST_DP
from Probability_helper import *
from Losses import *
from scipy.stats import zscore
from keras.utils import to_categorical
from utils import sample_generator, val_gen, data_split_10, label_encoding, label_noise, make_segs
from scipy import signal
from sklearn.metrics import accuracy_score, confusion_matrix, top_k_accuracy_score, f1_score
from scipy.linalg import block_diag
# %%
'''
Preprocess and generator

with 1-s randomly cropped  signals
'''

def batch_band_pass(values, low_end_cutoff, high_end_cutoff, sampling_freq, btype='bandpass'):
    assert len(values.shape) == 3, "wrong input shape"
    S, T, C = values.shape
    X_filtered = np.empty(values.shape)
    lo_end_over_Nyquist = low_end_cutoff/(0.5*sampling_freq)
    hi_end_over_Nyquist = high_end_cutoff/(0.5*sampling_freq)

    bess_b,bess_a = signal.iirfilter(5,
                Wn=[lo_end_over_Nyquist,hi_end_over_Nyquist],
                btype=btype, ftype='bessel')

                
    for i in range(S):
        for j in range(C):
            X_filtered[i,:,j] = signal.filtfilt(bess_b,bess_a,values[i,:,j])
    
    return X_filtered

def batch_BLPF(values, high_end_cutoff, sampling_freq, btype='lp'):
    assert len(values.shape) == 3, "wrong input shape"
    S, T, C = values.shape
    X_filtered = np.empty(values.shape)
    sos = signal.butter(4, high_end_cutoff, btype, fs=sampling_freq, output='sos')
                
    for i in range(S):
        for j in range(C):
            X_filtered[i,:,j] = signal.sosfilt(sos, values[i,:,j])
    
    return X_filtered


def label_smoothing(Y):
    '''
    Y: N by 1
    '''
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

def count_off_neibor(M):   
    for i in range(1, M.shape[0]):
        M[i-1:i+2,i]=0
    M[1,0] = 0
    M[-2,-1] = 0
    M[0,0] = 0
    M[-1,-1] = 0

    return M, np.sum(M)

def adj_score(Ypred, Ytrue):
    '''
    Ypred: prediction score for each class
    Ytrue: the true label
    '''
    pred_rk = np.argsort(Ypred, axis=1)[:, ::-1] # descending
    # N_samples, _ = Ypred.shape
    match_cond = np.abs(pred_rk[:,0] - pred_rk[:,1]) == 1 
    true_cond = pred_rk[:,0] == Ytrue
    true_cond_2nd = np.logical_or( pred_rk[:,1] == Ytrue, true_cond)

    match = np.where(match_cond)[0]
    true_match = np.where(true_cond & match_cond)[0]
    true_match_2nd = np.where(true_cond_2nd & match_cond)[0]

    return match, true_match, true_match_2nd 

def onehot(Y):
    return to_categorical(Y, 5)


def Embed_X(X, seglen=128):
    return Embed_DREAMER(X.transpose(0,2,1), seglen)[...,None]

def DataGen(XData, YData, X_transform=None, batchsize=200, shuffle=True, SampleWeights=None):
    '''
    XData: data to chose from  (segments, time, channel)
    YData: onehot like encoded label
    batchsize better be a mulitple of 5

    Possible improvements:
        dynamic probability for sampling.
    '''

    # X = np.zeros((batchsize, seg_len, XData.shape[-1]))
    if len(YData.shape)>1:
        YLabel = np.argmax(YData, axis=1)
    else:
        YLabel = YData.copy()
    label_pos = np.unique(YLabel)
    indexes = np.arange(batchsize)

    Type_idx = [np.where(YLabel == i)[0] for i in label_pos]
    
    num_per_label = batchsize//len(label_pos)

    while True:
        # Y = np.zeros((batchsize, 5))  #for the problem Y‘dimension is increasing every time after transformation per interation
        
        # for i, _idx in enumerate(Type_idx):
        #     selected_idx = np.random.choice(_idx, size=num_per_label,replace=True)
        #     X[i*num_per_label:(i+1)*num_per_label] = XData[selected_idx]
        #     Y[i*num_per_label:(i+1)*num_per_label] = YData[selected_idx]
        selected_idx=[]
        for i, _idx in enumerate(Type_idx):
            selected_idx.append(np.random.choice(_idx, size=num_per_label,replace=True))
        selected_idx=np.array(selected_idx).reshape(-1)        
        X = XData[selected_idx]
        Y = YData[selected_idx]
        
        if shuffle:
            np.random.shuffle(indexes)
        
        if SampleWeights is None:
            if X_transform != None:
                XX = X_transform(X)
                yield XX[indexes], Y[indexes]
            else:
                yield X[indexes], Y[indexes]
        else:
            S_weights = SampleWeights[selected_idx]
            if X_transform != None:
                XX = X_transform(X)
                yield XX[indexes], Y[indexes], S_weights[indexes]
            else:
                yield X[indexes], Y[indexes], S_weights[indexes]

def prepare_Y(Y):
    '''
    input:
        Y: The integer label 0 - 4, shape: (22, 18, 3)
    '''
    dist = []
    for i in range(18):
        temp = []
        for j in range(3):
            '''
            alternatives possible below: for example picking the loc with miximum count
            '''
            temp.append( np.histogram(Y[:,i,j], bins=5, range=(0, 4))[0]/NUM_TRAIN_SUBS)
        dist.append(temp)
    
    return np.array(dist)

#%%
'''
Parameters
'''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--Test_Subject', help='subject index for Testing')
parser.add_argument('--ModelChoice', help='choice of classificatoin model')
parser.add_argument('--LabelMode', help='choice of labels for training')
# parser.add_argument('--ValMode', default='random', help='mode of validation, random or fix')
parser.add_argument('--LabelChoice', help='choice of label: V, A or D')
parser.add_argument('--Pretrain', default='F', help='Use pretrained model or not')

args = parser.parse_args()
Test_sub = int(args.Test_Subject)

# Train_sub = 1
# Test_sub = 2
# if Test_sub == 0:
#     Test_sub = 23


Seg_len = 128
Sampling_freq = 128
Stride_rate=1
Num_channels = 14
Num_sessions = 18
Num_subjects = 23
Total_cls = 5

Model_Choice = args.ModelChoice # 'MSBAM', 'EEGNet', 'HST'
Label_Code = args.LabelChoice # 'V', 'A', 'D' 
LabelMode = args.LabelMode  # Mixture, Gaussian, Regression OneHot_b or OneHot

# Model_Choice = 'EEGNet'
# Label_Code = 'V'
# LabelMode = 'Mixture'
# Pretrained = True

if args.Pretrain == 'T':
    Pretrained = True
    save_path = '/mnt/HDD/Benchmarks/DREAMER/ckpt_SI/DREAMER_S{:02d}_{}_{}_{}_reston1.h5'.format( Test_sub, Model_Choice, Label_Code, LabelMode)
    summary_path = '/mnt/HDD/Benchmarks/DREAMER/Summary_SI/DREAMER_Pretrained_{}_{}_{}_reston1'.format(Model_Choice, Label_Code, LabelMode)

else:
    Pretrained = False
    save_path = '/mnt/HDD/Benchmarks/DREAMER/ckpt_SI/DREAMER_S{:02d}Alone_{}_{}_{}_reston1.h5'.format(Test_sub, Model_Choice, Label_Code, LabelMode)
    summary_path = '/mnt/HDD/Benchmarks/DREAMER/Summary_SI/DREAMER_Alone_{}_{}_{}_reston1'.format(Model_Choice, Label_Code, LabelMode)

Discrete_mixture = False
USE_GENERATOR = True
Correct_Y = True
#%%
from utils import make_partition
import matplotlib.pyplot as plt
NUM_TRAIN_SUBS = 22
X = []
Y = []
for Subject in range(1,24):
    X.append(loadmat('/mnt/HDD/Datasets/DREAMER/S{:02d}_1min.mat'.format(Subject))['X'].transpose((1,0,2)))
    Y.append(loadmat('/mnt/HDD/Datasets/DREAMER/Labels.mat')['Y'][Subject-1])
X = np.array(X)  # (23, 18, 7680, 14)
Y = np.array(Y)-1.0  # (23, 18, 3)


if Test_sub == 1:
    Xtrain_raw=X[1:]
    LabelTrain_raw = Y[1:] 
elif Test_sub ==23:
    Xtrain_raw=X[:-1]
    LabelTrain_raw = Y[:-1]
else:
    Xtrain_raw=np.r_[X[:Test_sub-1], X[Test_sub:]]
    LabelTrain_raw = np.r_[Y[:Test_sub-1], Y[Test_sub:]]  

Xtest_raw = X[Test_sub-1]
LabelTest_raw = Y[Test_sub-1]
# Xtrain = batch_band_pass(Xtrain, 0.1, 50, Sampling_freq)
# Xtest = batch_band_pass(Xtest, 0.1, 50, Sampling_freq)
dist = prepare_Y(LabelTrain_raw) # label distribution per subject, per trial
Y_corrected = np.argmax(dist, axis=-1)

Xtrain_raw = np.reshape(Xtrain_raw, (-1, 7680, 14))
LabelTrain_raw = np.reshape(LabelTrain_raw, (-1,3))
Y_corrected = np.tile(Y_corrected, (NUM_TRAIN_SUBS,1))


X_prep = lambda x: x/np.max(np.abs(x), axis=1, keepdims=True)
Xtrain_raw = X_prep(Xtrain_raw)
Xtest_raw = X_prep(Xtest_raw)

Xtrain = make_partition(Xtrain_raw, fs=Sampling_freq , seconds=Seg_len//128, stride_rate=Stride_rate)
Xtest = make_partition(Xtest_raw, fs=Sampling_freq , seconds=Seg_len//128, stride_rate=Stride_rate)

# Xtrain = (Xtrain - np.mean(Xtrain, axis=1,keepdims=True))/( np.max(Xtrain, axis=1, keepdims=True) - np.min(Xtrain, axis=1, keepdims=True) )  #before or after partition?
# Xtest = (Xtest - np.mean(Xtest, axis=1,keepdims=True))/( np.max(Xtest, axis=1, keepdims=True) - np.min(Xtest, axis=1, keepdims=True) )


print('The range of Xtrain: max: {}, min:{}'.format(np.max(Xtrain), np.min(Xtrain)))
print('The range of Xtest: max: {}, min:{}'.format(np.max(Xtest), np.min(Xtest)))


if Label_Code == 'V':
    Label_choice = 0  #range [0, 1, 2, 3, 4]
elif Label_Code == 'A': 
    Label_choice = 1  #range [0, 1, 2, 3, 4]
elif Label_Code == 'D':
    Label_choice = 2  #range [0, 1, 2, 3, 4]
else:
    raise('Not valid Label!')

#%%
'''
check rating per trial
'''
# print(LabelTrain_raw[:,Label_choice])
# print(LabelTest_raw[:,Label_choice])
# fig, ax = plt.subplots(1,2,figsize=(6,4))
# ax[0].plot(LabelTrain_raw[:,Label_choice])
# ax[1].plot(LabelTest_raw[:,Label_choice])

if Model_Choice == 'MSBAM':
    X_trans = lambda x: Embed_X(x, seglen=Seg_len)
    Xtest = Embed_X(Xtest,seglen=Seg_len)
    if not USE_GENERATOR:
        Xtrain = Embed_X(Xtrain, seglen=Seg_len) 
else:
    X_trans = None

# LabelTrain = LabelTrain.reshape((-1,3))
if Correct_Y:
    LabelTrain= np.tile(Y_corrected, (60, 1)) 
else:
    LabelTrain= np.tile(LabelTrain_raw, (60, 1)) 
LabelTest = np.tile(LabelTest_raw, (60, 1))

if LabelMode in ['OneHot']:
    Ytrain = label_smoothing(LabelTrain[:,Label_choice])
    Ytest = onehot(LabelTest[:,Label_choice])
elif LabelMode in ['OneHot_N']:
    Ytrain = onehot(LabelTrain[:,Label_choice])
    Ytest = onehot(LabelTest[:,Label_choice])
elif LabelMode in ['Regression']:
    Ytrain = label_encoding(LabelTrain[:,Label_choice])
    Ytest = label_encoding(LabelTest[:,Label_choice])
else:
    Ytrain = LabelTrain[:,Label_choice]*1.0
    Ytest = LabelTest[:,Label_choice]*1.0

#%%
idx = np.arange(Xtrain.shape[0])
np.random.shuffle(idx)
Xtrain = Xtrain[idx]
Ytrain = Ytrain[idx]

#%%
'''
Prepare a weight dict
'''
YDist = np.sum(onehot(LabelTrain[:,Label_choice]), axis=0)
_max = np.max(YDist)
weight_dict= {}
for i in range(5):
    if YDist[i] != 0:
        weight_dict[i] = _max/YDist[i]
    else:
        weight_dict[i] = 1
# print(np.sum(Ytrain, axis=0))

#%%
'''
check label distribution between Ytrain and Ytest
'''
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1,2,figsize=(6,4))
# ax[0].hist(1+np.argmax(Ytrain,axis=1))
# ax[0].set_title('{}'.format(np.sum(Ytrain, axis=0)))
# ax[1].hist(1+np.argmax(Ytest, axis=1))
# ax[1].set_title('{}'.format(np.sum(Ytest, axis=0)))

#%%
'''
Prepare models
'''
Lr_init=1e-2

if LabelMode == 'Mixture':
    # prob_model = lambda x: mixture_logistic(x, Discrete_mixture, 3)
    prob_model = lambda x: mixture_gaussian(x, Discrete_mixture, 3)
elif LabelMode == 'Gaussian':
    prob_model = lambda x: Single_Gaussian(x, Discrete_mixture)
elif LabelMode == 'Gaussian_2':
    prob_model = lambda x: Two_Gaussians(x)
else:
    prob_model = None

if Model_Choice == 'EEGNet':
    model = EEGNet(prob_model, nb_classes = 5, LabelMode=LabelMode, activation='elu',
                    Chans = Num_channels, Samples = Seg_len,
                    dropoutRate = 0.5, kernLength = 8, F1 = 8, 
                    D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout')
elif Model_Choice == 'MSBAM':
    model = MSBAM(input_shape = (9,9,Seg_len,1), LabelMode=LabelMode,
                  Nclass = 5, Prob_model = prob_model, Droprate=0.5)
elif Model_Choice == 'HST':
    Version = 0
    if Version == 0:
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
                        activation='elu', droprate=0.2, F1=8, ksize=64)
        
    elif Version == 1:
        '''
        Version 2
        '''
        def create_adj(): # adjacency matrix excluding self loop
            A1 = np.zeros((3,3))
            A2 = np.zeros((3,3))
            A3 = np.zeros((2,2))

            A_R = np.zeros((5,5))

            A1[0,1] = 1.0
            A1[1,0] = 1.0
            A1[0,2] = 1.0
            A1[2,0] = 1.0
            A1[1,2] = 1.0
            A1[2,1] = 1.0
    

            A2[0,1] = 1.0
            A2[1,0] = 1.0
            A2[1,2] = 1.0
            A2[2,1] = 1.0


            A3[0,1] = 1
            A3[1,0] = 1

            A4 = A2.copy()
            A5 = A1[::-1,::-1]


            A_R[0,1] = 1.0
            A_R[1,0] = 1.0
            A_R[1,2] = 1.0
            A_R[2,1] = 1.0
            A_R[2,3] = 1.0
            A_R[3,2] = 1.0  
            A_R[3,4] = 1.0
            A_R[4,3] = 1.0            
            A_R[0,4] = 1.0
            A_R[4,0] = 1.0

            return [A1, A2, A3, A4, A5], A_R

        adjN, adjR = create_adj()
        model = HST_model(input_shape = (128,14), LabelMode=LabelMode, 
                        nb_classes=5, Prob_model=prob_model, mergeMode='dense',
                        loc=[0,3,6,8,11,14], adj_list=adjN, 
                        hop_list=[1,2,1,2,1], 
                        region_adjM=adjR, 
                        region_hop=2, poly_type='chebyshev',
                        activation='elu', droprate=0.2, F1=8, ksize=64)    
    
    elif Version == 2:
        '''
        Version 3
        '''
        def create_adj(): # adjacency matrix excluding self loop
            A1 = np.zeros((5,5))
            A2 = np.zeros((4,4))

            A_R = np.zeros((3,3))

            A1[0,2] = 1.0
            A1[2,0] = 1.0
            A1[1,2] = 1.0
            A1[2,1] = 1.0
            A1[1,3] = 1.0
            A1[3,1] = 1.0
            A1[2,3] = 1.0
            A1[3,2] = 1.0
            A1[3,4] = 1.0
            A1[4,3] = 1.0
    

            A2[0,1] = 1.0
            A2[1,0] = 1.0
            A2[1,2] = 1.0
            A2[2,1] = 1.0
            A2[2,3] = 1.0
            A2[3,2] = 1.0

            A3 = A1[::-1,::-1]

            A_R[0,1] = 1.0
            A_R[1,0] = 1.0
            A_R[1,2] = 1.0
            A_R[2,1] = 1.0          
            A_R[0,2] = 1.0
            A_R[2,0] = 1.0

            return [A1, A2, A3], A_R

        adjN, adjR = create_adj()
        model = HST_model(input_shape = (128,14), LabelMode=LabelMode, 
                        nb_classes=5, Prob_model=prob_model, mergeMode='dense',
                        loc=[0,5,9,14], adj_list=adjN, 
                        hop_list=[3,3,3], 
                        region_adjM=adjR, 
                        region_hop=1, poly_type='chebyshev',
                        activation='elu', droprate=0.2, F1=8, ksize=64)   

elif Model_Choice == 'HSTDP':
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

        return [A1, A2, A3, A4]
    adjN = create_adj()
    A_C = block_diag(adjN[0], adjN[1], adjN[2], adjN[3])
    # from spektral.layers import GCNConv
    # G = GCNConv.preprocess(A_C[None,...])
    model = HST_DP(input_shape = (128,14), LabelMode=LabelMode, 
                nb_classes=5, Prob_model=prob_model, Adj_C = A_C[None,...], n_regions=len(adjN),
                activation='elu', droprate=0.2, F1=8, ksize=64)


# %%  
'''
training
''' 
from tensorflow.keras.callbacks import  LearningRateScheduler

if Pretrained:
    Batchsize = 120
    # USE_SampleWeights = False
    Epochs = 100

    def scheduler(epoch, lr):
        if epoch<50:
            return lr
        else:
            return Lr_init/5

    lr_s = LearningRateScheduler(scheduler)

    if LabelMode in ['Mixture', 'Gaussian_2']:
        model.compile( 
                    loss=NLL_loss,
                    # loss=hybrid_loss,
                    #   metrics=['mean_absolute_error'],
                    optimizer=Adam(Lr_init)
        )
    elif LabelMode == 'Gaussian':
        model.compile( 
                    # loss = KL_loss,
                    # loss = hybrid_loss,
                    loss = NLL_loss,
                    #   metrics=['mean_absolute_error'],
                    optimizer=Adam(Lr_init)
        )
    elif LabelMode in ['OneHot', 'OneHot_b', 'OneHot_N']:
        model.compile( 
                    loss='categorical_crossentropy',
                    # loss ='mean_squared_error',
                    metrics=['accuracy'],
                    optimizer=Adam(Lr_init)
        )
    elif LabelMode == 'Regression':
        model.compile( 
                    loss='mean_absolute_error',
                    # loss = Noise_loss(),
                    #   metrics=['mean_absolute_error'],
                    optimizer=Adam(Lr_init)
        )
    model.summary()

    if USE_GENERATOR:
        train_gen = DataGen(Xtrain, Ytrain, batchsize=Batchsize, 
                            X_transform=X_trans, shuffle=True,
                            SampleWeights=None)
        steps = Xtrain.shape[0]//Batchsize               
        hist = model.fit(train_gen,
                        epochs=Epochs, 
                        steps_per_epoch=steps,
                        validation_data = (Xtest, Ytest),
                        #  validation_steps= 2,
                        #  validation_split=0.2,
                        verbose=1,
                        callbacks=[lr_s],
                        #  shuffle = True,
            #                  class_weight = weight_dict
                        )
    else:
        hist = model.fit(Xtrain[...,None], Ytrain,
                        epochs=Epochs, 
                        batch_size = Batchsize,
                        #  steps_per_epoch=steps,
                        validation_data = (Xtest, Ytest),
                        #  validation_steps= 2,
                        #  validation_split=0.2,
                        verbose=1,
                        callbacks=[lr_s],
                        shuffle = True,
                        # sample_weight = SampleWeights if USE_SampleWeights else None,
                        class_weight = weight_dict
                        )

    # pred = model.predict(Xtest)
    # model.load_weights(ckpt_path)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # # plt.subplot(1,2,1)
    # plt.plot(hist.history['loss'])
    # plt.plot(hist.history['val_loss'])
    # plt.subplot(1,2,2)
    # plt.plot(hist.history['my_acc'])
    # plt.plot(hist.history['val_my_acc'])

    '''
    Second stage tuning prepartion
    Specify layer weights that will remain frozen
    '''
    if Model_Choice == 'MSBAM':
        for l in model.layers:
            if l.name not in ['Sdense','T1dense','T2dense','last_dense']:
                l.trainable = False
    else:
        for l in model.layers[:3]:
            l.trainable = False 

FineT_init_lr = 1e-3
def fineT_scheduler(epoch, lr):
    if epoch<200:
        return lr
    elif epoch >= 200 and epoch <= 350:
        return FineT_init_lr/5
    else:
        return FineT_init_lr/10

f_lr_s = LearningRateScheduler(fineT_scheduler)

if LabelMode in ['Mixture', 'Gaussian_2']:
    model.compile( 
                loss=NLL_loss,
                # loss=hybrid_loss,
                #   metrics=['mean_absolute_error'],
                optimizer=Adam(FineT_init_lr)
    )
elif LabelMode == 'Gaussian':
    model.compile( 
                # loss = KL_loss,
                # loss = hybrid_loss,
                loss = NLL_loss,
                #   metrics=['mean_absolute_error'],
                optimizer=Adam(FineT_init_lr)
    )
elif LabelMode in ['OneHot', 'OneHot_b', 'OneHot_N']:
    model.compile( 
                loss='categorical_crossentropy',
                # loss ='mean_squared_error',
                metrics=['accuracy'],
                optimizer=Adam(FineT_init_lr)
    )
elif LabelMode == 'Regression':
    model.compile( 
                loss='mean_absolute_error',
                # loss = Noise_loss(),
                #   metrics=['mean_absolute_error'],
                optimizer=Adam(FineT_init_lr)
    )
model.summary()

# %%
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# Ytest_ord = np.argmax(Ytest, axis=1)
# Xtrain_FT, Xtest_FT, Ytrain_FT, Ytest_FT = train_test_split(Xtest, Ytest, test_size=0.8, 
#                                                             random_state=42, shuffle=True,
#                                                             stratify=Ytest_ord)
# print('Fine tune Y distribution: {}'.format(YDist_FT) )
# print('Using class weight: {}'.format(cls_weight))

trainFT_win = (0, 128*10)
# trainFT_gen = sample_generator(Xtest_raw, LabelTest_raw[:,Label_choice], X_trans, onehot, Seg_len, trainFT_win,
#                                batchsize=Batchsize)

Xtrain_FT = make_partition(Xtest_raw[:,:trainFT_win[-1],:], fs=Sampling_freq , 
                          seconds=Seg_len//128, stride_rate=Stride_rate/4)
Ytrain_FT = np.tile(LabelTest_raw[:,Label_choice], Xtrain_FT.shape[0]//18)
Ytrain_FT_OH = onehot(Ytrain_FT)

Xtest_FT = make_partition(Xtest_raw[:,trainFT_win[-1]:,:], fs=Sampling_freq , 
                          seconds=Seg_len//128, stride_rate=Stride_rate)
Ytest_FT = np.tile(LabelTest_raw[:,Label_choice], Xtest_FT.shape[0]//18)
Ytest_FT_OH = onehot(Ytest_FT)

if Model_Choice == 'MSBAM':
    Xtrain_FT = Embed_X(Xtrain_FT, seglen=Seg_len) 
    Xtest_FT = Embed_X(Xtest_FT, seglen=Seg_len)   
else:
    Xtrain_FT = Xtrain_FT[...,None]
    Xtest_FT = Xtest_FT[...,None]  

YDist_FT = np.sum(Ytrain_FT_OH, axis=0)
_max = np.max(YDist_FT)
cls_weight = {}
for i in range(5):
    if YDist_FT[i] != 0:
        cls_weight[i] = _max/YDist_FT[i]
    else:
        cls_weight[i] = 1
print(np.sum(Ytrain_FT_OH, axis=0))

if LabelMode == 'Regression':
    Y_in_Train = label_encoding(Ytrain_FT)
    Y_in_Test = label_encoding(Ytest_FT)
    SampleWeights = np.ones_like(Y_in_Train)
    regression_y  = [-1.0, -0.5, -0.0, 0.5, 1.0]
    for i in range(5):
        SampleWeights[np.where(Y_in_Train==regression_y[i])] = cls_weight[i]

elif LabelMode in ['OneHot_N']:
    Y_in_Train = Ytrain_FT_OH
    Y_in_Test = Ytest_FT_OH
elif LabelMode in ['OneHot']:
    Y_in_Train = label_smoothing(Ytrain_FT)
    Y_in_Test = Ytest_FT_OH
else:
    Y_in_Train = Ytrain_FT*1.0
    Y_in_Test = Ytest_FT*1.0

#%%
FT_Batchsize = 100
FT_Epochs=400

hist = model.fit(Xtrain_FT, Y_in_Train,
                epochs=FT_Epochs, 
                batch_size = FT_Batchsize,
                validation_data = (Xtest_FT, Y_in_Test),
                verbose=1,
                # callbacks=[f_lr_s],
                shuffle = True,
                sample_weight = SampleWeights if LabelMode == 'Regression' else None,
                class_weight = cls_weight if LabelMode != 'Regression' else None
                )

# steps = (train_win[1] - train_win[0]-Seg_len)*Num_sessions//FT_Batchsize if len(train_win) == 2 else len(train_win)*Num_sessions//FT_Batchsize                
# hist = tt.fit(train_gen,
#                  epochs=FT_Epochs, 
#                  steps_per_epoch=steps//5,
#                  validation_data = (Xtest_FT, Ytest_FT),
#                  #  validation_steps= 2,
#                  #  validation_split=0.2,
#                  verbose=1,
#                 callbacks=[f_lr_s],
#                 #  shuffle = True,
#     #                  class_weight = weight_dict
#                 )

# pred = model.predict(Xtest)
# model.load_weights(ckpt_path)
# plt.figure()
# plt.subplot(1,2,1)
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.subplot(1,2,2)
# plt.plot(hist.history['accuracy'])
# plt.plot(hist.history['val_accuracy'])

if Model_Choice == 'HST':
    model.save_weights(save_path[:-3] + '_V{}'.format(Version) + '.h5')
else:
    model.save_weights(save_path)

# %%

# pred_FT = model.predict(Xtest_FT)
# pred_ord = np.argmax(pred_FT, axis=1)
# true_ord = np.argmax(Ytest_FT_OH, axis=1)
# FT_label = np.unique(true_ord).astype(np.int8)  # Ytest should not be transformed for the following to work


# print('Top-1 acc: {:.02f}'.format(100*top_k_accuracy_score(true_ord, pred_FT[:,FT_label], k=1)))
# print('Top-2 acc: {:.02f}'.format(100*top_k_accuracy_score(true_ord, pred_FT[:,FT_label], k=2)))
# print('F1: {:.02f}'.format(100*f1_score(true_ord, pred_ord, average='weighted')))

# CM = confusion_matrix(true_ord, pred_ord)
# print(CM)
# _, off_s = count_off_neibor(CM)
# adj_a, adj_b, adj_c = adj_score(pred_FT, true_ord)
# print(off_s)
# print("adj:{}, adj_in_true:{}, adj_in_true_2nd:{}".format(
#         len(adj_a)/len(Ytest), len(adj_b)/len(Ytest), len(adj_c)/len(Ytest)) )
# %%
import pickle
# summary_pkl_path = os.path.join(summary_path,'.pkl')
if Model_Choice == 'HST':
    summary_pkl_path = summary_path+'_V{}'.format(Version)+'.pkl'
else:
    summary_pkl_path = summary_path+'.pkl'
_key = 'S{:02d}'.format(Test_sub)

if os.path.isfile(summary_pkl_path ):
    with open(summary_pkl_path, 'rb') as pkl:
        past_summary = pickle.load(pkl)
    summary = past_summary.copy()
    summary[_key] = {}
else:
    # temp_dict = {
    #     'top_1_acc': [],
    #     'top_2_acc': [],
    #     'F1':[],
    #     'CM':[],
    #     'off_neibor':[],
    #     'closeness':[]
    # }

    summary = {_key:{}}

FT_label = np.unique(Ytest_FT).astype(np.int8)  # Ytest should not be transformed for the following to work

if LabelMode == 'Mixture':
    num_samples = 1024
    pred_model = model(Xtest_FT)
    pred_samples = pred_model.sample(num_samples ).numpy()
    # print(pred_samples.shape)
    if not Discrete_mixture:
        temp = pred_samples[None,...] - np.array([0,1,2,3,4]).reshape([5,1,1])

        pred_samples = np.argmin(np.abs(temp), axis=0)

    pred_hist = np.array([np.sum(pred_samples==i, axis=0) for i in range(5)])
    pred_hist_score = pred_hist.T/num_samples  
    # print(pred_hist_score.shape)
    # print(pred_hist_score[:2])
    top1acc =  100*top_k_accuracy_score(Ytest_FT, pred_hist_score[:, FT_label], k=1)
    top2acc =  100*top_k_accuracy_score(Ytest_FT, pred_hist_score[:, FT_label], k=2)
    f1 = 100*f1_score(Ytest_FT, np.argmax(pred_hist_score, axis=1), average='weighted')
    # print('Top-1 acc: {:.02f}'.format(100*top_k_accuracy_score(Ytest, pred_hist_score[:, Ytest_label], k=1)))
    # print('Top-2 acc: {:.02f}'.format(100*top_k_accuracy_score(Ytest, pred_hist_score[:, Ytest_label], k=2)))
    CM = confusion_matrix(Ytest_FT, np.argmax(pred_hist_score, axis=1))

# if LabelMode in ['Gaussian','Gaussian_2']:
#     num_samples = 1024
#     pred_model = model(Xtest_FT)

#     pred_samples = pred_model.sample(num_samples ).numpy().squeeze()
#     temp = pred_samples[None,...] - np.array([0,1,2,3,4]).reshape([5,1,1])
#     pred_samples = np.argmin(np.abs(temp), axis=0)
#     pred_hist = np.array([np.sum(pred_samples==i, axis=0) for i in range(5)])
#     pred_hist_score = pred_hist.T/num_samples

#     top1acc =  100*top_k_accuracy_score(Ytest_FT, pred_hist_score[:, FT_label], k=1)
#     top2acc =  100*top_k_accuracy_score(Ytest_FT, pred_hist_score[:, FT_label], k=2)
#     f1 = 100*f1_score(Ytest_FT, np.argmax(pred_hist_score, axis=1), average='weighted')
#     # print('Top-1 acc: {:.02f}'.format(100*top_k_accuracy_score(Ytest, pred_hist_score[:, Ytest_label], k=1)))
#     # print('Top-2 acc: {:.02f}'.format(100*top_k_accuracy_score(Ytest, pred_hist_score[:, Ytest_label], k=2)))
#     CM = confusion_matrix(Ytest_FT, np.argmax(pred_hist_score, axis=1))    

elif LabelMode in ['OneHot','OneHot_N']:
    pred = model.predict(Xtest_FT)
    top1acc =  100*top_k_accuracy_score(Ytest_FT, pred[:, FT_label], k=1)
    top2acc =  100*top_k_accuracy_score(Ytest_FT, pred[:, FT_label], k=2)
    f1 = 100*f1_score(Ytest_FT, np.argmax(pred, axis=1), average='weighted')
    # print('Top-1 acc: {:.02f}'.format(100*top_k_accuracy_score(Ytest, pred[:, Ytest_label], k=1)))
    # print('Top-2 acc: {:.02f}'.format(100*top_k_accuracy_score(Ytest, pred[:, Ytest_label], k=2)))
    CM = confusion_matrix(Ytest_FT, np.argmax(pred, axis=1))

# elif LabelMode == 'OneHot_b':
#     pred = model.predict(Xtest_FT)
#     top1acc = 100*top_k_accuracy_score(Ytest_FT,  np.argmax(pred,axis=1), k=1)
#     top2acc = 100
#     f1 = 100*f1_score(Ytest_FT, np.argmax(pred, axis=1), average='binary')
#     # print('Top-1 acc: {:.02f}'.format(100*top_k_accuracy_score(Ytest,  np.argmax(pred,axis=1), k=1)))
#     # print('Top-2 acc: {:.02f}'.format(100*top_k_accuracy_score(Ytest_b,  pred, k=2)))
#     CM = confusion_matrix(Ytest_FT, np.argmax(pred, axis=1))    

elif LabelMode == 'Regression':
    pred = model.predict(Xtest_FT)
    distant = np.abs(pred - np.array([-1, -0.5, 0, 0.5, 1]))
    pred_label = np.argsort(distant, axis=1)
    pred_hist_score = np.exp(-distant)/np.sum( np.exp(-distant), axis=1, keepdims=True)
    # acc = accuracy_score(Ytest, pred_label[:,0])
    # print('acc: {:.2f}'.format(100*acc))
    top1acc =  100*top_k_accuracy_score(Ytest_FT, pred_hist_score[:, FT_label], k=1)
    top2acc =  100*top_k_accuracy_score(Ytest_FT, pred_hist_score[:, FT_label], k=2)
    f1 = 100*f1_score(Ytest_FT, np.argmax(pred_hist_score, axis=1), average='weighted')
    # print('Top-1 acc: {:.02f}'.format(100*top_k_accuracy_score(Ytest, pred_hist_score[:, Ytest_label], k=1)))
    # print('Top-2 acc: {:.02f}'.format(100*top_k_accuracy_score(Ytest, pred_hist_score[:, Ytest_label], k=2)))
    CM = confusion_matrix(Ytest_FT, pred_label[:,0])

if LabelMode != 'OneHot_b':
    _, off_s = count_off_neibor(CM)
    # print(off_s)
    if LabelMode in ['OneHot', 'OneHot_N']:
        adj_a, adj_b, adj_c = adj_score(pred, Ytest_FT)
    else:
        adj_a, adj_b, adj_c = adj_score(pred_hist_score, Ytest_FT)
    # print("adj:{}, adj_in_true:{}, adj_in_true_2nd:{}".format(
    #         len(adj_a)/len(Ytest), len(adj_b)/len(Ytest), len(adj_c)/len(Ytest)) )

'''
collect performance into summary dictionary
'''
summary[_key]['top_1_acc'] = top1acc
summary[_key]['top_2_acc'] = top2acc
summary[_key]['F1'] = f1
summary[_key]['CM'] = CM
summary[_key]['off_neibor'] = off_s
summary[_key]['closeness'] = [len(adj_a)/len(Ytest_FT), 
                                len(adj_b)/len(Ytest_FT), 
                                len(adj_c)/len(Ytest_FT)]

#%%
'''
Write summary to file
'''
# for _k, _v in summary.items():
#     summary[_k] = np.array(_v)

with open(summary_pkl_path, 'wb') as pkl:
    pickle.dump(summary, pkl)

entries = ['top_1_acc', 'top_2_acc', 'F1', 'off_neibor']

# summary_txt_path = os.path.join(summary_path,'.txt')
if Model_Choice == 'HST':
    summary_txt_path = summary_path + '_V{}'.format(Version) +'.txt'
else:
    summary_txt_path = summary_path + '.txt'
with open(summary_txt_path, 'a') as file:
    file.write('\n' + '='*60 + '\n')
    file.write('{}\'s performance on Subject S{:02d}: \n'.format(Model_Choice, Test_sub))
    file.writelines(['{:.02f}    '.format(summary[_key][s]) for s in entries])
    file.writelines(['{:.02f}    '.format(summary[_key]['closeness'][i]) for i in range(3)])
    file.write('\n')