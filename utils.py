'''
Utility functions for training
'''

import numpy as np
import random
from scipy.stats import truncnorm

#%%
def make_partition(x, fs=128, seconds=1, stride_rate=1):
    '''
    x: input of shape (trials, time_len, channels)
    fs:sampling frequency
    seconds: lasting time of segments
    stride_rate: ratio in terms of fs 
    '''
    temp = []
    length = int(fs*seconds)
    stride = int(length*stride_rate)
    loc = 0
    while loc+length <= x.shape[1]:
        temp.append(x[:,loc:loc+length,:])
        loc=loc+stride
    return np.vstack(temp)


#%%
def sample_generator(XData, YLabel, X_transform=None, Y_transform=None, 
                     seg_len=128, window=(0,2000), batchsize=128):
    '''
    XData: data to slice from  (session, time, channel)
    YLabel: label score (not encoded yet)
    window: if tuple/list of length 2, will random sample within the range of (low, high)
            if list of length>2, will just random sample the contained elements
    '''
    # num_cls = YLabel.shape[1]
    X = np.zeros((batchsize, seg_len, XData.shape[-1]))
    # Y = np.zeros(batchsize)
    
    label_pos = np.unique(YLabel)
    
    Type_idx = [np.where(YLabel == i)[0] for i in label_pos]

    while True:
        Y = np.zeros(batchsize)  #for the problem Y‘dimension is increasing every time after transformation per interation
        if len(window) == 2:
            start_idx = np.random.randint(window[0], window[1]-seg_len, batchsize)
        else:
            start_idx = random.sample(window, batchsize)
            
        for i, idx in enumerate(start_idx):
            dice = np.random.randint(0, len(label_pos))
            trial_num = np.random.randint(0, len(Type_idx[dice]))
            selected = Type_idx[dice][trial_num]         
            # if dice:
            #     trial_num = np.random.randint(0, len(T_idx), 1)
            #     selected = T_idx[trial_num]

            # else:
            #     trial_num = np.random.randint(0, len(F_idx), 1)
            #     selected = F_idx[trial_num]
           
            X[i] = XData[selected, idx:idx+seg_len, :]
            Y[i] = YLabel[selected]
        
        if Y_transform != None:
            YY = Y_transform(Y)
            if X_transform != None:
                XX = X_transform(X)
                yield XX, YY
            else:
                yield X, YY
        else:
            if X_transform != None:
                XX = X_transform(X)
                yield XX, Y
            else:
                yield X, Y
 
def val_gen(XData, YLabel, X_transform=None, Y_transform = None, seg_len=128, window=(0,2000), batchsize=128):
    # num_cls = YLabel.shape[1]
    X = np.zeros((batchsize, seg_len, XData.shape[-1]))
    Y = np.zeros(batchsize)
    
    label_pos = np.unique(YLabel)
        
    Type_idx = [np.where(YLabel == i)[0] for i in label_pos]

    if len(window) == 2:
        start_idx = np.random.randint(window[0], window[1]-seg_len, batchsize)
    else:
        start_idx = random.sample(window, batchsize)

    for i, idx in enumerate(start_idx):
        dice = np.random.randint(0, len(label_pos))
        trial_num = np.random.randint(0, len(Type_idx[dice]))
        selected = Type_idx[dice][trial_num]         
        
        X[i] = XData[selected, idx:idx+seg_len, :]
        Y[i] = YLabel[selected]

    if Y_transform != None:
        YY = Y_transform(Y)
        if X_transform != None:
            XX = X_transform(X)
            return XX, YY
        else:
            return X, YY
    else:
        if X_transform != None:
            XX = X_transform(X)
            return XX, Y
        else:
            return X, Y

#%%    
def iter_gen_mxp(XData, YLabel, X_transform=None, Y_transform = None, seg_len=128, 
                window=(0,2000), n_per_cls = 3, mix_between_class=4, beta=0.5,
                weight_mode=None, shuffle=False):
    '''
    For augmentation with the mixup method
    '''
    # num_cls = YLabel.shape[1]
     
    label_pos = np.unique(YLabel)   # number of unique labels
    Type_idx = [np.where(YLabel == i)[0] for i in label_pos] # trial index wrt each label
    
    n_cls = len(label_pos)
    batchsize = n_cls*n_per_cls + (n_cls-1)*mix_between_class*n_per_cls

    X = np.zeros((batchsize, seg_len, XData.shape[-1]))
    sample_weights = np.ones((batchsize, 1))
    
    if isinstance(weight_mode, float):
        sample_weights[n_per_cls*n_cls:] = weight_mode
            
    while True:
        Y = np.zeros(batchsize)  
        if len(window) == 2:
            start_idx = np.random.randint(window[0], window[1]-seg_len, n_per_cls)
        else:
            start_idx = random.sample(window, n_per_cls)    
        
        j=0 # the end of filled 
        for i in range(n_cls):
            trial_num = np.random.randint(0, len(Type_idx[i]))
            selected = Type_idx[i][trial_num] 
            
            for idx in start_idx:
                X[j] = XData[selected, idx:idx+seg_len, :]
                Y[j] = YLabel[selected]
                j = j+1
        
        '''
        The mixup part
        '''
        # end_of_not_augmented = j.copy()
        # _lambda = np.random.beta(beta,beta, size=mix_between_class)
        idx_shuffled = np.arange(batchsize)
        if shuffle:
            np.random.shuffle(idx_shuffled)
        
        if Y_transform != None:
            YY = Y_transform(Y)
            for i in range(n_cls-1):
                start = i*n_per_cls
                end = (i+1)*n_per_cls
                # _lambda = np.random.beta(beta,beta, size=mix_between_class)
                # for _l in _lambda:
                #     X[j:j+n_per_cls] = (1-_l)*X[start:end] + _l*X[start+n_per_cls:end+n_per_cls]            
                #     YY[j:j+n_per_cls] = (1-_l)*YY[start:end] + _l*YY[start+n_per_cls:end+n_per_cls]
                #     if weight_mode == 'auto':
                #         sample_weights[j:j+n_per_cls] = max(_l, 1-_l)                    
                #     j = j+n_per_cls
                
                for _ in range(mix_between_class):
                    _lambda = np.random.beta(beta,beta, size=[n_per_cls,1])
                    X[j:j+n_per_cls] = (1-_lambda[...,None])*X[start:end] + _lambda[...,None]*X[start+n_per_cls:end+n_per_cls]            
                    if len(YY.shape) > 1:
                        YY[j:j+n_per_cls] = (1-_lambda)*YY[start:end] + _lambda*YY[start+n_per_cls:end+n_per_cls]
                    else:
                        YY[j:j+n_per_cls] = (1-_lambda[:,0])*YY[start:end] + _lambda[:,0]*YY[start+n_per_cls:end+n_per_cls]
                    
                    if weight_mode == 'auto':
                        sample_weights[j:j+n_per_cls] = np.maximum(_lambda, 1-_lambda)                    
                    j = j+n_per_cls
            if X_transform != None:
                XX = X_transform(X)
                yield XX[idx_shuffled], YY[idx_shuffled], sample_weights[idx_shuffled]
            else:
                yield X[idx_shuffled], YY[idx_shuffled], sample_weights[idx_shuffled]
        else:
            for i in range(n_cls-1):
                start = i*n_per_cls
                end = (i+1)*n_per_cls
                # _lambda = np.random.beta(beta,beta, size=mix_between_class)
                # for _l in _lambda:
                #     X[j:j+n_per_cls] = (1-_l)*X[start:end] + _l*X[start+n_per_cls:end+n_per_cls]            
                #     Y[j:j+n_per_cls] = (1-_l)*Y[start:end] + _l*Y[start+n_per_cls:end+n_per_cls]
                #     if weight_mode == 'auto':
                #         sample_weights[j:j+n_per_cls] = max(_l, 1-_l)                          
                #     j = j+n_per_cls     
                for _ in range(mix_between_class):
                    _lambda = np.random.beta(beta,beta, size=[n_per_cls,1])
                    X[j:j+n_per_cls] = (1-_lambda[...,None])*X[start:end] + _lambda[...,None]*X[start+n_per_cls:end+n_per_cls]            
                    Y[j:j+n_per_cls] = (1-_lambda[...,0])*Y[start:end] + _lambda[...,0]*Y[start+n_per_cls:end+n_per_cls]
                    if weight_mode == 'auto':
                        sample_weights[j:j+n_per_cls] = np.maximum(_lambda, 1-_lambda)                    
                    j = j+n_per_cls             
            if X_transform != None:
                XX = X_transform(X)
                yield XX[idx_shuffled], Y[idx_shuffled], sample_weights[idx_shuffled]
            else:
                yield X[idx_shuffled], Y[idx_shuffled], sample_weights[idx_shuffled]

    
def fix_gen_mxp(XData, YLabel, X_transform=None, Y_transform = None, seg_len=128, 
                window=(0,2000), n_per_cls = 3, mix_between_class=4, beta=0.5):
    '''
    For augmentation with the mixup method
    '''
    # num_cls = YLabel.shape[1]
     
    label_pos = np.unique(YLabel)   # number of unique labels
    Type_idx = [np.where(YLabel == i)[0] for i in label_pos] # trial index wrt each label
    
    n_cls = len(label_pos)
    batchsize = n_cls*n_per_cls + (n_cls-1)*mix_between_class*n_per_cls

    X = np.zeros((batchsize, seg_len, XData.shape[-1]))
    Y = np.zeros(batchsize)      

    if len(window) == 2:
        start_idx = np.random.randint(window[0], window[1]-seg_len, n_per_cls)
    else:
        start_idx = random.sample(window, n_per_cls)    
    
    j=0 # the end of filled 
    for i in range(n_cls):
        trial_num = np.random.randint(0, len(Type_idx[i]))
        selected = Type_idx[i][trial_num] 
        
        for idx in start_idx:
            X[j] = XData[selected, idx:idx+seg_len, :]
            Y[j] = YLabel[selected]
            j = j+1
    
    '''
    The mixup part
    '''
    # end_of_not_augmented = j.copy()
    _lambda = np.random.beta(beta,beta, size=mix_between_class)
    idx_shuffled = np.arange(batchsize)
    np.random.shuffle(idx_shuffled)
        
    if Y_transform != None:
        YY = Y_transform(Y)
        for i in range(n_cls-1):
            start = i*n_per_cls
            end = (i+1)*n_per_cls
            for _l in _lambda:
                X[j:j+n_per_cls] = (1-_l)*X[start:end] + _l*X[start+n_per_cls:end+n_per_cls]            
                YY[j:j+n_per_cls] = (1-_l)*YY[start:end] + _l*YY[start+n_per_cls:end+n_per_cls]                        
                j = j+n_per_cls
        if X_transform != None:
            XX = X_transform(X)
            return XX[idx_shuffled], YY[idx_shuffled]
        else:
            return X[idx_shuffled], YY[idx_shuffled]
    else:
        for i in range(n_cls-1):
            start = i*n_per_cls
            end = (i+1)*n_per_cls
            for _l in _lambda:
                X[j:j+n_per_cls] = (1-_l)*X[start:end] + _l*X[start+n_per_cls:end+n_per_cls]            
                Y[j:j+n_per_cls] = (1-_l)*Y[start:end] + _l*Y[start+n_per_cls:end+n_per_cls]                        
                j = j+n_per_cls                  
        if X_transform != None:
            XX = X_transform(X)
            return XX[idx_shuffled], Y[idx_shuffled]
        else:
            return X[idx_shuffled], Y[idx_shuffled]

#%%
'''
 The following generator for the hybrid case
'''
# def hybrid_sample_generator(XData, YLabel, Y_transform=[None,None], seg_len=128, window=(0,2000), batchsize=128):
#     '''
#     XData: data to slice from  (session, time, channel)
#     YLabel: label score (not encoded yet)
#     '''
#     # num_cls = YLabel.shape[1]
#     X = np.zeros((batchsize, seg_len, XData.shape[-1]))
#     # Y = np.zeros(batchsize)
    
#     label_pos = np.unique(YLabel)
    
#     Type_idx = [np.where(YLabel == i)[0] for i in label_pos]

#     # T_idx = np.where(label_pos == 1)[0]
#     # F_idx = np.where(label_pos == 0)[0]

#     while True:
#         Y = np.zeros(batchsize)  #for the problem Y‘dimension is increasing every time after transformation per interation
#         start_idx = np.random.randint(window[0], window[1]-seg_len, batchsize)
#         for i, idx in enumerate(start_idx):
#             dice = np.random.randint(0, len(label_pos))
#             trial_num = np.random.randint(0, len(Type_idx[dice]))
#             selected = Type_idx[dice][trial_num]         
#             # if dice:
#             #     trial_num = np.random.randint(0, len(T_idx), 1)
#             #     selected = T_idx[trial_num]

#             # else:
#             #     trial_num = np.random.randint(0, len(F_idx), 1)
#             #     selected = F_idx[trial_num]
           
#             X[i] = XData[selected, idx:idx+seg_len, :]
#             Y[i] = YLabel[selected]
        
#         yield X, [ _transform(Y) if _transform != None else Y for _transform in Y_transform ]
    
 
# def hybrid_val_gen(XData, YLabel, Y_transform = [None,None], seg_len=128, window=(0,2000), batchsize=128):
#     # num_cls = YLabel.shape[1]
#     X = np.zeros((batchsize, seg_len, XData.shape[-1]))
#     Y = np.zeros(batchsize)
    
#     label_pos = np.unique(YLabel)
        
#     Type_idx = [np.where(YLabel == i)[0] for i in label_pos]

#     start_idx = np.random.randint(window[0], window[1]-seg_len, batchsize)
#     for i, idx in enumerate(start_idx):
#         dice = np.random.randint(0, len(label_pos))
#         trial_num = np.random.randint(0, len(Type_idx[dice]))
#         selected = Type_idx[dice][trial_num]         
        
#         X[i] = XData[selected, idx:idx+seg_len, :]
#         Y[i] = YLabel[selected]

#     return X, [ trans(Y) if trans != None else Y for trans in Y_transform ]


#%%
'''
Preparing train/val/test
'''
def data_split_10(fold_ind, time_len):
    '''
    test/train/val --- fold_ind=0
    train/val/test --- fold_ind=9
    train/val/test/train --- other case

    128Hz
    '''
    each_fold_len = time_len//10
    split_ind = np.arange(0, time_len, each_fold_len)
    test_ind = (split_ind[fold_ind], split_ind[fold_ind]+each_fold_len)
    val_ind = (split_ind[fold_ind-1], split_ind[fold_ind-1]+each_fold_len)

    if fold_ind == 0:
        train_ind = (test_ind[-1], val_ind[0])
    elif fold_ind == 9:
        train_ind = (0, val_ind[0])
    else:
        part_1 = [i for i in np.arange(0, val_ind[0]-128)]
        part_2 = [i for i in np.arange(test_ind[-1], time_len-128)]
        train_ind = part_1 + part_2

    return train_ind, val_ind, test_ind

'''
no validation case
'''
def data_split_10_noval(fold_ind, time_len):
    '''
    test/train --- fold_ind=0
    train/test --- fold_ind=9
    train/test/train --- other case

    128Hz
    '''
    each_fold_len = time_len//10
    split_ind = np.arange(0, time_len, each_fold_len)
    test_ind = (split_ind[fold_ind], split_ind[fold_ind]+each_fold_len)
    
    if fold_ind == 0:
        train_ind = (test_ind[-1], time_len)
    elif fold_ind == 9:
        train_ind = (0, test_ind[0])
    else:
        part_1 = [i for i in np.arange(0, test_ind[0]-128)]
        part_2 = [i for i in np.arange(test_ind[-1], time_len-128)]
        train_ind = part_1 + part_2

    return train_ind, test_ind

#%%
'''
Embedding 1d data into 2d sparse matrix
'''
def Embed_DREAMER(X,seglen=128):
    '''
    Embed the 14*128 format into a 9*9*128 format
    Channels are 
    AF3: (1,3) 
    F7: (2,0) 
    F3: (2,2) 
    FC5: (3,1) 
    T7: (4,0)
    P7: (6,0)
    O1: (8,3)
    O2: (8,5)
    P8: (6,8)
    T8: (4,8) 
    FC6: (3,7) 
    F4: (2,6) 
    F8: (2,8) 
    AF4: (1,5)

    X: (batch, 14, 128)
    return : (batch, 9, 9, 128)
    '''
    assert len(X.shape) == 3
    M = np.zeros([X.shape[0], 9,9,seglen])
    loc_dict = {
        0: (1,3),
        1: (2,0),
        2: (2,2),
        3: (3,1),
        4: (4,0),
        5: (6,0),
        6: (8,3),
        7: (8,5),
        8: (6,8),
        9: (4,8),
        10: (3,7),
        11: (2,6),
        12: (2,8),
        13: (1,5)
    }
    for k, v in loc_dict.items():
        M[:,v[0],v[1],:] = X[:,k]
    return M

def Embed_DEAP(X, subject, seglen=128):
    '''
    Embed the 32*128 format into a 9*9*128 format
    Channels are 
    FP1: (0,3)
    AF3: (1,3) 
    F7: (2,0) 
    F3: (2,2) 
    FC5: (3,1) 
    FC1: (3,3)
    T7: (4,0)
    C3: (4,2)
    CP5: (5,1)
    CP1: (5,3)
    P7: (6,0)
    P3: (6,2)
    PO3: (7,3)
    O1: (8,3)
    O2: (8,5)
    PO4: (7,5)
    P4: (6,6)
    P8: (6,8)
    CP6: (5,7)
    CP2: (5,5)
    T8: (4,8) 
    C4: (4,6)
    FC6: (3,7)
    FC2: (3,5) 
    F4: (2,6) 
    F8: (2,8) 
    AF4: (1,5)
    FP2: (0,5)
    FZ: (2,4)
    CZ: (4,4)
    Pz: (6,4)
    Oz: (8,4)

    note:     ch_list = 
              {'Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 
               'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 
               'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4',
               'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 
               'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2',
               'Fz', 'Cz'
               };  % DEAP: 1-22

    ch_list = {'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 
               'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 
               'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 
               'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 
               'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8',
               'PO4', 'O2'
               };  % DEAP: 23-32


    X: (batch, 32, 128)
    return : (batch, 9, 9, 128)
    '''
    assert len(X.shape) == 3
    M = np.zeros([X.shape[0], 9,9,seglen])
    if subject<23:
        loc_dict = {
            0: (0,3),
            1: (1,3),
            2: (2,0),
            3: (2,2),
            4: (3,3),
            5: (3,1),
            6: (4,0),
            7: (4,2),
            8: (5,3),
            9: (5,1),
            10: (6,0),
            11: (6,2),
            12: (6,4),
            13: (7,3),
            14: (8,3),
            15: (8,4),
            16: (8,5),
            17: (7,5),
            18: (6,6),
            19: (6,8),
            20: (5,7),
            21: (5,5),
            22: (4,6),
            23: (4,8),
            24: (3,7),
            25: (3,5),
            26: (2,6),
            27: (2,8),
            28: (1,5),
            29: (0,5),
            30: (2,4),
            31: (4,4) 
        }
    else:
        loc_dict = {
            0: (0,3),
            1: (1,3),
            3: (2,0),
            2: (2,2),
            5: (3,3),
            4: (3,1),
            7: (4,0),
            6: (4,2),
            9: (5,3),
            8: (5,1),
            11: (6,0),
            10: (6,2),
            12: (7,3),
            13: (8,3),
            14: (8,4),
            15: (6,4),
            16: (0,5),
            17: (1,5),
            18: (2,4),
            19: (2,6),
            20: (2,8),
            21: (3,7),
            22: (3,5),
            23: (4,4),
            24: (4,6),
            25: (4,8),
            26: (5,7),
            27: (5,5),
            28: (6,6),
            29: (6,8),
            30: (7,5),
            31: (8,5) 
        }

    for k, v in loc_dict.items():
        M[:,v[0],v[1],:] = X[:,k]
    return M

#%%
def make_segs(data, seg_len, stride):
    t_len = data.shape[1]
    segs = np.stack([data[:,i*stride:i*stride+seg_len,:] for i in range(t_len//stride) if i*stride+seg_len<=t_len], axis= 1)
    # print(segs.shape)
    return segs.reshape((-1, seg_len, data.shape[-1]))

def label_encoding(Y):
    '''
    throw some random noise to the label
    '''
    # return 0.5*(Y - 2)
    # noise = np.random.normal(0, 0.1, len(Y))
    # noise = np.random.uniform(-0.1, 0.1, len(Y))
    # scale = 0.2
    # low = -0.2
    # high = 0.2
    # noise = truncnorm(low/scale, high/scale, loc=0, scale=scale).rvs(len(Y))

    # return 0.5*(Y-2) + noise
    return 0.5*(Y-2)

def label_noise(Y):
    scale = 0.1
    low = -0.2
    high = 0.2
    noise = truncnorm(low/scale, high/scale, loc=0, scale=scale).rvs(len(Y))

    return Y+noise

