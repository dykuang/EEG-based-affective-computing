'''
Network Models For Exploration
'''
#%%
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow_probability as tfp
from Modules import *
from tensorflow.keras.constraints import max_norm
#%%
'''
For dealing with last layer
'''
class Out_choice(layers.Layer):
    def __init__(self, LabelMode, nb_classes, Prob_model=None, **kwargs):
        super(Out_choice, self).__init__(**kwargs)
        self.LabelMode = LabelMode
        self.ncls = nb_classes
       
        # Probability type
        if Prob_model != None:
            self.Prob_model = tfp.layers.DistributionLambda(Prob_model)
        
            if self.LabelMode == 'Mixture':
                self.dense  = layers.Dense(self.ncls*3, activation='linear', 
                                            name = 'last_dense')
            elif self.LabelMode == 'Gaussian':
                self.dense  = layers.Dense(2, activation='linear', 
                                        name = 'last_dense')
            elif self.LabelMode == 'Gaussian_2':
                self.dense  = layers.Dense(5, activation='linear', 
                                        name = 'last_dense')
        # Deterministic type 
        else:   
            self.Prob_model = None                            
            if self.LabelMode in ['OneHot', 'OneHot_N', 'OneHot_Mixup']:
                self.dense  = layers.Dense(self.ncls, 
                                            # kernel_constraint = max_norm(norm_rate), 
                                            name = 'last_dense')
                self.activation = layers.Activation('softmax', name = 'softmax')
                # self.activation = layers.Activation('linear', name = 'linear')

            elif self.LabelMode == 'OneHot_b':
                self.dense  = layers.Dense(2, 
                                            # kernel_constraint = max_norm(norm_rate), 
                                            name = 'last_dense')
                self.activation = layers.Activation('softmax', name = 'softmax')

            elif self.LabelMode == 'Regression':
                self.dense  = layers.Dense(1, 
                                            # kernel_constraint = max_norm(norm_rate), 
                                            name = 'last_dense')
                self.activation = layers.Activation('tanh', name = 'tanh')       
       
    def __call__(self, X):
        out = self.dense(X)
        if self.Prob_model != None or self.LabelMode in ['Mixture', 'Gaussian', 'Gaussian_2']:
            return self.Prob_model(out)
        else:
            return self.activation(out)

#%%
'''
The MSBAM
'''
def MSBAM(input_shape = (9,9,128,1), Nclass = 2, LabelMode='OneHot', 
          Prob_model=None, Droprate=0.7):
    '''
    input_shape: 9 by 9 by 128
    '''
    time_len = input_shape[-1]
    x_in = layers.Input(shape = input_shape)
    
    ##################################################
    '''The spatial feature extractor'''
    x_s = layers.Conv3D(16, (1,1,time_len), padding='same',name='spat1')(x_in)
    x_s = layers.Activation('selu')(x_s)
    x_s = layers.Lambda(lambda x: x[...,0,:])(x_s)
    
    for i in range(2):
        x_s = layers.Conv2D(16, (3,3), padding='same', name='spat{}'.format(i+2))(x_s)
        # x_s = layers.SeparableConv2D(16, (3,3), name='spat{}'.format(i+2))(x_s)
        x_s = layers.Activation('selu')(x_s)
    
    x_s = layers.Flatten()(x_s)
    x_s = layers.Dense(40, activation='linear', name='Sdense')(x_s)
    x_s = layers.BatchNormalization()(x_s)
    x_s = layers.Activation('softmax')(x_s)
    ###################################################
    '''The temporal feature extractor'''

    '''Split left - right hemisphere'''
    x_left = layers.Lambda(lambda x: x[...,:5,:,:])(x_in)
    # x_right = layers.Lambda(lambda x: tf.reverse(x[...,5:,:,:],axis=-3))(x_in)
    x_right = layers.Lambda(lambda x: x[...,4:,:,:][...,::-1,:,:])(x_in) #flip

    # print(x_left.shape)
    # print(x_right.shape)

    '''First Branch'''
    t1_shared_conv = layers.Conv3D(16, (9,5,128), strides = (9,5,64), padding='valid',
                                activation='selu', name='Tconv1')
    # t1_shared_conv = layers.SeparableConv3D(16, (9,5,128), strides = (9,5,64), 
    #                             activation='selu', name='Tconv1')
    x_t1_left = t1_shared_conv(x_left)
    x_t1_right = t1_shared_conv(x_right)
    x_t1_diff = layers.Add()([x_t1_left, -x_t1_right])
    
    x_t1_fuse = layers.Concatenate(axis=-1, name='t1_fuse')([x_t1_left, x_t1_diff, x_t1_right])
    x_t1_flatten = layers.Flatten()(x_t1_fuse)
    x_t1 = layers.Dense(40, activation='linear', name='T1dense')(x_t1_flatten)

    '''Second Branch'''
    t2_shared_conv = layers.Conv3D(16, (9,5,64), strides = (9,5,32), padding='valid',
                                activation='selu', name='Tconv2')
    # t2_shared_conv = layers.SeparableConv3D(16, (9,5,64), strides = (9,5,32), 
    #                             activation='selu', name='Tconv2')
    x_t2_left = t2_shared_conv(x_left)
    x_t2_right = t2_shared_conv(x_right)
    x_t2_diff = layers.Add()([x_t2_left, -x_t2_right])
    
    x_t2_fuse = layers.Concatenate(axis=-1, name='t2_fuse')([x_t2_left, x_t2_diff, x_t2_right])
    x_t2_flatten = layers.Flatten()(x_t2_fuse)
    x_t2 = layers.Dense(40, activation='linear', name='T2dense')(x_t2_flatten)

    # ''' Make concatenation'''
    # x_t = layers.Concatenate(axis=-1)([x_t1, x_t2])
    ##################################################################
    ''' Last classification '''
    x_f = layers.Concatenate(axis=-1, name='final_feature')([x_s, x_t1, x_t2])
    x_f = layers.Dropout(Droprate)(x_f)
    # x_out = layers.Dense(Nclass, activation='softmax', name='cls')(x_f)
    x_out = Out_choice(LabelMode, Nclass, Prob_model)(x_f)

    return Model(x_in, x_out)

#%%
'''
EEGNet typed model 
'''

def EEGNet(Prob_model, nb_classes, LabelMode, Chans = 64, Samples = 128, 
           dropoutRate = 0.5, kernLength = 64, F1 = 8, activation='swish',
           D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout', **kwargs):
   
    if dropoutType == 'SpatialDropout2D':
        dropoutType = layers.SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = layers.Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = layers.Input(shape = (Samples, Chans, 1))
    block1       = layers.Conv2D(F1, (kernLength, 1), padding = 'same',
                                   use_bias = False, name = 'Conv2D')(input1)

    # block1       = R_corr(name = 'Rcorr', corr_passed=np.zeros((8,8)).astype(np.float32))(block1)
    # block1       = R_corr(name = 'Rcorr', corr_passed=None)(block1)
    
    block1       = layers.BatchNormalization(axis = -1, name='BN-1')(block1)  # normalization on channels or time
    block1       = layers.DepthwiseConv2D((1, Chans), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.),
                                   name = 'DepthConv')(block1)
    block1       = layers.BatchNormalization(axis = -1, name = 'BN-2')(block1)
    block1       = layers.Activation(activation)(block1)

    block1       = layers.AveragePooling2D((2, 1))(block1)
    block1       = dropoutType(dropoutRate)(block1) 
    
    block2       = layers.SeparableConv2D(F2, (5, 1),
                                   use_bias = False, padding = 'same',
                                    name = 'SepConv-1')(block1)

    # block2       = R_corr(name = 'Rcorr')(block2)
    # block2         = attach_attention_module(block2, 'se_block', ratio=4)
    # block2       = layers.Lambda(lambda x: x[...,0,:])(block2)
    # block2       = K_attention(name = 'Katt0')(block2, offdiag_mask=False)
    # # # # # block2       = C_attention(name = 'Catt')(block2, use_mask=False)
    # # # # # block2       = K_attention_MH(num_heads=2, name='Katt')(block2)
    # # # # # block2       = qKv_attention(dim=16, num_heads=4, dropout_rate=0.003,
    # # # #                             #  name='Katt')(block2,kernel='Linear',use_mask=False)
    # block2       = layers.Lambda(lambda x: x[...,None,:])(block2)

    block2       = layers.BatchNormalization(axis = -1, name = 'BN-3')(block2)
    block2       = layers.Activation(activation)(block2)
    block2       = layers.AveragePooling2D((2, 1))(block2)
    # block2       = tfa.layers.AdaptiveAveragePooling2D((32, 1))(block2)   #will pool to fixed size
    block2       = dropoutType(dropoutRate)(block2)


    ###############################################################
    # block2       = layers.SeparableConv2D(F2, (5, 1),
    #                                use_bias = False, padding = 'same',
    #                                 name = 'SepConv-2')(block2)

    # # block2       = R_corr(name = 'Rcorr')(block2)
    # # block2         = attach_attention_module(block2, 'se_block', ratio=4)
    # # block2       = layers.Lambda(lambda x: x[...,0,:])(block2)
    # # # block2       = layers.Permute((2,1))(block2)
    # # block2       = K_attention(name = 'Katt')(block2, offdiag_mask=False)
    # # # # # # block2       = C_attention(name = 'Catt')(block2, use_mask=False)
    # # # # # # block2       = K_attention_MH(num_heads=2, name='Katt')(block2)
    # # # # # # block2       = qKv_attention(dim=16, num_heads=4, dropout_rate=0.003,
    # # # # #                             #  name='Katt')(block2,kernel='Linear',use_mask=False)
    # # # block2       = layers.Permute((2,1))(block2)
    # # block2       = layers.Lambda(lambda x: x[...,None,:])(block2)

    # block2       = layers.BatchNormalization(axis = -1, name = 'BN-4')(block2)
    # block2       = layers.Activation(activation)(block2)
    # block2       = layers.AveragePooling2D((2, 1))(block2)
    # # block2       = tfa.layers.AdaptiveAveragePooling2D((16, 1))(block2)  #will pool to fixed size
    # block2       = dropoutType(dropoutRate)(block2)
    ######################################################################
       
    flatten      = layers.Flatten(name = 'flatten')(block2)
    
    output = Out_choice(LabelMode, nb_classes, Prob_model)(flatten)
    # if LabelMode == 'Mixture':
    #     dense        = layers.Dense(nb_classes*3, activation='linear', 
    #                                 name = 'last_dense')(flatten)
    #     output       = tfp.layers.DistributionLambda(Prob_model)(dense)
    # if LabelMode == 'Gaussian':
    #     dense        = layers.Dense(2, activation='linear', 
    #                                 name = 'last_dense')(flatten)
    #     output       = tfp.layers.DistributionLambda(Prob_model)(dense)

    # elif LabelMode == 'OneHot':
    #     dense        = layers.Dense(nb_classes, 
    #                                 kernel_constraint = max_norm(norm_rate), 
    #                                 name = 'last_dense')(flatten)
    #     output       = layers.Activation('softmax', name = 'softmax')(dense)
    # elif LabelMode == 'OneHot_b':
    #     dense        = layers.Dense(2, 
    #                                 kernel_constraint = max_norm(norm_rate), 
    #                                 name = 'last_dense')(flatten)
    #     output       = layers.Activation('softmax', name = 'softmax')(dense)

    # elif LabelMode == 'Regression':
    #     dense        = layers.Dense(1, 
    #                                 # kernel_constraint = max_norm(norm_rate), 
    #                                 name = 'last_dense')(flatten)
    #     output       = layers.Activation('tanh', name = 'tanh')(dense)
    #     # output       = Act_tanh(b=1.2, name='tanh', learnable=True)(dense)
    #     # output       = layers.Dense(1, activation = 'linear',
    #     #                             # kernel_constraint = max_norm(norm_rate), 
    #     #                             name = 'last_dense')(flatten)

    Mymodel      = Model(inputs=input1, outputs=output)
        
    return Mymodel

#%%
def KAMNet(Prob_model, nb_classes, LabelMode, Chans = 64, Samples = 128, 
           dropoutRate = 0.5, kernLength = 64, F1 = 8, activation='swish',
           D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout', **kwargs):
   
    if dropoutType == 'SpatialDropout2D':
        dropoutType = layers.SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = layers.Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = layers.Input(shape = (Samples, Chans, 1))
    block1       = layers.Conv2D(F1, (kernLength, 1), padding = 'same',
                                   use_bias = False, name = 'Conv2D')(input1)

    # block1       = R_corr(name = 'Rcorr', corr_passed=np.zeros((8,8)).astype(np.float32))(block1)
    # block1       = R_corr(name = 'Rcorr', corr_passed=None)(block1)
    
    block1       = layers.BatchNormalization(axis = -1, name='BN-1')(block1)  # normalization on channels or time
    block1       = layers.DepthwiseConv2D((1, Chans), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.),
                                   name = 'DepthConv')(block1)
    block1       = layers.BatchNormalization(axis = -1, name = 'BN-2')(block1)
    block1       = layers.Activation(activation)(block1)

    block1       = layers.AveragePooling2D((2, 1))(block1)
    block1       = dropoutType(dropoutRate)(block1) 
    
    block2       = layers.SeparableConv2D(F2, (5, 1),
                                   use_bias = False, padding = 'same',
                                    name = 'SepConv-1')(block1)

    # block2       = R_corr(name = 'Rcorr')(block2)
    # block2         = attach_attention_module(block2, 'se_block', ratio=4)
    block2       = layers.Lambda(lambda x: x[...,0,:])(block2)
    block2       = K_attention(name = 'Katt')(block2, offdiag_mask=False)
    # # # # block2       = C_attention(name = 'Catt')(block2, use_mask=False)
    # # # # block2       = K_attention_MH(num_heads=2, name='Katt')(block2)
    # # # # block2       = qKv_attention(dim=16, num_heads=4, dropout_rate=0.003,
    # # #                             #  name='Katt')(block2,kernel='Linear',use_mask=False)
    block2       = layers.Lambda(lambda x: x[...,None,:])(block2)

    block2       = layers.BatchNormalization(axis = -1, name = 'BN-3')(block2)
    block2       = layers.Activation(activation)(block2)
    block2       = layers.AveragePooling2D((2, 1))(block2)
    # block2       = tfa.layers.AdaptiveAveragePooling2D((32, 1))(block2)   #will pool to fixed size
    block2       = dropoutType(dropoutRate)(block2)
       
    flatten      = layers.Flatten(name = 'flatten')(block2)
    
    output = Out_choice(LabelMode, nb_classes, Prob_model)(flatten)
    Mymodel      = Model(inputs=input1, outputs=output)
        
    return Mymodel

#%%
'''
A model for incorporating priors of sensor clusers
'''
class Region_collapse(layers.Layer):
    '''
    (B,T,R)->(B,T,1) by weighted average
    '''
    def __init__(self, numC, **kwargs):
        super(Region_collapse, self).__init__( **kwargs)
        self.pool = layers.GlobalAveragePooling1D()
        self.numC = numC
        self.att  = layers.Dense(self.numC, activation = 'softmax')
        
    def call(self, X):
        X_pooled = self.pool(X) # (B, C)
        
        W = tf.expand_dims(self.att(X_pooled), -1) # (B, C, 1)
        
        return X @ W # (B, T, C) @ (B, C, 1) = (B, T, 1)
    

class Region_fusion(layers.Layer):
    '''
    group signals according to regions 
    two choices of group weight.
        * dense for all, then softmax per region
        * dense per region with softmax
    '''
    def __init__(self, loc=[0,4,7,10,14], **kwargs):
        super(Region_fusion, self).__init__(**kwargs)
        self.pool = layers.GlobalAveragePooling1D()
        self.loc = loc
        self.n_per_region = [loc[i+1] - loc[i] for i in range(len(loc)-1)]
        self.n_regions = len(self.n_per_region)

        self.att = [layers.Dense(_n, 
                                #  name = '{}{}'.format(name, i), 
                                 activation = 'softmax'
                                 ) for i, _n in enumerate(self.n_per_region)]
        
    def call(self, X):
        '''
        input: 
            X: (B ,T , C)
            loc: a list, locations to make partition, must start from 0
        output: [B, T, RC], RC number of regions 
        '''
        # X_pooled = tf.squeeze(self.pool(X)) # (B, C)
        X_pooled = self.pool(X) # (B, 1, C)
        assert self.loc[0] == 0, 'loc must starts with 0'
        assert self.loc[-1] == X.shape[-1], 'loc must ends with the same channels of X'
        # assert len(self.loc) == self.n_regions, 'loc length must be region numbers + 1'
        
        X_R = [X_pooled[:,self.loc[i]:self.loc[i+1]] for i in range(self.n_regions) ] # make partition
        print(X_R)

        W_R = [tf.expand_dims(self.att[i](X_R[i]), axis=-1) for  i in range(self.n_regions)] # get learned weights
                
        return tf.concat([ X[...,self.loc[i]:self.loc[i+1]] @ W_R[i] for i in range(self.n_regions)  ], axis=-1)     


class Grouped_Separable1D(layers.Layer):
    '''
    Perform different pointwise convolution on subgroups defined for channels 
    '''
    def __init__(self, loc = [0,4,7,10,14], knum=1, ksize=5, stride = 1, 
                 activation='softmax', **kwargs):
        super(Grouped_Separable1D, self).__init__(**kwargs)
        self.loc = loc
        self.ksize = ksize
        self.stride = stride
        self.knum = knum
        self.group_conv = [layers.SeparableConv1D(self.knum, self.ksize, self.stride, 
                                                  activation = activation,
                                                  padding = 'same',
                                                  pointwise_constraint = max_norm(1.), # 
                                                  use_bias=False, 
                                                #   name='{}{}'.format(name, i),
                                                  **kwargs
                                                  ) for i in range(len(loc)-1)]
        # self.group_conv = [layers.Conv1D(self.c_multiplier, self.ksize, self.stride, 
        #                                  activation = 'softmax',
        #                                 use_bias=False, **kwargs) for _ in range(len(loc)-1)]
    def call(self, X):
        '''
        input:
            X: (B,T,C)
        output:
            shape: (B,T,R) # R is the number of regions
        '''
        x_r = [X[...,self.loc[i]:self.loc[i+1]] for i in range(len(self.loc)-1) ] # make partition
        x_out = [self.group_conv[i](x) for i, x in enumerate(x_r)]
        return tf.concat(x_out, axis=-1)


def normalize_A(A):
    D = np.diag( np.sum(A, axis=1) )  #degree
    D_half = 1/np.sum(A, axis=1, keepdims=True)**0.5
    A_normalized = D_half.transpose()*A*D_half

    return A_normalized

def get_laplacian(A):
    L = np.diag( np.sum(A, axis=1) )  - A 
    max_EV = get_max_eigenvalue(L)
    return L, 2.0*L/max_EV - np.eye(L.shape[0]), max_EV

def get_max_eigenvalue(A):
    u, _ =  np.linalg.eig(A)
    return np.max(np.abs(u)) #only valid for cases where A*=A, hence adjacency matrix is good.

class MessagePassing(layers.Layer):
    '''
    an implementation of message passing among different channels
    (B, T, C) @ (C, C)
    * could be interesting to think about cases such as (C, C1), where C1 < C for aggregation
    '''
    def __init__(self, A, hops, type='regular', **kwargs):
        '''
        A: adjacent matrix
        hops: maximum steps of walk away from a graph node 
        '''
        super(MessagePassing, self).__init__(**kwargs)

        self.A = A
        assert hops > 0, 'hops must be a positive integer'
        self.hops = hops
        self.type = type
        self.A_normalized = normalize_A(A)
        self.Laplacian, self.L_normalized, self.max_L_EV = get_laplacian(A)
        self.poly = self.prepare_poly()       
        self.W = self.add_weight(
                            shape=(self.hops+1,),
                            # initializer= tf.constant_initializer(value=0.01), 
                            # constraint=lambda x: tf.clip_by_value(x, -0.01, 10),
                            trainable=True, 
                            name='PolyG_W',   #This is neccessary for saving weights!!!
                            )  
    # def build(self, input_shape): #somehow self.W not created/tracked during initialization if being putted here
    #     self.W = self.add_weight(
    #                         shape=(self.hops+1,),
    #                         # initializer= tf.constant_initializer(value=0.01), 
    #                         # constraint=lambda x: tf.clip_by_value(x, -0.01, 10),
    #                         trainable=True, name='PolyG_W',
    #                         )

    def prepare_poly(self):
        '''
        type:
            * regular or chebyshev
        '''        
        if self.type == 'regular':
            Poly = [np.eye(self.A.shape[0]), self.A_normalized]
            for i in range(self.hops-1):
                Poly.append(Poly[-1]@self.A_normalized)

        elif self.type == 'chebyshev':
            Poly = [np.eye(self.A.shape[0]), self.L_normalized]
            for i in range(self.hops-1):
                Poly.append(2.0*self.L_normalized@Poly[-1] - Poly[-2])
        
        Poly = np.stack(Poly, axis=-1)
        
        return tf.constant(Poly, tf.float32)
    
    def call(self, X):
        P_filter = tf.einsum('k,ijk->ij',self.W, self.poly)
        
        return X @ P_filter # X is of shape (B, T, C)

class Region_partition(layers.Layer):
    '''
    partitioning the region into several parts
    '''
    def __init__(self, loc, **kwargs):
        super(Region_partition, self).__init__(**kwargs)
        self.loc = loc
        self.num_regions = len(loc) -1 
    def call(self, X):
        X_R = [X[...,self.loc[i]:self.loc[i+1]] for i in range( self.num_regions ) ] # make partition
        return X_R

class Multiple_MP(layers.Layer):
    '''
    perform message passing within each disjoint region
    '''
    def __init__(self, loc, A_list, hop_list, **kwargs):
        '''
        make sure these arguments are compatiable with each other
        '''
        super(Multiple_MP, self).__init__(**kwargs)
        self.loc = loc
        self.A_list = A_list
        self.hop_list = hop_list
        self.num_regions = len(A_list)

    def call(self, X):

        X_R = [X[...,self.loc[i]:self.loc[i+1]] for i in range( self.num_regions ) ] # make partition, could be modified for arbitary selection of nodes instead of continum
        out_list  = [MessagePassing(self.A_list[i], self.hop_list[i])(X_R[i]) for i in range(self.num_regions)]

        return out_list


# class Hierachy(layers.Layer):
#     '''
#     A realization of information separation among node, region and graph levels
#     '''
#     def __init__(self, loc, A_list, hop_list, region_adjM):
#         super(Hierachy, self).__init__()
#         self.loc = loc
#         self.A_list = A_list
#         self.hop_list = hop_list
#         self.region_adjM = region_adjM
#         self.num_regions = len(A_list)

#%%
def build_hierachy(X, mergeMode,
                   loc, A_list, hop_list, region_adjM, region_hop, 
                   poly_type, activation, blockIDX = 0):
    '''
    A realization of information separation among node, region and graph levels
    '''

    '''node level'''    
    X_per_R = Region_partition(loc,name='RP_{}'.format(blockIDX ))(X)
    X_per_R_agg = [MessagePassing(A_list[i], hop_list[i], poly_type,
                            name='NodeMP_{}_{}'.format(blockIDX, i)
                            )(X_per_R[i]) for i in range(len(A_list))]
    #add time convolution here?

    X_agg = [layers.SeparableConv1D(A_list[i].shape[0], 5, 1, 
                                    activation = 'linear',
                                    padding = 'same',
                                    pointwise_constraint = max_norm(1.), # 
                                    use_bias=False, 
                                    name='NodeSP_{}_{}'.format(blockIDX, i)
                                   )(X_per_R_agg[i]) for i in range(len(A_list)) ]
    X_agg = layers.Concatenate(axis=-1, name='N_concate_{}'.format(blockIDX))(X_agg)
    X_agg = layers.BatchNormalization(axis=-1, name='N_BN_{}'.format(blockIDX))(X_agg)
    X_agg = layers.Activation(activation)(X_agg)
    X_out = layers.Add(name='N_Add_{}'.format(blockIDX))([X, X_agg])

    # print(X_out.shape)
    # X_out = layers.BatchNormalization(axis=-1)(X_out)
    # X_out = layers.Activation(activation)

    '''get weighted signal per region'''
    if mergeMode == 'sepconv':
        X_R = Grouped_Separable1D(loc = loc, knum=1, 
                                  ksize=5, stride = 1, name='GS_{}'.format(blockIDX))(X)
        X_R_agg = MessagePassing(region_adjM, region_hop, poly_type,
                                 name='RMP_{}'.format(blockIDX))(X_R)

        X_R_latter = Grouped_Separable1D(loc = loc, knum=1, 
                                         ksize=5, stride = 1, 
                                         name='GS_latter_{}'.format(blockIDX))(X_agg)
    elif mergeMode == 'dense':
        X_R = Region_fusion(loc = loc, name='RF_{}'.format(blockIDX))(X)
        X_R_agg = MessagePassing(region_adjM, region_hop, poly_type,
                                 name='RMP_{}'.format(blockIDX))(X_R)

        X_R_latter = Region_fusion(loc = loc, name='RF_latter_{}'.format(blockIDX))(X_agg)

    
    X_R_out = layers.Add(name='R_Add_{}'.format(blockIDX))([X_R_agg, X_R_latter])
    
    
    '''get weighted signal from region signal to form the global feature'''
    X_G = Region_collapse(len(A_list), name='GRC{}'.format(blockIDX))(X_R)
    X_G_agg = Region_collapse(len(A_list), name='GRC_latter_{}'.format(blockIDX))(X_R_latter)
    X_G_out = layers.Add(name='G_Add_{}'.format(blockIDX))([X_G, X_G_agg])

    return [X_out, X_R_out, X_G_out]


def HST_model(input_shape, LabelMode, nb_classes, Prob_model,
              mergeMode,loc, adj_list, hop_list, region_adjM, region_hop, poly_type,
              activation, droprate, F1=8, ksize=5): 
    '''
    One realization of Hierachical Spatial temporal model
    '''
    '''The first part '''
    input   = layers.Input(shape = input_shape)
    block1  = layers.Lambda(lambda x: x[...,None])(input)
    block1       = layers.Conv2D(F1, (ksize, 1), padding = 'same',
                                   use_bias = False, name = 'Conv2D')(block1)  
    # block1       = layers.BatchNormalization(axis = -1, name='BN-0')(block1)  # normalization on channels or time
    block1       = layers.Permute((1,3,2))(block1)
    block1       = layers.DepthwiseConv2D((1, F1), use_bias = False, 
                                   depth_multiplier = 1,
                                   depthwise_constraint = max_norm(1.),
                                   name = 'DepthConv0')(block1)
    
    # block1       = layers.SeparableConv2D(1, (5, 1), use_bias = False, 
    #                                depth_multiplier = 1, 
    #                                depthwise_constraint = max_norm(1.),
    #                                name = 'SepConv0')(block1)  
    # block1       = layers.Permute((1,3,2))(block1)

    block1       = layers.BatchNormalization(axis = -1, name = 'BN-1')(block1)
    block1       = layers.Activation(activation)(block1)    

    # block1  = layers.Lambda(lambda x: x[...,None,:])(input)

    block1       = layers.DepthwiseConv2D((5, 1), use_bias = False, 
                                   depth_multiplier = 1,
                                   depthwise_constraint = max_norm(1.),
                                   name = 'DepthConv')(block1)
    block1       = layers.BatchNormalization(axis = -1, name = 'BN-2')(block1)
    block1       = layers.Activation(activation)(block1)

    block1       = layers.AveragePooling2D((2, 1))(block1)
    block1       = layers.Dropout(droprate)(block1) 
    
    # block2       = Region_partition(loc)(block1)
    # block2_sep   = [layers.DepthwiseConv2D]

    block1       = layers.Lambda(lambda x: x[...,0,:])(block1)
    # print(block1.shape)
    '''The second part'''
    [X_out, X_R_out, X_G_out] = build_hierachy(block1, mergeMode,
                                               loc, adj_list, hop_list, 
                                               region_adjM, region_hop, poly_type,
                                               activation,blockIDX = 0)
    #
    #######################################################                                            activation,blockIDX=0)

    '''The third part'''
    X_fused = layers.Concatenate(axis=-1, name='Fuse_concate')([X_out, X_R_out, X_G_out] )
    # X_fused = layers.SeparableConv1D(14, 5, 1, 
    #                                 #  activation = 'linear',
    #                                  padding = 'same',
    #                                 #  pointwise_constraint = max_norm(1.), # 
    #                                 #  use_bias=False, 
    #                                  name = 'Sep_fuse'
    #                                 )(X_fused)
    # X_fused = layers.Add()([block1, X_fused])                                
    # X_fused = layers.BatchNormalization(axis=-1, name='BN_fuse')(X_fused)
    # X_fused = layers.Activation(activation)(X_fused)
    # X_fused = layers.AveragePooling1D(2)(X_fused)
    # X_fused = layers.Dropout(droprate)(X_fused)

    '''The 4th part'''
    # X_flatten = layers.Flatten(name = 'flatten')(X_fused)
    X_flatten = layers.GlobalAveragePooling1D()(X_fused)
    X_flatten = layers.Dropout(droprate)(X_flatten)
    output = Out_choice(LabelMode, nb_classes, Prob_model)(X_flatten)

    return Model(input, output)   

'''
HST with DiffPool
'''
from spektral.layers import DiffPool
def HST_DP(input_shape, LabelMode, nb_classes, Prob_model, Adj_C, n_regions,
            activation, droprate, F1=8, ksize=5): 
    '''
    One realization of Hierachical Spatial temporal model
    '''
    '''The first part '''
    input   = layers.Input(shape = input_shape)
    # A_input = layers.Input(shape = (14,14))
    block1  = layers.Lambda(lambda x: x[...,None])(input)
    block1       = layers.Conv2D(F1, (ksize, 1), padding = 'same',
                                   use_bias = False, name = 'Conv2D')(block1)  
    # block1       = layers.BatchNormalization(axis = -1, name='BN-0')(block1)  # normalization on channels or time
    block1       = layers.Permute((1,3,2))(block1)
    block1       = layers.DepthwiseConv2D((1, F1), use_bias = False, 
                                   depth_multiplier = 1,
                                   depthwise_constraint = max_norm(1.),
                                   name = 'DepthConv0')(block1)
    


    block1       = layers.BatchNormalization(axis = -1, name = 'BN-1')(block1)
    block1       = layers.Activation(activation)(block1)    

    # block1  = layers.Lambda(lambda x: x[...,None,:])(input)

    block1       = layers.DepthwiseConv2D((5, 1), use_bias = False, 
                                   depth_multiplier = 1,
                                   depthwise_constraint = max_norm(1.),
                                   name = 'DepthConv')(block1)
    block1       = layers.BatchNormalization(axis = -1, name = 'BN-2')(block1)
    block1       = layers.Activation(activation)(block1)

    block1       = layers.AveragePooling2D((2, 1))(block1)
    block1       = layers.Dropout(droprate)(block1) 
    
    # block2       = Region_partition(loc)(block1)
    # block2_sep   = [layers.DepthwiseConv2D]

    block1       = layers.Lambda(lambda x: x[...,0,:])(block1) # (., 62, 14)
    block1       = layers.Permute((2,1))(block1)    
    '''
    The 2nd part
    '''

    block2, Adj_R = DiffPool(n_regions)([block1, Adj_C])
    block3, _ = DiffPool(1)([block2, Adj_R])

    ''' The 3rd part'''
    X_fused = layers.Concatenate(axis=1)([block1, block2, block3])
    X_fused = layers.Permute((2,1))(X_fused)

    '''The 4th part'''
    # X_flatten = layers.Flatten(name = 'flatten')(X_fused)
    X_flatten = layers.GlobalAveragePooling1D()(X_fused)
    X_flatten = layers.Dropout(droprate)(X_flatten)
    output = Out_choice(LabelMode, nb_classes, Prob_model)(X_flatten)

    return Model(input, output) 



#%%
if __name__ == '__main__':
    # model = MSBAM(input_shape = (9,9,128,1), Nclass = 2, Droprate=0.7)
    # model.summary()

    # #%%
    # def test_model(input_shape=(128,14), activation='elu'):

    #     x_in = layers.Input(shape=input_shape)

    #     # block1       = Grouped_Separable1D(loc = [0,5,7,9,14], c_multiplier=1, 
    #     #                                    ksize=3, stride = 1, name='GS_1')(x_in)
    #     block1       = Region_fusion(loc=[0,5,7,9,14], name='RF')(x_in)
    #     block1       = layers.BatchNormalization(axis = -1, name = 'BN-2')(block1)
    #     block1       = layers.Activation(activation)(block1) 
    #     # block1       = Region_collapse(4)(block1)
    #     block1       =  Message_Passing(np.ones((4,4)), 2, type='chebyshev', name='MP1')(block1)

    #     return Model(x_in, block1)

    # model = test_model() 
    # model.summary()


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

    model = HST_model(input_shape = (128,14), LabelMode='OneHot', 
                  nb_classes=5, Prob_model=None, mergeMode='dense',
                  loc=[0,4,7,10,14], adj_list=adjN, 
                  hop_list=[2,2,2,2], 
                  region_adjM=adjR, 
                  region_hop=2, poly_type='chebyshev',
                  activation='elu', droprate=0.7)
    model.summary()



























# %%
