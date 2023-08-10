'''
For gathering benchmark results from 
the '1to1' subject indepedent benchmark
'''
#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
summary_folder = '/mnt/HDD/Benchmarks/DREAMER/Summary_SI'
#%%
def fetch_performance(model_token, encoding_type, label_token, pretrained):
    '''
    model_token: EEGNet, HST or MSBAM
    encoding_type: OneHot, OneHot_N, Mixture, Regression
    label_token: V or A
    pretrained: True or False
    '''
    if pretrained:
        file = os.path.join(summary_folder, 'DREAMER_Pretrained_{}_{}_{}.pkl'.format(model_token, label_token, encoding_type) )
    else:
        file = os.path.join(summary_folder, 'DREAMER_Alone_{}_{}_{}.pkl'.format(model_token, label_token, encoding_type) )

    try:
        with open(file, 'rb') as pkl:
            temp = pickle.load(pkl)    
        summary = {
            'top_1_acc':[],
            'top_2_acc':[],
            'F1':[],
            'CM':[],
            'off_neibor':[],
            'closeness': []
        }
        for i in range(23):
            for _k in ['top_1_acc', 'top_2_acc', 'CM', 'F1', 'off_neibor', 'closeness']:
                summary[_k].append(temp['S{:02d}'.format(i+1)][_k])

        for _k in summary.keys():
            try:
                summary[_k] = np.array(summary[_k])
            except:
                summary[_k] = None

    except:
       summary = None
       raise FileNotFoundError
    return summary


def get_target_performance(encoding_type, label_token, pretrained=True):

    P_EEGNet = fetch_performance('EEGNet', encoding_type, label_token, pretrained)
    P_MSBAM= fetch_performance('MSBAM', encoding_type, label_token, pretrained)
    P_HST = fetch_performance('HST', encoding_type, label_token, pretrained)
    
    print('='*60)
    for _key in ['top_1_acc', 'top_2_acc', 'F1']:
        
        print('EEGNet: {}'.format(_key))
        print('    Mean : {:.02f}'.format( np.mean(P_EEGNet[_key]) ))
        print('    Std: {:.02f}'.format( np.std(P_EEGNet[_key]) ) )
        
        print('MSBAM: {}'.format(_key))
        print('    Mean: {:.02f}'.format( np.mean(P_MSBAM[_key])) )
        print('    Std: {:.02f}'.format( np.std(P_MSBAM[_key]) ) )

        print('HST: {}'.format(_key))
        print('    Mean: {:.02f}'.format( np.mean(P_HST[_key])) )
        print('    Std: {:.02f}'.format( np.std(P_HST[_key]) ) )

        print('\n')

    print('EEGNet: {}'.format('off_neibor'))
    print('    Mean : {:.02f}'.format( np.mean(P_EEGNet['off_neibor']) /900.0*100)) # later 50 seconds of data
    print('    Std: {:.02f}'.format( np.std(P_EEGNet['off_neibor']) /900.0*100) ) 
    
    print('MSBAM: {}'.format('off_neibor'))
    print('    Mean: {:.02f}'.format( np.mean(P_MSBAM['off_neibor'])/900.0*100) )
    print('    Std: {:.02f}'.format( np.std(P_MSBAM['off_neibor']) /900.0*100 ))

    print('HST: {}'.format('off_neibor'))
    print('    Mean: {:.02f}'.format( np.mean(P_HST['off_neibor'])/900.0*100))
    print('    Std: {:.02f}'.format( np.std(P_HST['off_neibor']) /900.0*100) )

    print('\n')
        
    print('EEGNet: closeness') # [chances that top 2 predictions are adjacent (I), (I) and top-1 prediction is true, (I) and true in top-2 prediction]
    print('    Mean: {}'.format( np.mean(P_EEGNet['closeness'],  axis=0)) )
    print('    Std: {}'.format( np.std(P_EEGNet['closeness'], axis=0) )  )
    
    print('MSBAM: closeness')
    print('    Mean: {}'.format( np.mean(P_MSBAM['closeness'], axis=0) ) )
    print('    Std: {}'.format( np.std(P_MSBAM['closeness'], axis=0) )  )

    print('HST: closeness')
    print('    Mean: {}'.format( np.mean(P_HST['closeness'], axis=0) ) )
    print('    Std: {}'.format( np.std(P_HST['closeness'], axis=0 ) ) )    
        
    print('\n')

    print('='*60)


#%%
EEGNet_OneHot_V_Pretrained = fetch_performance('EEGNet', 'OneHot', 'V', True)

#%%
def fetch_HST_version(encoding_type, label_token, pretrained, version):
    '''
    compare HST with different graph structure
    encoding_type: OneHot, OneHot_N, Mixture, Regression
    label_token: V or A
    pretrained: True or False
    '''
    if pretrained:
        file = os.path.join(summary_folder, 'DREAMER_Pretrained_HST_{}_{}_V{}.pkl'.format(label_token, encoding_type, version) )
    else:
        file = os.path.join(summary_folder, 'DREAMER_Alone_HST_{}_{}_V{}.pkl'.format(label_token, encoding_type, version) )

    try:
        with open(file, 'rb') as pkl:
            temp = pickle.load(pkl)    
        summary = {
            'top_1_acc':[],
            'top_2_acc':[],
            'F1':[],
            'CM':[],
            'off_neibor':[],
            'closeness': []
        }
        for i in range(23):
            for _k in ['top_1_acc', 'top_2_acc', 'CM', 'F1', 'off_neibor', 'closeness']:
                summary[_k].append(temp['S{:02d}'.format(i+1)][_k])

        for _k in summary.keys():
            try:
                summary[_k] = np.array(summary[_k])
            except:
                summary[_k] = None

    except:
       summary = None
       raise FileNotFoundError
    return summary

HST_OneHot_V_Pretrained_V0 = fetch_performance('HST', 'OneHot', 'V', True)
HST_OneHot_V_Pretrained_V1 = fetch_HST_version('OneHot', 'V', True, 1)
HST_OneHot_V_Pretrained_V2 = fetch_HST_version('OneHot', 'V', True, 2)


#%%
def HST_ablation_graph(label_token):
    HST_OneHot_Pretrained_V0 = fetch_performance('HST', 'OneHot', label_token, True)
    HST_OneHot_Pretrained_V1 = fetch_HST_version('OneHot', label_token, True, 1)
    HST_OneHot_Pretrained_V2 = fetch_HST_version('OneHot', label_token, True, 2)

    print('='*60)
    for _key in ['top_1_acc', 'top_2_acc', 'F1']:
        
        print('HST_V0: {}'.format(_key))
        print('    Mean : {:.02f}'.format( np.mean(HST_OneHot_Pretrained_V0[_key]) ))
        print('    Std: {:.02f}'.format( np.std(HST_OneHot_Pretrained_V0[_key]) ) )
        
        print('HST_V1: {}'.format(_key))
        print('    Mean: {:.02f}'.format( np.mean(HST_OneHot_Pretrained_V1[_key])) )
        print('    Std: {:.02f}'.format( np.std(HST_OneHot_Pretrained_V1[_key]) ) )

        print('HST_V2: {}'.format(_key))
        print('    Mean: {:.02f}'.format( np.mean(HST_OneHot_Pretrained_V2[_key])) )
        print('    Std: {:.02f}'.format( np.std(HST_OneHot_Pretrained_V2[_key]) ) )

        print('\n')

    print('HST_V0: {}'.format('off_neibor'))
    print('    Mean : {:.02f}'.format( 100- np.mean(HST_OneHot_Pretrained_V0['off_neibor'])/900.0*100)) # later 50 seconds of data
    print('    Std: {:.02f}'.format( np.std(HST_OneHot_Pretrained_V0['off_neibor']) /900.0*100) ) 

    print('HST_V1: {}'.format('off_neibor'))
    print('    Mean: {:.02f}'.format( 100-np.mean(HST_OneHot_Pretrained_V1['off_neibor'])/900.0*100) )
    print('    Std: {:.02f}'.format( np.std(HST_OneHot_Pretrained_V1['off_neibor']) /900.0*100 ))

    print('HST_V2: {}'.format('off_neibor'))
    print('    Mean: {:.02f}'.format( 100-np.mean(HST_OneHot_Pretrained_V2['off_neibor'])/900.0*100))
    print('    Std: {:.02f}'.format( np.std(HST_OneHot_Pretrained_V2['off_neibor']) /900.0*100) )

    print('\n')
        
    print('HST_V0: closeness') # [chances that top 2 predictions are adjacent (I), (I) and top-1 prediction is true, (I) and true in top-2 prediction]
    print('    Mean: {}'.format( 100*np.mean(HST_OneHot_Pretrained_V0['closeness'],  axis=0)) )
    print('    Std: {}'.format( 100*np.std(HST_OneHot_Pretrained_V0['closeness'], axis=0) )  )

    print('HST_V1: closeness')
    print('    Mean: {}'.format( 100*np.mean(HST_OneHot_Pretrained_V1['closeness'], axis=0) ) )
    print('    Std: {}'.format( 100*np.std(HST_OneHot_Pretrained_V1['closeness'], axis=0) )  )

    print('HST_V2: closeness')
    print('    Mean: {}'.format( 100*np.mean(HST_OneHot_Pretrained_V2['closeness'], axis=0) ) )
    print('    Std: {}'.format( 100*np.std(HST_OneHot_Pretrained_V2['closeness'], axis=0 ) ) )    
        
    print('\n')

    print('='*60)

    datalist = [np.array([np.mean(HST_OneHot_Pretrained_V0['F1']), np.mean(HST_OneHot_Pretrained_V1['F1']), np.mean(HST_OneHot_Pretrained_V2['F1'])]),
                np.array([np.mean(HST_OneHot_Pretrained_V0['top_2_acc']), np.mean(HST_OneHot_Pretrained_V1['top_2_acc']), np.mean(HST_OneHot_Pretrained_V2['top_2_acc'])]),
                np.array([100-np.mean(HST_OneHot_Pretrained_V0['off_neibor']/900*100), 100-np.mean(HST_OneHot_Pretrained_V1['off_neibor'])/900*100, 100-np.mean(HST_OneHot_Pretrained_V2['off_neibor'])/900*100]),
                np.array([100*np.mean(HST_OneHot_Pretrained_V0['closeness'][:,-1]), 100*np.mean(HST_OneHot_Pretrained_V1['closeness'][:,-1]), 100*np.mean(HST_OneHot_Pretrained_V2['closeness'][:,-1])])
    ]

    return datalist

# %%
def score_bar(datalist, colorlist, labellist, namelist,
              width=0.15, ylim = [0., 1.1], alpha = 0.5, figsize=(8,30),):
    '''
    datalist: a list of data to plot, each member is a numpy array
    colorlist: a list of color for each group member
    labellist: a list, name for each group
    namelist: a list, name for the legend
    '''
    # Setting the positions and width for the bars
    pos = list(range(len(labellist))) 
#    width = 0.1 
        
    # Plotting the bars
    fig, ax = plt.subplots(figsize=figsize)
    num_data = len(datalist)
    # Create a bar with pre_score data,
    # in position pos,
    for i in range(num_data):
        plt.bar([p+width*i for p in pos], datalist[i], 
                width=width, alpha=alpha, color=colorlist[i], label=namelist[i])
 
    
    # Set the y axis label
    ax.set_ylabel('Value(%)')
    ax.set_xlabel('Metric')
    # Set the chart's title
    # ax.set_title('Summaries')
    
    # Set the position of the x ticks
    ax.set_xticks([p + 1.0 * width for p in pos])
    # Set the labels for the x ticks
    ax.set_xticklabels(labellist)
    
    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos)-width, max(pos)+4*width)
    plt.ylim(ylim)
    plt.xticks(rotation = 0)
    # Adding the legend and showing the plot
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

'''
make bar plots for different variations of HST
show F1, top-2-acc, off_neibor and clossness[-1]
'''
HST_ablation_data = HST_ablation_graph('V')
score_bar(np.array(HST_ablation_data).T, colorlist=['r','k','b'], 
          labellist=['F1','Top2Acc.','off_neibor', 'closeness'], 
          namelist = ['G0', 'G1', 'G2'],
          width=0.25,
          ylim = [70., 95.0],
          figsize=(6,4))
# %%
import seaborn as sns
sns.set_theme(style="whitegrid")
import pandas as pd
# penguins = sns.load_dataset("penguins")

# # Draw a nested barplot by species and sex
# g = sns.catplot(
#     data=penguins, kind="bar",
#     x="species", y="body_mass_g", hue="sex",
#     palette="dark", alpha=.6, height=6
# )
# g.despine(left=True)
# g.set_axis_labels("", "Body mass (g)")
# g.legend.set_title("")
# %%
'''
A different plot with seaborn
'''
data_array = np.r_[HST_OneHot_V_Pretrained_V0['F1'], 
                   HST_OneHot_V_Pretrained_V1['F1'], 
                   HST_OneHot_V_Pretrained_V2['F1'],
                   HST_OneHot_V_Pretrained_V0['top_2_acc'], 
                   HST_OneHot_V_Pretrained_V1['top_2_acc'], 
                   HST_OneHot_V_Pretrained_V2['top_2_acc'],
                   100-HST_OneHot_V_Pretrained_V0['off_neibor']/900*100, 
                   100-HST_OneHot_V_Pretrained_V1['off_neibor']/900*100, 
                   100-HST_OneHot_V_Pretrained_V2['off_neibor']/900*100,
                   HST_OneHot_V_Pretrained_V0['closeness'][:,-1]*100, 
                   HST_OneHot_V_Pretrained_V1['closeness'][:,-1]*100, 
                   HST_OneHot_V_Pretrained_V2['closeness'][:,-1]*100
                ]
df_HST_abl = pd.DataFrame(data=data_array, columns=['Value'])
# %%
df_HST_abl.insert(1,'Metric',['F1']*23*3 + ['top_2_acc']*23*3 + ['off_neibor']*23*3 + ['closeness']*23*3)
df_HST_abl.insert(2,'ID',[i for i in range(1,24)]*3*4)
temp_G_name = ['G0']*23 + ['G1']*23 + ['G2']*23
df_HST_abl.insert(3,'Graph', temp_G_name*4)

g = sns.catplot(
    data=df_HST_abl, kind="bar",
    x="Metric", y='Value', hue="Graph", ci = None,
    palette="dark", alpha=.6, height=6,
)
g.despine(left=True)
g.set_axis_labels("", "Value (%)")
g.set(ylim=(70, 95))
# g.legend.set_title("")
# %%
'''
Load Model
'''
from Models import HST_model,MSBAM, EEGNet
from Probability_helper import mixture_gaussian

def load_model_SD(Subject, LabelMode,Label_Code , ModelKey='HST'):
    if LabelMode == 'Mixture':
        prob_model = lambda x: mixture_gaussian(x, False, 3)
    else:
        prob_model = None

    if ModelKey == 'HST':
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
    elif ModelKey == 'EEGNet':
        model = EEGNet(prob_model, nb_classes = 5, LabelMode=LabelMode, activation='elu',
                Chans = 14, Samples = 128,
                dropoutRate = 0.75, kernLength = 5, F1 = 8, 
                D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout')
    
    elif ModelKey == 'MSBAM':
        model = MSBAM(input_shape = (9,9,128,1), LabelMode=LabelMode,
                    Nclass = 5, Prob_model = prob_model, Droprate=0.75)
    
    ckpt_path = '/mnt/HDD/Benchmarks/DREAMER/ckpt/DREAMER_S{:02d}_{}_{}_{}'.format(Subject, ModelKey, Label_Code, LabelMode)

    model.load_weights(ckpt_path)
    

    return model


def load_model_SI(Subject, LabelMode,Label_Code, Version, Pretrained=True):
    if LabelMode == 'Mixture':
        prob_model = lambda x: mixture_gaussian(x, False, 3)
    else:
        prob_model = None
    
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
    
    if Pretrained:
        save_path = '/mnt/HDD/Benchmarks/DREAMER/ckpt_SI/DREAMER_S{:02d}_{}_{}_{}_reston1.h5'.format( Subject, 'HST', Label_Code, LabelMode)
    else:
        save_path = '/mnt/HDD/Benchmarks/DREAMER/ckpt_SI/DREAMER_S{:02d}Alone_{}_{}_{}_reston1.h5'.format(Subject, 'HST', Label_Code, LabelMode)
  
    model.load_weights(save_path[:-3] + '_V{}'.format(Version) + '.h5')
    

    return model

#%%
'''
Visualize deep feature with UMAP
Comparing between OneHot Label encoding and the specially smoothed encoding method
'''
import umap.umap_ as umap
from scipy.io import loadmat
from utils import Embed_DREAMER,make_segs
from keras.utils import to_categorical
from tensorflow.keras.models import Model

def label_smoothing(Y):
    grid = np.arange( 5 )
    spread = 0.5
    temp = np.exp(-0.5*(grid[None,:]-Y[:,None])**2/spread**2)
    return temp/np.sum(temp,axis=1, keepdims=True)

def onehot(Y):
    return to_categorical(Y, 5)

def Embed_X(X):
    return Embed_DREAMER(X.transpose(0,2,1))[...,None]

#%%
'''
Just an exploration
'''

# Subject = 23   # most obvious -- V: S3 ;  A: S23, 

# X = loadmat('/mnt/HDD/Datasets/DREAMER/S{:02d}_1min.mat'.format(Subject))['X'].transpose((1,0,2))
# Y = loadmat('/mnt/HDD/Datasets/DREAMER/Labels.mat')['Y'][Subject-1] #careful here for the indexing

# Label_V = Y[:,0] - 1
# Label_A = Y[:,1] - 1

# X_prep = lambda x: x/np.max(np.abs(x), axis=1, keepdims=True)
# X = X_prep(X)

# X_segs = make_segs(X, 128, 128)
# X_segs_M = Embed_X(X_segs)

# Y_segs_V = np.repeat(Label_V, X_segs.shape[0]//18, axis=0)
# # Y_V_OH = onehot(Y_segs_V)
# # Y_V_S = label_smoothing(Y_segs_V)

# Y_segs_A = np.repeat(Label_A, X_segs.shape[0]//18, axis=0)
# # Y_A_OH = onehot(Y_segs_A)
# # Y_A_S = label_smoothing(Y_segs_A)

# model = load_model_SD(Subject, 'OneHot', 'A', ModelKey='HST')
# feature_map = Model(model.input, model.layers[-2].output)
# feature  = feature_map(X_segs).numpy()

# mapper = umap.UMAP(n_neighbors=100, n_components=2, metric='euclidean', 
#                 spread=0.5, min_dist=0.2, local_connectivity=1.0,
#                 output_metric='euclidean', init='spectral', 
#                 densmap=False, random_state = 1234)

# embeded=[]
# embeded.append( mapper.fit_transform(feature) )
# plt.figure()
# sns.scatterplot(x = embeded[0][:,0], y = embeded[0][:,1], hue = Y_segs_A+1, palette="rainbow")
# plt.xticks([])
# plt.yticks([])

#%%
'''
Saving deep features for embedding plots
'''
embeded = []
for label_code in ['V', 'A']:
    if label_code == 'V':
        Subject = 3
    else:
        Subject = 23

    X = loadmat('/mnt/HDD/Datasets/DREAMER/S{:02d}_1min.mat'.format(Subject))['X'].transpose((1,0,2))
    Y = loadmat('/mnt/HDD/Datasets/DREAMER/Labels.mat')['Y'][Subject-1] #careful here for the indexing
    
    if label_code == 'V':
        Label = Y[:,0] 
    else:
        Label = Y[:,1] 

    X_prep = lambda x: x/np.max(np.abs(x), axis=1, keepdims=True)
    X = X_prep(X)

    X_segs = make_segs(X, 128, 128)
    X_segs_M = Embed_X(X_segs)

    Y_segs = np.repeat(Label, X_segs.shape[0]//18, axis=0)


    for label_mode in ['OneHot_N', 'OneHot']:
        temp = []
        for _MK in ['EEGNet', 'MSBAM', 'HST']:
            model = load_model_SD(Subject, label_mode, label_code , ModelKey=_MK)
            feature_map = Model(model.input, model.layers[-2].output)
            if _MK == 'MSBAM':
                feature  = feature_map(X_segs_M).numpy()
            else:
                feature  = feature_map(X_segs).numpy()

            mapper = umap.UMAP(n_neighbors=100, n_components=2, metric='euclidean', 
                            spread=0.5, min_dist=0.2, local_connectivity=1.0,
                            output_metric='euclidean', init='spectral', 
                            densmap=False, random_state = 1234)

            temp.append( mapper.fit_transform(feature) )
        embeded.append(temp)

with open('/mnt/HDD/Benchmarks/DREAMER/featureV3A23.pkl', 'wb') as pkl:
    pickle.dump(embeded, pkl)



#%%
'''
Embedding plot for feature
'''
# with open('/mnt/HDD/Benchmarks/DREAMER/featureV3A23.pkl', 'rb') as pkl:
#     embeded = pickle.load(pkl)
model_keys = ['EEGNet', 'MSBAM', 'HiSTN']
row_name = ['OneHot', 'Specially Smoothed']
for label_code in ['V', 'A']:
    if label_code == 'V':
        Subject = 3
    else:
        Subject = 23

    Y = loadmat('/mnt/HDD/Datasets/DREAMER/Labels.mat')['Y'][Subject-1] #careful here for the indexing
    if label_code == 'V':
        Label = Y[:,0] 
    else:
        Label = Y[:,1] 
    Y_segs = np.repeat(Label, 1080//18, axis=0)

    # plt.figure(figsize=(12,8))
    # for i in range(2):
    #     for j in range(3):
    #         plt.subplot(2,3,3*i+j+1)
    #         if label_code == 'V':
    #             sns.scatterplot(x = embeded[i][j][:,0], y = embeded[i][j][:,1], hue = Y_segs, palette="rainbow")
    #         else:
    #             sns.scatterplot(x = embeded[i+2][j][:,0], y = embeded[i+2][j][:,1], hue = Y_segs, palette="rainbow")
    #         plt.xticks([])
    #         plt.yticks([])
    #         if i == 0:
    #             plt.title(model_keys[j],fontsize=16)
    #         if j==0:
    #             plt.ylabel(row_name[i],fontsize=14)
    
    plt.figure(figsize=(8,12))
    for i in range(3):
        for j in range(2):
            plt.subplot(3,2,2*i+j+1)
            if label_code == 'V':
                pt_X = embeded[j][i][:,0]
                pt_Y = embeded[j][i][:,1]
                
            else:
                pt_X = embeded[j+2][i][:,0]
                pt_Y = embeded[j+2][i][:,1]
            sns.scatterplot(x = pt_X, y = pt_Y, hue = Y_segs, palette="rainbow")
            centers = []
            for k in range(1,6):
                ind = np.where(Y_segs == k)[0]
                centers.append((np.mean(pt_X[ind]),np.mean(pt_Y[ind])))
                # center.append((np.median(pt_X),np.median(pt_Y)))
            for k in range(4):
                plt.arrow(centers[k][0], centers[k][1], 
                          centers[k+1][0]-centers[k][0], centers[k+1][1]-centers[k][1],
                          width=0.08, facecolor='black', alpha = 0.5,
                          head_starts_at_zero=True, length_includes_head=True)
            if i == 0:
                plt.title(row_name[j],fontsize=16)
            if j==0:
                plt.ylabel(model_keys[i],fontsize=14)
            plt.xticks([])
            plt.yticks([])
    if label_code == 'V':
        plt.suptitle('Valence', y=0.1, fontsize=18)
    else:
        plt.suptitle('Arousal', y=0.1, fontsize=18)

    


# %%
