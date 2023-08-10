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
        if model_token =='HST':
            file = os.path.join(summary_folder, 'DREAMER_Pretrained_{}_{}_{}_reston1_V0.pkl'.format(model_token, label_token, encoding_type) )
        else:
            file = os.path.join(summary_folder, 'DREAMER_Pretrained_{}_{}_{}_reston1.pkl'.format(model_token, label_token, encoding_type) )

    else:
        if model_token =='HST':
            file = os.path.join(summary_folder, 'DREAMER_Alone_{}_{}_{}_reston1_V0.pkl'.format(model_token, label_token, encoding_type) )
        else:
            file = os.path.join(summary_folder, 'DREAMER_Alone_{}_{}_{}_reston1.pkl'.format(model_token, label_token, encoding_type) )

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
HST_OneHot_V_Pretrained = fetch_performance('HST', 'OneHot', 'V', True)
HST_OneHot_V_Alone = fetch_performance('HST', 'OneHot', 'V', False)

HST_OneHot_A_Pretrained = fetch_performance('HST', 'OneHot', 'A', True)
HST_OneHot_A_Alone = fetch_performance('HST', 'OneHot', 'A', False)

for _k in ['top_1_acc', 'top_2_acc', 'F1']:
    print('Mean: {}'.format(_k))
    print(np.mean(HST_OneHot_V_Pretrained[_k]))
    print(np.mean(HST_OneHot_V_Alone[_k]))
    print('\n')
    print('Std: {}'.format(_k))
    print(np.std(HST_OneHot_V_Pretrained[_k]))
    print(np.std(HST_OneHot_V_Alone[_k]))
    print('\n')

print('Mean: {}'.format('off_neibor'))
print(100-np.mean(HST_OneHot_V_Pretrained['off_neibor'])/900*100)
print(100-np.mean(HST_OneHot_V_Alone['off_neibor'])/900*100)
print('\n')
print('Std: {}'.format('off_neibor'))
print(np.std(HST_OneHot_V_Pretrained['off_neibor'])/900*100)
print(np.std(HST_OneHot_V_Alone['off_neibor'])/900*100)
print('\n')  

print('Mean: {}'.format('closeness'))
print(100*np.mean(HST_OneHot_V_Pretrained['closeness'], axis=0))
print(100*np.mean(HST_OneHot_V_Alone['closeness'], axis=0))
print('\n')
print('Std: {}'.format('closeness'))
print(100*np.std(HST_OneHot_V_Pretrained['closeness'], axis=0))
print(100*np.std(HST_OneHot_V_Alone['closeness'], axis=0))
print('\n') 
# %%
'''
Box Plot
'''
#==============================================================================
# A grouped boxplot
#==============================================================================
import seaborn as sns
import pandas as pd

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color = color, facecolor = color, linewidth=2)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
#    plt.setp(bp['medians'], color=color)
    plt.setp(bp['medians'], linewidth=2)
    plt.setp(bp['fliers'], markersize=4)

def show_box_group(data, names, ticks, colors, box_width = 0.3, 
                   sparsity = 3, ymin=0, ymax = 1, style = 'bmh', fontsize=12):
    # with plt.style.context(style):
        plt.figure()
        for i, sample in enumerate(data):
            bp = plt.boxplot(sample, positions=np.array(np.arange(sample.shape[1]))*sparsity-0.6+0.4*i,  widths=box_width, sym = 'o',
                    notch=True, patch_artist=True)
            set_box_color(bp, colors[i])
            for patch in bp['boxes']:
                patch.set_alpha(0.8)
            plt.plot([], c=colors[i], label=names[i])
        plt.legend(loc='lower right')

        plt.xticks(np.arange(0, len(ticks) * sparsity, sparsity), ticks, 
                  rotation = 45, fontsize=fontsize)
        plt.xlim(-2, len(ticks)*sparsity-0.4)
        plt.ylim(ymin, ymax)
        plt.ylabel('Value(%)', fontsize=fontsize)
        #plt.title('Different methods on selected regions')
        plt.grid()
        plt.tight_layout()


#%%
'''
A box plot compare the effects of pretrained or not
'''
ticks = ['F1.', 'Top2Acc.', 'Tri-P', 'Seq2HR']
# colors = ['#2C7BB6', '#999900', '#2ca25f', '#9400d3','#636363']
palette = sns.color_palette("tab10", 10).as_hex()
colors = palette[:5]
box_width = 0.3
sparsity = 3 
legend_list = ['Alone-V', 'Pretrained-V', 'Alone-A', 'Pretrained-A']


summary_all = [
    np.c_[HST_OneHot_V_Alone['F1'],
          HST_OneHot_V_Alone['top_2_acc'],
          100-HST_OneHot_V_Alone['off_neibor']/900*100,
          HST_OneHot_V_Alone['closeness'][:,-1]*100
    ],  
    np.c_[HST_OneHot_V_Pretrained['F1'],
          HST_OneHot_V_Pretrained['top_2_acc'],
          100-HST_OneHot_V_Pretrained['off_neibor']/900*100,
          HST_OneHot_V_Pretrained['closeness'][:,-1]*100
    ],
    np.c_[HST_OneHot_A_Alone['F1'],
          HST_OneHot_A_Alone['top_2_acc'],
          100-HST_OneHot_A_Alone['off_neibor']/900*100,
          HST_OneHot_A_Alone['closeness'][:,-1]*100
    ],  
    np.c_[HST_OneHot_A_Pretrained['F1'],
          HST_OneHot_A_Pretrained['top_2_acc'],
          100-HST_OneHot_A_Pretrained['off_neibor']/900*100,
          HST_OneHot_A_Pretrained['closeness'][:,-1]*100
    ]    
]  
show_box_group(summary_all , legend_list,  ticks, colors, ymin=60, ymax=100.0)
# %%
'''
A boxplot comparing pretrained on all or pretrain on one subject
'''
def fetch_performance_1on1(model_token, encoding_type, label_token, pretrained):
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

P_HST_V_1on1 = fetch_performance_1on1('HST', 'OneHot', 'V', True)
P_HST_A_1on1 = fetch_performance_1on1('HST', 'OneHot', 'A', True)
# %%no correction
legend_list = ['Alone-V', 'Pretrained_on_rest-V', 'Pretrained_on_prev.-V']
summary_all = [
    np.c_[HST_OneHot_V_Alone['F1'],
          HST_OneHot_V_Alone['top_2_acc'],
          100-HST_OneHot_V_Alone['off_neibor']/900*100,
          HST_OneHot_V_Alone['closeness'][:,-1]*100
    ],  
    np.c_[HST_OneHot_V_Pretrained['F1'],
          HST_OneHot_V_Pretrained['top_2_acc'],
          100-HST_OneHot_V_Pretrained['off_neibor']/900*100,
          HST_OneHot_V_Pretrained['closeness'][:,-1]*100
    ],
    np.c_[P_HST_V_1on1['F1'],
          P_HST_V_1on1['top_2_acc'],
          100-P_HST_V_1on1['off_neibor']/900*100,
          P_HST_V_1on1['closeness'][:,-1]*100
    ]    
]  
show_box_group(summary_all , legend_list,  ticks, colors, ymin=60, ymax=100.0)

legend_list = ['Alone-A', 'Pretrained_on_rest-A', 'Pretrained_on_prev.-A']
summary_all = [
    np.c_[HST_OneHot_A_Alone['F1'],
          HST_OneHot_A_Alone['top_2_acc'],
          100-HST_OneHot_A_Alone['off_neibor']/900*100,
          HST_OneHot_A_Alone['closeness'][:,-1]*100
    ],  
    np.c_[HST_OneHot_A_Pretrained['F1'],
          HST_OneHot_A_Pretrained['top_2_acc'],
          100-HST_OneHot_A_Pretrained['off_neibor']/900*100,
          HST_OneHot_A_Pretrained['closeness'][:,-1]*100
    ],
    np.c_[P_HST_A_1on1['F1'],
          P_HST_A_1on1['top_2_acc'],
          100-P_HST_A_1on1['off_neibor']/900*100,
          P_HST_A_1on1['closeness'][:,-1]*100
    ]    
]  
show_box_group(summary_all , legend_list,  ticks, colors, ymin=60, ymax=100.0)




# %%
def fetch_HST_version(encoding_type, label_token, pretrained, version, Correct_Y=True):
    '''
    compare HST with different graph structure
    encoding_type: OneHot, OneHot_N, Mixture, Regression
    label_token: V or A
    pretrained: True or False
    '''
    if pretrained:
        if Correct_Y:
            file = os.path.join(summary_folder, 'DREAMER_Pretrained_HST_{}_{}_reston1_V{}.pkl'.format(label_token, encoding_type, version) )
        else:
            file = os.path.join(summary_folder, 'DREAMER_Pretrained_HST_{}_{}_reston1_rawY_V{}.pkl'.format(label_token, encoding_type, version) )
    else:
        if Correct_Y:
            file = os.path.join(summary_folder, 'DREAMER_Alone_HST_{}_{}_reston1_V{}.pkl'.format(label_token, encoding_type, version) )
        else:
            file = os.path.join(summary_folder, 'DREAMER_Pretrained_HST_{}_{}_reston1_rawY_V{}.pkl'.format(label_token, encoding_type, version) )

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

HST_OneHot_V_Pretrained_V0 = fetch_HST_version('OneHot', 'V', True, 0)
HST_OneHot_V_Pretrained_V1 = fetch_HST_version('OneHot', 'V', True, 1)
HST_OneHot_V_Pretrained_V2 = fetch_HST_version('OneHot', 'V', True, 2)
#%%
'''
Valence
'''
legend_list = ['No Pretraining', 'Pretrained_on_prev.', 'Pretrained_on_rest(NC)','Pretrained_on_rest']
HST_OneHot_V_Pretrained_V0_rawY = fetch_HST_version('OneHot', 'V', True, 0, False)
summary_all = [
    np.c_[HST_OneHot_V_Alone['F1'],
          HST_OneHot_V_Alone['top_2_acc'],
          100-HST_OneHot_V_Alone['off_neibor']/900*100,
          HST_OneHot_V_Alone['closeness'][:,-1]*100
    ],  # no pretrain
    np.c_[P_HST_V_1on1['F1'],
          P_HST_V_1on1['top_2_acc'],
          100-P_HST_V_1on1['off_neibor']/900*100,
          P_HST_V_1on1['closeness'][:,-1]*100
    ],  # pretrain on pevious subject
    np.c_[HST_OneHot_V_Pretrained_V0_rawY['F1'],
          HST_OneHot_V_Pretrained_V0_rawY['top_2_acc'],
          100-HST_OneHot_V_Pretrained_V0_rawY['off_neibor']/900*100,
          HST_OneHot_V_Pretrained_V0_rawY['closeness'][:,-1]*100
    ],  # pretrain on rest (No Correction)
    np.c_[HST_OneHot_V_Pretrained['F1'],
          HST_OneHot_V_Pretrained['top_2_acc'],
          100-HST_OneHot_V_Pretrained['off_neibor']/900*100,
          HST_OneHot_V_Pretrained['closeness'][:,-1]*100
    ]
  
]  
show_box_group(summary_all , legend_list,  ticks, colors, 
               ymin=60, ymax=100.0, box_width=0.3, sparsity=3,
               fontsize=12)

#%%
'''
Arousal
'''
# legend_list = ['Alone-A', 'Pretrained_on_rest(no correction)-A','Pretrained_on_rest-A', 'Pretrained_on_prev.-A']
HST_OneHot_A_Pretrained_V0_rawY = fetch_HST_version('OneHot', 'A', True, 0, False)
summary_all = [
    np.c_[HST_OneHot_A_Alone['F1'],
          HST_OneHot_A_Alone['top_2_acc'],
          100-HST_OneHot_A_Alone['off_neibor']/900*100,
          HST_OneHot_A_Alone['closeness'][:,-1]*100
    ], # no pretrain
    np.c_[P_HST_A_1on1['F1'],
          P_HST_A_1on1['top_2_acc'],
          100-P_HST_A_1on1['off_neibor']/900*100,
          P_HST_A_1on1['closeness'][:,-1]*100
    ], # pretrain on previous subject
    np.c_[HST_OneHot_A_Pretrained_V0_rawY['F1'],
          HST_OneHot_A_Pretrained_V0_rawY['top_2_acc'],
          100-HST_OneHot_A_Pretrained_V0_rawY['off_neibor']/900*100,
          HST_OneHot_A_Pretrained_V0_rawY['closeness'][:,-1]*100
    ], # pretrain on rest (no Y correction)  
    np.c_[HST_OneHot_A_Pretrained['F1'],
          HST_OneHot_A_Pretrained['top_2_acc'],
          100-HST_OneHot_A_Pretrained['off_neibor']/900*100,
          HST_OneHot_A_Pretrained['closeness'][:,-1]*100
    ] # pretrain on rest 
]  
show_box_group(summary_all , legend_list,  ticks, colors, ymin=60, ymax=100.0,
               box_width=0.3, sparsity=3,
               fontsize=12)

#%%
def HST_ablation_graph(label_token):
    HST_OneHot_Pretrained_V0 = fetch_HST_version('OneHot', label_token, True, 0)
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
          width=0.25, alpha=0.5,
          ylim = [70., 95.0],
          figsize=(6,4))
# %%
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
df_HST_abl.insert(1,'Metric',['F1']*23*3 + ['Top2Acc.']*23*3 + ['Tri-P']*23*3 + ['Seq2HR']*23*3)
df_HST_abl.insert(2,'ID',[i for i in range(1,24)]*3*4)
temp_G_name = ['G0']*23 + ['G1']*23 + ['G2']*23
df_HST_abl.insert(3,'Graph', temp_G_name*4)

#%%
g = sns.catplot(
    data=df_HST_abl, kind="bar",
    x="Metric", y='Value', hue="Graph", errorbar="ci",
    palette="Set2", alpha=.6, height=6, 
)
g.despine(left=True)
g.set_axis_labels("", "Value (%)")
g.set(ylim=(70, 95))

#%%
HST_OneHot_A_Pretrained_V0 = fetch_HST_version('OneHot', 'A', True, 0)
HST_OneHot_A_Pretrained_V1 = fetch_HST_version('OneHot', 'A', True, 1)
HST_OneHot_A_Pretrained_V2 = fetch_HST_version('OneHot', 'A', True, 2)

data_A_array = np.r_[HST_OneHot_A_Pretrained_V0['F1'], 
                   HST_OneHot_A_Pretrained_V1['F1'], 
                   HST_OneHot_A_Pretrained_V2['F1'],
                   HST_OneHot_A_Pretrained_V0['top_2_acc'], 
                   HST_OneHot_A_Pretrained_V1['top_2_acc'], 
                   HST_OneHot_A_Pretrained_V2['top_2_acc'],
                   100-HST_OneHot_A_Pretrained_V0['off_neibor']/900*100, 
                   100-HST_OneHot_A_Pretrained_V1['off_neibor']/900*100, 
                   100-HST_OneHot_A_Pretrained_V2['off_neibor']/900*100,
                   HST_OneHot_A_Pretrained_V0['closeness'][:,-1]*100, 
                   HST_OneHot_A_Pretrained_V1['closeness'][:,-1]*100, 
                   HST_OneHot_A_Pretrained_V2['closeness'][:,-1]*100
                ]
df_A_HST_abl = pd.DataFrame(data=data_A_array, columns=['Value'])
# %%
df_A_HST_abl.insert(1,'Metric',['F1']*23*3 + ['Top2acc']*23*3 + ['Tri-P']*23*3 + ['Seq2HR']*23*3)
df_A_HST_abl.insert(2,'ID',[i for i in range(1,24)]*3*4)
temp_G_name = ['G0']*23 + ['G1']*23 + ['G2']*23
df_A_HST_abl.insert(3,'Graph', temp_G_name*4)

#%%
g = sns.catplot(
    data=df_A_HST_abl, kind="bar",
    x="Metric", y='Value', hue="Graph", errorbar="ci",
    palette="Set2", alpha=.6, height=6,
)
g.despine(left=True)
g.set_axis_labels("", "Value (%)")
g.set(ylim=(70, 95))
# %%
'''
Load Model
'''
from Models import HST_model
from Probability_helper import mixture_gaussian
def load_model_SD(Subject, LabelMode,Label_Code):
    if LabelMode == 'Mixture':
        prob_model = lambda x: mixture_gaussian(x, False, 3)
    else:
        prob_model = None

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
    
    ckpt_path = '/mnt/HDD/Benchmarks/DREAMER/ckpt/DREAMER_S{:02d}_{}_{}_{}'.format(Subject, 'HST', Label_Code, LabelMode)

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


# %%
from scipy.io import loadmat
from tensorflow.keras.models import Model
male_idx = [1,3,6,8,10,11,12,13,14,15,19,21,22,23]
female_idx = [2,4,5,7,9,16,17,18,20]

Yall = []
for i in range(1,24):
    tempY = loadmat('/mnt/HDD/Datasets/DREAMER/Labels.mat')['Y'][i-1]
    Yall.append(tempY)

Yall = np.array(Yall)
Yall_std = np.std(Yall, axis=0)
trial_idx = np.argmin(Yall_std, axis=0) # select the trial with the least std among subjects.

# Yall_std_male = np.std(Yall[np.array(male_idx)-1], axis=0)
# trial_idx_male = np.argmin(Yall_std_male, axis=0)

# Yall_std_female = np.std(Yall[np.array(female_idx)-1], axis=0)
# trial_idx_female = np.argmin(Yall_std_female, axis=0)
#%%
Subject_male = 1
Subject_female = 2
X_male = loadmat('/mnt/HDD/Datasets/DREAMER/S{:02d}_1min.mat'.format(Subject_male))['X'].transpose((1,0,2))
Y_male = Yall[Subject_male-1]
X_female = loadmat('/mnt/HDD/Datasets/DREAMER/S{:02d}_1min.mat'.format(Subject_female))['X'].transpose((1,0,2))
Y_female = Yall[Subject_female-1]

X_prep = lambda x: x/np.max(np.abs(x), axis=1, keepdims=True)
X_raw_male = X_prep(X_male)
X_raw_female = X_prep(X_female)

def get_feature_model(model):
    
    return Model(model.input, model.layers[-4].output)

X_male_V_rep = np.reshape(X_raw_male[trial_idx[0]], (60,128,14))
X_female_V_rep = np.reshape(X_raw_female[trial_idx[0]], (60,128,14))

X_male_A_rep = np.reshape(X_raw_male[trial_idx[1]], (60,128,14))
X_female_A_rep = np.reshape(X_raw_female[trial_idx[1]], (60,128,14))

def get_feature(X, Subject, LabelMode, Label_Code, Version = 0, Pretrained = True):
    model = load_model_SI(Subject, LabelMode, Label_Code,  0, Pretrained=True)
    feature_model = get_feature_model(model)
    feature = feature_model.predict(X)

    plt.matshow(feature[:,:14].T/np.max(np.abs(feature[:,:14].T), axis=0, keepdims=True), cmap='bwr')
    plt.colorbar()
    plt.matshow(feature[:, 14:-1].T/np.max(np.abs(feature[:,:14].T), axis=0, keepdims=True), cmap='bwr')
    plt.colorbar()
    plt.figure()
    plt.plot(feature[:,-1])

    return feature

# male_model = load_model_SI(Subject_male, 'OneHot','V',  0, Pretrained=True)
# male_feature_model = get_feature_model(male_model)
# male_feature = male_feature_model.predict(X_male_A_rep)

# plt.matshow(male_feature[:,:14].T, cmap='bwr')
# plt.matshow(male_feature[:, 14:-1].T, cmap='bwr')
# plt.figure()
# plt.plot(male_feature[:,-1])


# %%
def save_feature(LabelMode, Label_Code, Version = 0, Pretrained = True):
    f_stack = []
    for i in range(1,24):
        X = loadmat('/mnt/HDD/Datasets/DREAMER/S{:02d}_1min.mat'.format(i))['X'].transpose((1,0,2))
        Y = Yall[i-1]
        X_raw = X_prep(X)
        model = load_model_SI(i, LabelMode, Label_Code,  Version, Pretrained)
        feature_model = get_feature_model(model)
        if Label_Code == 'V':
            X_rep = np.reshape(X_raw[trial_idx[0]], (60,128,14))
            Y_rep = Y[trial_idx[0],0]
        elif Label_Code == 'A':
            X_rep = np.reshape(X_raw[trial_idx[1]], (60,128,14))
            Y_rep = Y[trial_idx[1],1]
        feature = feature_model.predict(X_rep)
        
        fig, ax = plt.subplots(3,1, figsize=(13,9))
        ax[0].matshow(feature[:,:14].T/np.max(np.abs(feature[:,:14].T), axis=0, keepdims=True), cmap='bwr')
        ax[1].matshow(feature[:, 14:-1].T/np.max(np.abs(feature[:,14:-1].T), axis=0, keepdims=True), cmap='bwr')
        ax[2].plot(feature[:,-1])
        plt.tight_layout()
        plt.title('{:02d}_{}_{}_Ver{}_{}_Y{}'.format(i, LabelMode, Label_Code, 
                                                   Version, str(Pretrained),
                                                    Y_rep))
        plt.savefig('/mnt/HDD/Benchmarks/DREAMER/Plots/' + 
                    'Plot{:02d}_{}_{}_V{}_{}.png'.format(i, LabelMode, Label_Code, 
                                                         Version, str(Pretrained)) )
        
        f_stack.append(feature)

    np.save('/mnt/HDD/Benchmarks/DREAMER/Plots/' + 
            'Array_{}_{}_V{}_{}.npy'.format(LabelMode, Label_Code, 
                                            Version, str(Pretrained)), np.array(f_stack))


        # plt.figure()
        # plt.matshow(feature[:,:14].T/np.max(np.abs(feature[:,:14].T), axis=0, keepdims=True), cmap='bwr')
        # plt.colorbar()
        # plt.title('{:02d}_{}_{}_V{}_{}_nodes'.format(i, LabelMode, Label_Code, Version, str(Pretrained)))
        # plt.savefig('/mnt/HDD/Benchmarks/DREAMER/PLots/' + '{:02d}_{}_{}_V{}_{}_nodes.png'.format(i, LabelMode, Label_Code, Version, str(Pretrained)))
        

        # plt.figure()
        # plt.matshow(feature[:, 14:-1].T/np.max(np.abs(feature[:,:14].T), axis=0, keepdims=True), cmap='bwr')
        # plt.colorbar()
        # plt.title('{:02d}_{}_{}_V{}_{}_Region'.format(i, LabelMode, Label_Code, Version, str(Pretrained)))
        # plt.savefig('/mnt/HDD/Benchmarks/DREAMER/PLots/' + '{:02d}_{}_{}_V{}_{}_Region.png'.format(i, LabelMode, Label_Code, Version, str(Pretrained)))
        
        # plt.figure()
        # plt.plot(feature[:,-1])
        # plt.title('{:02d}_{}_{}_V{}_{}_G'.format(i, LabelMode, Label_Code, Version, str(Pretrained)))

        # plt.savefig('/mnt/HDD/Benchmarks/DREAMER/PLots/' + '{:02d}_{}_{}_V{}_{}_G.png'.format(i, LabelMode, Label_Code, Version, str(Pretrained)))

# %%
def save_feature_SD(LabelMode, Label_Code, Version = 0, Pretrained = True):
    f_stack = []
    for i in range(1,24):
        X = loadmat('/mnt/HDD/Datasets/DREAMER/S{:02d}_1min.mat'.format(i))['X'].transpose((1,0,2))
        Y = Yall[i-1]
        X_raw = X_prep(X)

        model = load_model_SD(i, LabelMode, Label_Code)
        feature_model = get_feature_model(model)
        if Label_Code == 'V':
            X_rep = np.reshape(X_raw[trial_idx[0]], (60,128,14))
            Y_rep = Y[trial_idx[0],0]
        elif Label_Code == 'A':
            X_rep = np.reshape(X_raw[trial_idx[1]], (60,128,14))
            Y_rep = Y[trial_idx[1],1]
        feature = feature_model.predict(X_rep)
        
        channels = ['AF3','F7', 'F3', 'FC5',
            'T7', 'P7', 'O1', 'O2', 'P8',
            'T8', 'FC6', 'F4', 'F8', 'AF4'
        ]
        fig, ax = plt.subplots(3,1, figsize=(13,9))
        ax[0].matshow(feature[:,:14].T/np.max(np.abs(feature[:,:14].T), axis=0, keepdims=True), cmap='bwr')
        ax[0].set_yticks(np.arange(0,14), channels, fontsize=10)
        ax[0].set_ylabel('Channels')
        ax[0].grid(None)
        ax[1].matshow(feature[:, 14:-1].T/np.max(np.abs(feature[:,14:-1].T), axis=0, keepdims=True), cmap='bwr')
        ax[1].set_yticks(np.arange(0,4), ['FL', 'PL', 'PR', 'FR'])
        ax[1].set_ylabel('Regions')
        ax[1].grid(None)
        ax[-1].matshow(feature[:,-1:].T, cmap='bwr')
        ax[-1].grid(None)
        ax[-1].set_yticks([0],['Graph'])
        plt.tight_layout()
        plt.title('{:02d}_{}_{}_Ver{}_{}_Y{}_SD'.format(i, LabelMode, Label_Code, 
                                                   Version, str(Pretrained),
                                                    Y_rep))
        plt.savefig('/mnt/HDD/Benchmarks/DREAMER/Plots/' + 
                    'Plot{:02d}_{}_{}_V{}_{}_SD.png'.format(i, LabelMode, Label_Code, 
                                                         Version, str(Pretrained)) )
        
        f_stack.append(feature)

    np.save('/mnt/HDD/Benchmarks/DREAMER/Plots/' + 
            'Array_{}_{}_V{}_{}_SD.npy'.format(LabelMode, Label_Code, 
                                            Version, str(Pretrained)), np.array(f_stack))


# %%
from scipy.io import savemat
def npy2mat(filepath):
    _dict={}
    temp_array = np.load(filepath)
    for i in range(23):
        _dict['S'+str(i+1)] = temp_array[i]
    savemat(filepath[:-4]+'.mat', _dict)

# npyfilepaths = ['/mnt/HDD/Benchmarks/DREAMER/Plots/Array_OneHot_V_V0_False.npy',
#              '/mnt/HDD/Benchmarks/DREAMER/Plots/Array_OneHot_V_V0_True.npy',
#              '/mnt/HDD/Benchmarks/DREAMER/Plots/Array_OneHot_A_V0_True.npy',
#              '/mnt/HDD/Benchmarks/DREAMER/Plots/Array_OneHot_A_V0_False.npy',
#              '/mnt/HDD/Benchmarks/DREAMER/Plots/Array_OneHot_A_V0_True_SD.npy',
#              '/mnt/HDD/Benchmarks/DREAMER/Plots/Array_OneHot_V_V0_False_SD.npy']

npyfilepaths = [
             '/mnt/HDD/Benchmarks/DREAMER/Plots/Array_OneHot_V_V1_True.npy',
             '/mnt/HDD/Benchmarks/DREAMER/Plots/Array_OneHot_V_V2_True.npy',
             '/mnt/HDD/Benchmarks/DREAMER/Plots/Array_OneHot_A_V1_True.npy',
             '/mnt/HDD/Benchmarks/DREAMER/Plots/Array_OneHot_A_V2_True.npy'
             ]
for filepath in npyfilepaths:
    npy2mat(filepath)

# %%
'''
Plot Node location
'''
import pickle
with open('/home/dykuang/Codes/SEED/ch_pos_1020.pkl', 'rb') as pkl:
    pos_dict = pickle.load(pkl)

ch_list = [ 'AF3', 'F7', 'F3', 'FC5',
            'T7',  'P7',
             'O1', 'O2', 'P8', 'T8', 'FC6', 
            'F4', 'F8',  'AF4'
            ]

XY = []
for ch in ch_list:
    if ch in pos_dict.keys():
        XY.append(pos_dict[ch][:2])
XY = np.array(XY)
plt.figure(figsize=(6,6))
plt.plot(XY[:,0], XY[:,1], 'k.', markersize=20)
plt.grid(False)
plt.axis('off')
# %%
'''
make a plot for global level feature change
'''
# feature_file = '/mnt/HDD/Benchmarks/DREAMER/Plots/Array_OneHot_A_V0_True_SD.npy'
# feature_record = np.load(feature_file)

def make_time_plot(data):
    '''
    data: (subject, time)
    '''
    m = np.mean(data, axis=0)
    s = np.std(data, axis=0)
    t = np.linspace(0,1,60)
    fig,ax = plt.subplots()
    ax.plot(t, m , 'b-', label='mean')
    ax.fill_between(t, m+s, m-s, fc='red', alpha=0.2, linewidth=0, label='1-$\sigma$')
    ax.set_xlabel('time')
    ax.set_ylabel('feature value')
    ax.legend(loc='upper left')

# make_time_plot(feature_record[:,:,-1])

# make_time_plot(feature_record[[i-1 for i in male_idx],:,-1])
# make_time_plot(feature_record[[i-1 for i in female_idx],:,-1])

def make_time_plot_from_file(filepath):
    feature_record = np.load(filepath)
    make_time_plot(feature_record[:,:,-1])

    make_time_plot(feature_record[[i-1 for i in male_idx],:,-1])
    make_time_plot(feature_record[[i-1 for i in female_idx],:,-1])   

make_time_plot_from_file('/mnt/HDD/Benchmarks/DREAMER/Plots/Array_OneHot_V_V0_False_SD.npy') 
# %%
