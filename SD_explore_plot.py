'''
For gathering benchmark results from 
the subject depedent benchmark
'''
#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
summary_folder = '/mnt/HDD/Benchmarks/DREAMER/Summary'
#%%
def fetch_subject_performance(model_token, encoding_type, label_token, subject):
    '''
    model_token: 'EEGNet', 'HST' or 'MSBAM'
    encoding_type: 'OneHot', 'OneHot_N', 'Mixture', 'Regression'
    label_token: 'V' or 'A'
    subject: int, subject index
    '''
    file = os.path.join(summary_folder, 'DREAMER_S{:02d}_{}_{}_{}.pkl'.format(subject, model_token, label_token, encoding_type) )

    try:
        with open(file, 'rb') as pkl:
            temp = pickle.load(pkl)    
    except:
       temp= None
       raise FileNotFoundError
    return temp

def gather_all_subject(model_token, encoding_type, label_token):
    
    summary = {
        'top_1_acc':[],
        'top_2_acc':[],
        'F1':[],
        'CM':[],
        'off_neibor':[],
        'closeness': []
    }
    for i in range(23):
        temp = fetch_subject_performance(model_token, encoding_type, label_token, i+1)
        for _k in ['top_1_acc', 'top_2_acc', 'F1','CM', 'off_neibor', 'closeness']:
            summary[_k].append(temp[_k])

    for _k in summary.keys():
        try:
            summary[_k] = np.array(summary[_k])
        except:
            summary[_k] = None

    return summary


#%%
# EEGNet_OneHot_V = gather_all_subject('EEGNet', 'OneHot', 'V')
# MSBAM_OneHot_V = gather_all_subject('MSBAM', 'OneHot', 'V')
# HST_OneHot_V = gather_all_subject('HST', 'OneHot', 'V')

# print('='*60)
# print('Mean : {:.02f}'.format( np.mean(np.mean(EEGNet_OneHot_V['top_1_acc'], axis=1))) )
# print('Std: {:.02f}'.format( np.std(np.mean(EEGNet_OneHot_V['top_1_acc'], axis=1)) ) )

# print('Mean: {:.02f}'.format( np.mean(np.mean(MSBAM_OneHot_V['top_1_acc'], axis=1))) )
# print('Std: {:.02f}'.format( np.std(np.mean(MSBAM_OneHot_V['top_1_acc'], axis=1)) ) )

# print('Mean: {:.02f}'.format( np.mean(np.mean(HST_OneHot_V['top_1_acc'], axis=1))) )
# print('Std: {:.02f}'.format( np.std(np.mean(HST_OneHot_V['top_1_acc'], axis=1)) ) )
# print('='*60)

def get_target_performance(encoding_type, label_token):

    P_EEGNet = gather_all_subject('EEGNet', encoding_type, label_token)
    P_MSBAM= gather_all_subject('MSBAM', encoding_type, label_token)
    P_HST = gather_all_subject('HST', encoding_type, label_token)
    
    print('='*60)
    for _key in ['top_1_acc', 'top_2_acc', 'F1']:
        
        print('EEGNet: {}'.format(_key))
        print('    Mean : {:.02f}'.format( np.mean(np.mean(P_EEGNet[_key], axis=1))) )
        print('    Std: {:.02f}'.format( np.std(np.mean(P_EEGNet[_key], axis=1)) ) )
        
        print('MSBAM: {}'.format(_key))
        print('    Mean: {:.02f}'.format( np.mean(np.mean(P_MSBAM[_key], axis=1))) )
        print('    Std: {:.02f}'.format( np.std(np.mean(P_MSBAM[_key], axis=1)) ) )

        print('HST: {}'.format(_key))
        print('    Mean: {:.02f}'.format( np.mean(np.mean(P_HST[_key], axis=1))) )
        print('    Std: {:.02f}'.format( np.std(np.mean(P_HST[_key], axis=1)) ) )

        print('\n')

    print('EEGNet: {}'.format('off_neibor'))
    print('    Mean : {:.02f}'.format( np.mean(np.mean(P_EEGNet['off_neibor'], axis=1)) /1000.0 *100)) # randomly generated 1000 test samples
    print('    Std: {:.02f}'.format( np.std(np.mean(P_EEGNet['off_neibor'], axis=1)) /1000.0 *100) ) 
    
    print('MSBAM: {}'.format('off_neibor'))
    print('    Mean: {:.02f}'.format( np.mean(np.mean(P_MSBAM['off_neibor'], axis=1))/1000.0 *100) )
    print('    Std: {:.02f}'.format( np.std(np.mean(P_MSBAM['off_neibor'], axis=1)) /1000.0 *100))

    print('HST: {}'.format('off_neibor'))
    print('    Mean: {:.02f}'.format( np.mean(np.mean(P_HST['off_neibor'], axis=1)) /1000.0*100))
    print('    Std: {:.02f}'.format( np.std(np.mean(P_HST['off_neibor'], axis=1)) /1000.0*100) )

    print('\n')
        
    print('EEGNet: closeness') # [chances that top 2 predictions are adjacent (I), (I) and top-1 prediction is true, (I) and true in top-2 prediction]
    print('    Mean: {}'.format( np.mean(np.mean(P_EEGNet['closeness'], axis=1), axis=0)) )
    print('    Std: {}'.format( np.std(np.mean(P_EEGNet['closeness'], axis=1), axis=0) )  )
    
    print('MSBAM: closeness')
    print('    Mean: {}'.format( np.mean(np.mean(P_MSBAM['closeness'], axis=1), axis=0) ) )
    print('    Std: {}'.format( np.std(np.mean(P_MSBAM['closeness'], axis=1), axis=0) )  )

    print('HST: closeness')
    print('    Mean: {}'.format( np.mean(np.mean(P_HST['closeness'], axis=1), axis=0) ) )
    print('    Std: {}'.format( np.std(np.mean(P_HST['closeness'], axis=1), axis=0 ) ) )    
        
    print('\n')

    print('='*60)

# %%
