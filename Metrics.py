'''
For containing some evaluation metrics
'''
import numpy as np

def count_off_neibor(M): 
    '''
    sum of elements that are not on the diagnal or off-diagnal
    '''  
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
    match: top1 pred and top2 pred are adjacent
    true_math: match & top1 pred is true
    true_match_2nd: match & true in top2 pred
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