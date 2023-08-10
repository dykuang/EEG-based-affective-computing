import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

# def Single_Trunc_Gaussian(params):
#     '''
#     Truncated Gaussian not implemeted yet
#     '''
#     loc, unscale = tf.split(params, num_or_size_splits=2, axis=-1)
#     scale = 1e-3 + tf.nn.softplus(params[:,1:2])
    

def KL_loss(Ytrue, Pred_dist):
    prior = tfd.Normal(loc = tf.cast(Ytrue, tf.float32), 
                       scale= 0.5*tf.ones(len(Ytrue), dtype=tf.float32 ) )
    # return tf.reduce_mean(tfp.distributions.kl_divergence(Pred_dist, prior)) #direction?
    return tf.reduce_mean(tfp.distributions.kl_divergence(prior, Pred_dist), axis=0) #direction?

def NLL_loss(Ytrue, Pred_dist):
    return -Pred_dist.log_prob(Ytrue)   

def hybrid_loss(Ytrue, Pred_dist):
    '''
    KL not implemented yet for mixture family
    '''
    # prior = tfp.distributions.Normal(Ytrue, 0.5*np.ones(len(Ytrue)))
    # alpha = 0.1
    # return -Pred_dist.log_prob(Ytrue) + \
    #         alpha*tf.reduce_mean( tfp.distributions.kl_divergence(prior, Pred_dist) )
    return NLL_loss(Ytrue, Pred_dist) + 0.01*KL_loss(Ytrue, Pred_dist) 

def Noise_loss(prior_type = tfd.Normal(loc = 0, scale= 0.25)):
    def my_loss(Ytrue, Ypred):
        return -prior_type.log_prob(Ypred-Ytrue)
    
    return my_loss