'''
For making probability models
'''
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb


def mixture_logistic(out, discrete=True, num=3): # 3 params per base distribution
    '''
    For a mixed discretized logistic distribution. If sampling, outputs are discrete values
    '''
    loc, un_scale, logits = tf.split(out, num_or_size_splits=num, axis=-1)
    scale = tf.nn.softplus(un_scale)

    if discrete:
        logistic_dist = tfd.QuantizedDistribution(
            distribution =tfd.TransformedDistribution(            # not sure about the necessity here
                distribution = tfd.Logistic(loc=loc, scale=scale),    
                # bijector = tfb.AffineScalar(shift=-0.5) ), #deprecated
                bijector = tfb.Shift(shift=-0.5)   ), # for the purpose of making integer in the center?
            low = 0,
            high = 4)  # the score
    else:
        logistic_dist =  tfd.Logistic(loc=loc, scale=scale)      # not sure about the necessity here
      
    mixture_dist = tfd.MixtureSameFamily(
        mixture_distribution = tfd.Categorical(logits = logits),
        components_distribution=logistic_dist)
    
    return mixture_dist

def mixture_gaussian(out, discrete = True, num=3): # 3 params per base distribution
    '''
    For a mixed gaussian distribution. 
    '''
    loc, un_scale, logits = tf.split(out, num_or_size_splits=num, axis=-1)
    scale = 1e-3 + tf.nn.softplus(0.05*un_scale) # preventing nan value

    if discrete:
        dist = tfd.QuantizedDistribution(
            distribution =tfd.TransformedDistribution(            # not sure about the necessity here
                distribution = tfd.Normal(loc=loc, scale=scale),  
                # distribution = tfd.TruncatedNormal(loc=loc, scale=scale),  
                # bijector = tfb.AffineScalar(shift=-0.5) ), #deprecated
                bijector = tfb.Shift(shift=-0.5)   ),
            low = 0,
            high = 4)  # the score
    else:
        dist =  tfd.Normal(loc=loc, scale=scale)
    mixture_dist = tfd.MixtureSameFamily(
        mixture_distribution = tfd.Categorical(logits = logits),
        components_distribution = dist)
    
    return mixture_dist

def Single_Gaussian(params, discrete=True):
    loc, unscale = tf.split(params, num_or_size_splits=2, axis=-1)
    scale = 1e-3 + tf.nn.softplus(0.05*unscale)

    if not discrete:
        return tfd.Normal(loc=loc,
                        scale=scale)
    else:
        return tfd.QuantizedDistribution(
                    distribution =tfd.TransformedDistribution(            # not sure about the necessity here
                    distribution = tfd.Normal(loc=loc, scale=scale),  
                    # distribution = tfd.TruncatedNormal(loc=loc, scale=scale),  
                    # bijector = tfb.AffineScalar(shift=-0.5) ), #deprecated
                    bijector = tfb.Shift(shift=-0.5)   ),
                low = 0,
                high = 4)   #no KL defined for quantized distribution yet

def Two_Gaussians(params):
    loc = tf.stack([params[:,:1], params[:,:1] + 1.0], axis=-1)
    unscale = params[:,1:3]
    logits = params[:,3:5]

    scale = 1e-3 + tf.nn.softplus(0.05*unscale)

    mixture_dist = tfd.MixtureSameFamily(
                    mixture_distribution = tfd.Categorical(logits = logits),
                    components_distribution = tfd.Normal(loc=loc, scale=scale))
    
    return mixture_dist