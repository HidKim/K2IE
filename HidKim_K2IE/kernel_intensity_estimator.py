import numpy as np
import tensorflow as tf
from . import eq_kernels_rfm
import time

class kernel_intensity_estimator:

    def __init__(self, kernel):

        self.ker = kernel
    
    def fit(self, d_spk, d_region, b):
                
        d_region = tf.cast(d_region,d_spk.dtype)
        b = tf.cast(b,d_spk.dtype)
        
        self.ikern = lambda x: self.ker.integral_eval_naive(x,d_region,b)
        self.kern  = lambda x,y: self.ker.eval_naive(x,y,b)
        self.d_spk = d_spk
        
        return 0.0

    def fit_sc(self, d_spk, d_region, b, cv=None, n_sample=1000):

        d_region = tf.cast(d_region,d_spk.dtype)
        b = tf.cast(b,d_spk.dtype)
        n_d_spk = tf.cast(d_spk.shape[0],d_spk.dtype)
        
        self.ikern = lambda x: self.ker.integral_eval_naive(x,d_region,b)
        self.kern  = lambda x,y: self.ker.eval_naive(x,y,b)
        self.d_spk = d_spk

        score =	0.0
        if cv == 'ls':
            zzz = self.kern(self.d_spk,self.d_spk) / self.ikern(self.d_spk)[:,tf.newaxis]
            score = tf.reduce_sum(zzz) - tf.linalg.trace(zzz)
            score *= 2.0*n_d_spk/(n_d_spk-1.0)
            fun = lambda x: self.predict(x)**2
            for reg in d_region:
                score -= self.mc_integration(fun,reg,n_sample=n_sample)

        elif cv == 'll':
            zzz = self.kern(self.d_spk,self.d_spk) / self.ikern(self.d_spk)[:,tf.newaxis]
            score = tf.math.log(tf.reduce_sum(zzz,1) - tf.linalg.diag_part(zzz))
            score = tf.reduce_sum(score)
            fun = lambda x: self.predict(x)
            for reg in d_region:
                score -= self.mc_integration(fun,reg,n_sample=n_sample)
            
        return score
    
    def predict(self, x):

        z = tf.reduce_sum(self.kern(x,self.d_spk),1) / self.ikern(x)

        return z
        
    def mc_integration(self, func, region, n_sample, seed=0):

     rng = np.random.default_rng(seed)
     p = np.array([rng.uniform(r[0],r[1],n_sample) for r in region]).T
     return np.mean(func(p))*np.prod(region[:,1]-region[:,0])
