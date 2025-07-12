import numpy as np
import tensorflow as tf
from . import eq_kernels_rfm
import time

class k2_intensity_estimator:

    def __init__(self, kernel):

        self.ker = kernel
    
    def fit(self, d_spk, d_region, a, b):
        elapse_t0 = time.time()
        
        d_region = tf.cast(d_region,d_spk.dtype)
        a = tf.cast(a,d_spk.dtype)
        b = tf.cast(b,d_spk.dtype)

        self.eq_ker = eq_kernels_rfm(self.ker,d_region,a,b)
        self.d_spk = d_spk
        
        return time.time() - elapse_t0

    def fit_ev(self, d_spk, d_region, a, b):
        elapse_t0 = time.time()
        
        d_region = tf.cast(d_region,d_spk.dtype)
        a = tf.cast(a,d_spk.dtype)
        b = tf.cast(b,d_spk.dtype)

        self.eq_ker = eq_kernels_rfm(self.ker,d_region,a,b)
        self.d_spk = d_spk

        ev = tf.reduce_sum(self.eq_ker.reduce_sum(self.d_spk,self.d_spk))
        ev -= tf.reduce_sum(tf.math.log(tf.sqrt(a)*tf.linalg.diag_part(self.eq_ker.chol)))
        
        return time.time() - elapse_t0, ev

    def fit_sc(self, d_spk, d_region, a, b, cv=None):
        
        d_region = tf.cast(d_region,d_spk.dtype)
        a = tf.cast(a,d_spk.dtype)
        b = tf.cast(b,d_spk.dtype)
        n_d_spk = tf.cast(d_spk.shape[0],d_spk.dtype)

        self.eq_ker = eq_kernels_rfm(self.ker,d_region,a,b)
        self.d_spk = d_spk
        score = 0.0

        if cv == 'ls':
            score1 = self.predict_integral_squared(d_region)
            score2 = tf.reduce_sum(self.eq_ker.reduce_sum(self.d_spk,self.d_spk)) \
                - tf.reduce_sum(self.eq_ker.psi(d_spk)**2)
            score = - (score1 - 2*score2 * n_d_spk/(n_d_spk-1.0))
        elif cv == 'll':
            score1 = self.predict_integral(d_region)
            zzz = self.eq_ker.reduce_sum(self.d_spk,self.d_spk)
            yyy = tf.reduce_sum(self.eq_ker.psi(d_spk)**2,0)
            score2 = tf.reduce_sum(tf.math.log(zzz-yyy))
            score = score2 - score1
            
        return score
    
    def predict(self, x):

        return self.eq_ker.reduce_sum(x,self.d_spk)
    
    def predict_integral(self, region):

        region = tf.cast(region,self.d_spk.dtype)
        
        return self.eq_ker.integral_reduce_sum(region,self.d_spk)

    def predict_integral_squared(self, region):

        region = tf.cast(region,self.d_spk.dtype)
        
        return self.eq_ker.integral_squared_reduce_sum(region,self.d_spk)
    
