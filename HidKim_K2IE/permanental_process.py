import numpy as np
import tensorflow as tf
from . import eq_kernels_rfm
from .optimizer_adam import optimizer_adam
import time

class permanental_process:

    def __init__(self, kernel):

        self.ker = kernel
    
    def fit(self, d_spk, d_region, a, b, lr=0.001, eps=1.e-7, n_ite=1000, display=True):
        elapse_t0 = time.time()
        
        d_region = tf.cast(d_region,d_spk.dtype)
        a = tf.cast(a,d_spk.dtype)
        b = tf.cast(b,d_spk.dtype)
        
        self.eq_ker = eq_kernels_rfm(self.ker,d_region,a,b)
        self.psi = self.eq_ker.eval_partial(d_spk)
        self.d_spk = d_spk

        @tf.function()
        def loss(v):
            z = tf.matmul(self.psi,v)
            q = tf.matmul(self.psi,z,transpose_a=True)
            l1 = tf.reduce_sum(z*z)
            l2 = 2.*tf.reduce_sum(tf.math.log(tf.abs(q)+1.e-9))
            return l1 - l2

        # Initial value of v ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        area = tf.reduce_sum(tf.reduce_prod(d_region[:,:,1]-d_region[:,:,0],1))
        m_rate = tf.cast(tf.shape(d_spk)[0],d_spk.dtype) / area
        init_v = 1./tf.sqrt(m_rate) * tf.ones((tf.shape(d_spk)[0],1),dtype=d_spk.dtype)
        
        # Map estimation ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        opt = optimizer_adam()
        
        opt = optimizer_adam(lr=lr,dtype=d_spk.dtype)
        count, state = opt.reset(loss,init_v)
        i, conv = 0, False
        while i < n_ite - 1:
            count, state, slope = opt.minimize(loss,count,state)
            min_value = loss(state[0])
            if (i+1)%20 == 0 and display:
                aa = tf.strings.as_string(i+1,width=3)
                bb = tf.strings.as_string(min_value,precision=2,scientific=True)
                cc = tf.strings.as_string(slope,precision=2,scientific=True)
                t_pri = '\r#ite: '+aa+', loss: '+bb+', slope: '+cc
                tf.print(t_pri, end='')
            if slope < eps and i > 100:
                conv = True
                break
            i += 1
        aa = tf.strings.as_string(i+1,width=3)
        bb = tf.strings.as_string(loss(state[0]),precision=2,scientific=True)
        cc = tf.strings.as_string(slope,precision=2,scientific=True)
        t_pri = '\r#ite: '+aa+', loss: '+bb+', slope: '+cc
        if display:
            tf.print(t_pri)

        self.est_v = state[0]

        return time.time() - elapse_t0
        
    def predict(self, x):

        z = tf.matmul(self.psi,self.est_v)
        q = self.eq_ker.eval_partial(x)
        y = tf.matmul(q,z,transpose_a=True)[:,0]
        
        return y*y
        
    def predict_integral(self, region):

        region = tf.cast(region,self.d_spk.dtype)

        return self.eq_ker.integral_squared_reduce_sum(region,self.d_spk,self.est_v)
        
