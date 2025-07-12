import numpy as np
import tensorflow as tf
from scipy import stats

class kernels_rfm:

    def __init__(self, n_dim, kernel='gaussian', n_rand_feature=0, seed=0, qmc=False):

        self.nrf2 = n_rand_feature//2 # num of frequency samples 
        self.nrf = self.nrf2*2 # num of ramdom feature maps
        self.seed = seed
        self.n_dim = n_dim

        if kernel=='gaussian':
            self.func  = lambda x: tf.exp(-x**2)
            self.ifunc = lambda x,r: 0.5*tf.sqrt(tf.cast(np.pi,x.dtype))*(tf.math.erf(x-r[0])-tf.math.erf(x-r[1]))
        if kernel=='laplace':
            self.func  = lambda x: tf.exp(-tf.abs(x))
            self.ifunc = lambda x,r: tf.sign(x-r[1])*(tf.exp(-tf.abs(x-r[1]))-1.)-tf.sign(x-r[0])*(tf.exp(-tf.abs(x-r[0]))-1.)
        if kernel=='cauchy':
            self.func  = lambda x: 2./(1.+x**2)
            self.ifunc = lambda x,r: 2.*tf.atan(x-r[0]) - 2.*tf.atan(x-r[1])

        if self.nrf > 0 and qmc == False:
            rng = np.random.default_rng(seed)
            if kernel=='gaussian':
                self.omega = rng.normal(scale=np.sqrt(2.),
                                        size=(self.nrf2,self.n_dim))
            if kernel=='laplace':
                self.omega = rng.standard_cauchy(size=(self.nrf2,self.n_dim))
            if kernel=='cauchy':
                self.omega = rng.standard_exponential(size=(self.nrf2,self.n_dim))

        if self.nrf > 0 and qmc == True:
            sampler = stats.qmc.Halton(d=n_dim, scramble=True, seed=0)
            q = sampler.random(n=self.nrf2)
            q = stats.qmc.scale(q, l_bounds=[1.e-7]*n_dim, u_bounds=[1.-1.e-7]*n_dim)
            if kernel=='gaussian':
                self.omega = stats.norm.isf(1. - q,scale=np.sqrt(2.))
            if kernel=='laplace':
                self.omega = stats.cauchy.isf(1. - q)
            if kernel=='cauchy':
                self.omega = stats.expon.isf(1. - q)
        
        self.omega = tf.constant(self.omega,dtype=tf.float64)
            
    def eval_naive(self, x, y, b):
        
        y = tf.cast(y,x.dtype)
        b = tf.cast(b,x.dtype)
        
        if np.shape(b) == ():
            b = b*tf.ones((self.n_dim,),dtype=x.dtype)
        
        v = tf.ones((tf.shape(x)[0],tf.shape(y)[0]),dtype=x.dtype)
        for i in tf.range(self.n_dim):
            z = b[i]*(x[:,i][:,tf.newaxis]-y[:,i][tf.newaxis,:])
            v *= self.func(z)
                        
        return v
    
    def integral_eval_naive(self, x, region, b):

        b = tf.cast(b,x.dtype)
        region = tf.cast(region,x.dtype)

        if np.shape(b) == ():
            b *= tf.ones((self.n_dim,),dtype=x.dtype)

        v = tf.zeros((x.shape[0],),dtype=x.dtype)
        for reg in region:
            z = tf.ones((x.shape[0],),dtype=x.dtype)
            for i in tf.range(self.n_dim):
                z *= self.ifunc(b[i]*x[:,i],b[i]*reg[i])/b[i]
            v += z

        return v
    
    def eval(self, x, y, b):

        y = tf.cast(y,x.dtype)
        b = tf.cast(b,x.dtype)
        
        return tf.matmul(self.rfm(x,b),self.rfm(y,b),transpose_a=True)
        
    def rfm(self, x, b):
        
        b = tf.cast(b,x.dtype)
        omega = tf.cast(self.omega,x.dtype)

        ww = tf.concat([omega,omega],axis=0)
        d0 = tf.zeros((self.nrf2,tf.shape(x)[0]),dtype=x.dtype)
        d1 = tf.constant(-0.5*np.pi,dtype=x.dtype) + d0
        dd = tf.concat([d0,d1],axis=0)
        
        if np.shape(b) == ():
            b *= tf.ones((self.n_dim,),dtype=x.dtype)

        # [self.nrf, len(x)]
        an = tf.sqrt( 1. / tf.cast(self.nrf2,x.dtype) )
        bb = tf.tile(tf.expand_dims(b,axis=0),(tf.shape(ww)[0],1))
        phase = tf.matmul(bb*ww,x,transpose_b=True) + dd

        return an * tf.cos(phase)

    #@tf.function()
    def integral_rfm(self, region, b):
        
        b = tf.cast(b,region.dtype)
        omega = tf.cast(self.omega,region.dtype)

        ww = tf.concat([omega,omega],axis=0)
        d0 = tf.zeros((self.nrf2,1),dtype=region.dtype)
        d1 = tf.constant(-0.5*np.pi,dtype=region.dtype) + d0
        dd = tf.concat([d0,d1],axis=0)
        
        if np.shape(b) == ():
            b *= tf.ones((self.n_dim,),dtype=region.dtype)

        an = tf.sqrt( 1. / tf.cast(self.nrf2,region.dtype) )
        bb = tf.tile(tf.expand_dims(b,axis=0),(tf.shape(ww)[0],1))

        v = tf.zeros((self.nrf,1),dtype=region.dtype)
        for reg in region:
            T0, T1 = reg[:,0][:,tf.newaxis], reg[:,1][:,tf.newaxis]
            bwTd = 0.5*tf.matmul(bb*ww,T0+T1) + dd
            A1 = tf.cos(bwTd)
            bwT = 0.5*bb*ww*tf.transpose(T1-T0)
            A2 = tf.ones((self.nrf,1),dtype=region.dtype)
            for i in tf.range(self.n_dim):
                A2 *= (T1-T0)[i,0]*self.sinc(bwT[:,i][:,tf.newaxis])
            v += an * A1*A2
        
        # [self.nrf,]
        return v

    @tf.function()
    def integral_rfm_rfm(self, region, b):
        
        b = tf.cast(b,region.dtype)
        omega = tf.cast(self.omega,region.dtype)

        ww = tf.concat([omega,omega],axis=0)
        d0 = tf.zeros((self.nrf2,1),dtype=region.dtype)
        d1 = tf.constant(-0.5*np.pi,dtype=region.dtype) + d0
        dd = tf.concat([d0,d1],axis=0)
        
        if np.shape(b) == ():
            b *= tf.ones((self.n_dim,),dtype=region.dtype)

        an = 1. / tf.cast(self.nrf,region.dtype)
        bb = tf.tile(tf.expand_dims(b,axis=0),(tf.shape(ww)[0],1))

        
        v = tf.zeros((self.nrf,self.nrf),dtype=region.dtype)
        for reg in region:
            T0, T1 = reg[:,0][:,tf.newaxis], reg[:,1][:,tf.newaxis]
            bwTd = 0.5*tf.matmul(bb*ww,T0+T1) + dd
            A1, B1 = tf.cos(bwTd+tf.transpose(bwTd)), tf.cos(bwTd-tf.transpose(bwTd))
            bwT = 0.5*bb*ww*tf.transpose(T1-T0)
            A2 = tf.ones((self.nrf,self.nrf),dtype=region.dtype)
            B2 = tf.ones((self.nrf,self.nrf),dtype=region.dtype)
            for i in tf.range(self.n_dim):
                A2 *= (T1-T0)[i,0]*self.sinc(bwT[:,i][:,tf.newaxis]+bwT[:,i][tf.newaxis,:])
                B2 *= (T1-T0)[i,0]*self.sinc(bwT[:,i][:,tf.newaxis]-bwT[:,i][tf.newaxis,:])
            v += an * (A1*A2 + B1*B2)
        
        """
        v = tf.zeros((self.nrf,self.nrf),dtype=region.dtype)
        for reg in region:
            T0, T1 = reg[:,0][:,tf.newaxis], reg[:,1][:,tf.newaxis]
            bwTd = 0.5*tf.matmul(bb*ww,T0+T1) + dd
            A1, B1 = tf.cos(bwTd+tf.transpose(bwTd)), tf.cos(bwTd-tf.transpose(bwTd))
            bwT = 0.5*bb*ww*tf.transpose(T1-T0)
            bwT_exp1 = bwT[:,:,None]
            bwT_exp2 = tf.transpose(bwT)[None,:,:]
            A2 = tf.reduce_prod(self.sinc(tf.transpose(bwT_exp1+bwT_exp2,perm=[1,0,2])),axis=0)
            B2 = tf.reduce_prod(self.sinc(tf.transpose(bwT_exp1-bwT_exp2,perm=[1,0,2])),axis=0)
            v += an * (A1*A2 + B1*B2)
        """

        """
        @tf.function()
        def qqq(reg):
            T0, T1 = reg[:,0][:,tf.newaxis], reg[:,1][:,tf.newaxis]
            bwTd = 0.5*tf.matmul(bb*ww,T0+T1) + dd
            A1, B1 = tf.cos(bwTd+tf.transpose(bwTd)), tf.cos(bwTd-tf.transpose(bwTd))
            bwT = 0.5*bb*ww*tf.transpose(T1-T0)
            bwT_exp1 = bwT[:,:,None]
            bwT_exp2 = tf.transpose(bwT)[None,:,:]
            A2 = tf.reduce_prod(self.sinc(tf.transpose(bwT_exp1+bwT_exp2,perm=[1,0,2])),axis=0)
            B2 = tf.reduce_prod(self.sinc(tf.transpose(bwT_exp1-bwT_exp2,perm=[1,0,2])),axis=0)
            return A1*A2 + B1*B2
        v = tf.reduce_sum(an * tf.vectorized_map(qqq,region),axis=0)
        """
        
        # [self.nrf, self.nrf]
        return v
    
    def sinc(self, x):
        return tf.where(tf.greater(tf.abs(x), 1.e-7), tf.sin(x)/x, 1.-x**2/6.+x**4/120.)
    
