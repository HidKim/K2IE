import tensorflow as tf
from . import kernels_rfm

class eq_kernels_rfm:
    
    def __init__(self, kernel, obs_region, a, b):

        ker = kernel

        # h(t,s) = <psi(t),psi(s)>, psi(t) = z.T * k_(t)
        self.integral_rfm_rfm = lambda r: ker.integral_rfm_rfm(r,b)
        self.chol = tf.linalg.cholesky(self.integral_rfm_rfm(obs_region)
                                       + 1./a*tf.eye(ker.nrf,dtype=obs_region.dtype))
        self.chol_inv = tf.linalg.cholesky_solve(self.chol, self.chol)
        self.psi = lambda x: tf.matmul(self.chol_inv,ker.rfm(x,b),transpose_a=True)
        self.psi_int = lambda reg: tf.matmul(self.chol_inv,ker.integral_rfm(reg,b),transpose_a=True)
        
    def eval(self, x, y):

        return tf.matmul(self.psi(x),self.psi(y),transpose_a=True)

    def eval_partial(self, x):

        return self.psi(x)

    def dot(self, x, y, v):

         return tf.matmul(self.psi(x),tf.matmul(self.psi(y),v),
                          transpose_a=True)
    
    def reduce_sum(self, x, y):

        z = tf.reduce_sum(self.psi(y),1,keepdims=True)
        return tf.matmul(self.psi(x),z,transpose_a=True)[:,0]
    
    def integral_reduce_sum(self, region, y):
        
        z = tf.reduce_sum(self.psi(y),1,keepdims=True)
        return tf.matmul(self.psi_int(region),z,transpose_a=True)[0,0]

    def integral_squared_reduce_sum(self, region, y, z=1.):

        y = tf.cast(y,region.dtype)
        z = tf.cast(z,region.dtype)
        
        v = tf.reduce_sum(self.psi(y)*tf.transpose(z),1,keepdims=True)
        q = tf.matmul(self.chol_inv,tf.matmul(self.integral_rfm_rfm(region),self.chol_inv),transpose_a=True)
        
        return tf.reduce_sum(v*tf.matmul(q,v))
    
    
