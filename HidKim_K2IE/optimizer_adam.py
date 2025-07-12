import tensorflow as tf

class optimizer_adam:
    
    def __init__(self, lr=0.001, beta_1=0.9,
                 beta_2=0.999, epsilon=1.e-07, dtype=tf.float64):
        self.lr, self.beta_1, self.beta_2, self.epsilon \
            = tf.cast(lr,dtype), tf.cast(beta_1,dtype), \
            tf.cast(beta_2,dtype), tf.cast(epsilon,dtype)
        self.dtype= dtype

    def reset(self, func, x):
        m = tf.zeros(tf.shape(x),self.dtype)
        v = tf.zeros(tf.shape(x),self.dtype)
        x = tf.cast(x,self.dtype)

        return tf.cast(0.,self.dtype), tf.stack([x,m,v])
    
    def minimize(self,func, t, state, par=1.):

        mask = tf.cast(par,self.dtype)
        x, m, v = tf.cast(state[0],self.dtype), tf.cast(state[1],self.dtype), tf.cast(state[2],self.dtype)

        with tf.GradientTape() as tape:
            tape.watch(x)
            f = func(x)
        g = tape.gradient(f,x) * mask

        t += 1.
        m = self.beta_1 * m + (1.-self.beta_1) * g
        v = self.beta_2 * v + (1.-self.beta_2) * tf.pow(g,2.0)
        mm = m / (1. - tf.pow(self.beta_1,t))
        vv = v / (1. - tf.pow(self.beta_2,t))

        f0 = func(x)
        x = x - self.lr * mm / (tf.sqrt(vv) + self.epsilon)
        f1 = func(x)
        
        slope = tf.abs((f0-f1)/f1)
        
        return t, tf.stack([x,m,v]), slope
