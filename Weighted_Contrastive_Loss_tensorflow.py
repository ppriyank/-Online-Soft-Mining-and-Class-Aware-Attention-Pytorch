import tensorflow as tf
# Implementation of Deep Metric Learning by Online Soft Mining and Class-Aware Attention
# https://arxiv.org/pdf/1811.01459v2.pdf

class OSM_CAA_Loss():
    def __init__(self, alpha=1.2, l=0.5, use_gpu=True, batch_size=32):
        self.use_gpu = use_gpu
        self.alpha = 1.2 # margin of weighted contrastive loss, as mentioned in the paper 
        self.l = 0.5 #  hyperparameter controlling weights of positive set and the negative set  
        self.osm_sigma = 0.8  #\sigma OSM (0.8) as mentioned in paper
        self.n = batch_size 
        
    def forward(self, x, labels , embd):
        '''
        x : feature vector : (n x d)
        labels : (n,)
        embd : Fully Connected weights of classification layer (dxC), C is the number of classes: represents the vectors for class
        '''
        x = tf.math.l2_normalize(x, 1)
        n = self.n
        r = tf.ones([n, 1], tf.float32)
        #         r = tf.reduce_sum(x*x, 1)
        #         r = tf.reshape(r, [-1, 1])
        dist = r - 2*tf.matmul(x, tf.transpose(x)) + tf.transpose(r)
        dist = tf.math.sqrt(dist)
        
        p_mask = tf.cast(tf.equal(labels[:, tf.newaxis], labels[tf.newaxis, :]), tf.float32)
        n_mask = 1- p_mask
        
        S = tf.exp(-1 * dist / (self.osm_sigma * self.osm_sigma ) )
        S_  = tf.nn.relu(self.alpha - dist)
        S = S * p_mask
        S_ = S_ * n_mask
        S  = S + S_

        embd = tf.math.l2_normalize(embd, 0)
        denom = tf.reduce_sum(tf.exp(tf.matmul(x , embd)),1)
        num =  tf.exp (tf.reduce_sum( x * tf.transpose(tf.gather(embd , labels , axis=1)) , 1 ))
        
        atten_class = num / denom
        temp = tf.tile(tf.expand_dims(atten_class, 0),[n,1])
        A =  tf.math.minimum(temp , tf.transpose(temp))

        W = S * A 
        W_P = W * p_mask
        W_N = W * n_mask
        W_P = W_P * (1 - tf.eye(n))
        W_N = W_N * (1 - tf.eye(n))

        L_P =  tf.reduce_sum(W_P * tf.pow(dist,2) ) / (2 * tf.reduce_sum(W_P) )
        L_N  = tf.reduce_sum(W_N * tf.pow(S_ , 2) ) / (2 * tf.reduce_sum(W_N) )

        L = (1- self.l) * L_P + self.l * L_N

        return L 
