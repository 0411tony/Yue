#coding:utf8
from base.DeepRecommender import DeepRecommender
import numpy as np
import random
from tool import config
import tensorflow as tf
from tensorflow import set_random_seed
from collections import defaultdict

set_random_seed(2)

class APR(DeepRecommender):
    # APR：Adversarial Personalized Ranking for Recommendation
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(APR, self).__init__(conf,trainingSet,testSet,fold)

    def readConfiguration(self):
        super(APR, self).readConfiguration()
        args = config.LineConfig(self.config['APR'])
        self.eps = float(args['-eps'])
        self.regAdv = float(args['-regA'])
        self.advEpoch = int(args['-advEpoch'])
        self.negativeCount = 3

    def _create_variables(self):
        #perturbation vectors
        self.adv_U = tf.Variable(tf.zeros(shape=[self.m, self.k]),dtype=tf.float32, trainable=False)
        self.adv_V = tf.Variable(tf.zeros(shape=[self.n, self.k]),dtype=tf.float32, trainable=False)

        self.neg_idx = tf.placeholder(tf.int32, [None], name="n_idx")
        self.V_neg_embed = tf.nn.embedding_lookup(self.V, self.neg_idx)
        #parameters
        self.eps = tf.constant(self.eps,dtype=tf.float32)
        self.regAdv = tf.constant(self.regAdv,dtype=tf.float32)

    def _create_inference(self):
        result = tf.subtract(tf.reduce_sum(tf.multiply(self.U_embed, self.V_embed), 1),
                                  tf.reduce_sum(tf.multiply(self.U_embed, self.V_neg_embed), 1))
        return result

    def _create_adv_inference(self):
        self.U_plus_delta = tf.add(self.U_embed, tf.nn.embedding_lookup(self.adv_U, self.u_idx))
        self.V_plus_delta = tf.add(self.V_embed, tf.nn.embedding_lookup(self.adv_V, self.v_idx))
        self.V_neg_plus_delta = tf.add(self.V_neg_embed, tf.nn.embedding_lookup(self.adv_V, self.neg_idx))
        result = tf.subtract(tf.reduce_sum(tf.multiply(self.U_plus_delta, self.V_plus_delta), 1),
                             tf.reduce_sum(tf.multiply(self.U_plus_delta, self.V_neg_plus_delta), 1))
        return result

    def _create_adversarial(self):
        #get gradients of Delta
        self.grad_U, self.grad_V = tf.gradients(self.loss_adv, [self.adv_U,self.adv_V])

        # convert the IndexedSlice Data to Dense Tensor
        self.grad_U_dense = tf.stop_gradient(self.grad_U)
        self.grad_V_dense = tf.stop_gradient(self.grad_V)
        # normalization: new_grad = (grad / |grad|) * eps
        self.update_U = self.adv_U.assign(tf.nn.l2_normalize(self.grad_U_dense, 1) * self.eps)
        self.update_V = self.adv_V.assign(tf.nn.l2_normalize(self.grad_V_dense, 1) * self.eps)

    def _create_loss(self):
        self.reg_lambda = tf.constant(self.regU, dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.nn.softplus(-self._create_inference()))
        self.reg_loss = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.U_embed)),
                               tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V_embed)))
        
        self.reg_loss = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.U_embed)), self.reg_loss)
        self.total_loss = tf.add(self.loss, self.reg_loss)
        #loss of adversarial training
        self.loss_adv = tf.multiply(self.regAdv, tf.reduce_sum(tf.nn.softplus(-self._create_adv_inference())))
        self.loss_adv = tf.add(self.loss,self.loss_adv)

    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(self.lRate)
        self.train = self.optimizer.minimize(self.total_loss)
        self.optimizer_adv = tf.train.AdamOptimizer(self.lRate)
        self.train_adv = self.optimizer.minimize(self.loss_adv)

    def initModel(self):
        super(APR, self).initModel()

        self.userListen = defaultdict(dict)
        for user in self.data.userRecord:
            for item in self.data.userRecord[user]:
                if item[self.recType] not in self.userListen[user]:
                    self.userListen[user][item[self.recType]] = 1
                self.userListen[user][item[self.recType]] += 1   
        print ('training...')

        self._create_variables()
        self._create_loss()
        self._create_adversarial()
        self._create_optimizer()


    def next_batch(self):
        batch_idx = np.random.randint(self.train_size, size=self.batch_size)

        users = [self.data.trainingData[idx]['user'] for idx in batch_idx]
        items = [self.data.trainingData[idx]['track'] for idx in batch_idx]
        user_idx,item_idx=[],[]
        neg_item_idx = []

        for i,user in enumerate(users):
            for j in range(self.negativeCount): #negative sampling
                item_j = random.randint(0,self.n-1)
                while self.data.id2name['track'][item_j] in self.userListen[user]:
                    item_j = random.randint(0, self.n - 1)
                user_idx.append(self.data.getId(user, 'user'))
                item_idx.append(self.data.getId(items[i], 'track'))
                neg_item_idx.append(item_j)
        return user_idx,item_idx,neg_item_idx

    def buildModel(self):
        print ('training...')
        iteration = 0
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            # train the model until converged
            for epoch in range(self.maxIter):
                user_idx,item_idx,neg_item_idx = self.next_batch()
                _,loss = sess.run([self.train,self.total_loss],feed_dict={self.u_idx: user_idx, self.v_idx: item_idx, self.neg_idx:neg_item_idx})
                print ('iteration:', epoch, 'loss:',loss)

                self.P = sess.run(self.U)
                self.Q = sess.run(self.V)
                self.ranking_performance()
            # start adversarial training
            for epoch in range(self.advEpoch):
                user_idx,item_idx,neg_item_idx = self.next_batch()
                sess.run([self.update_U, self.update_V],
                         feed_dict={self.u_idx: user_idx, self.v_idx: item_idx, self.neg_idx: neg_item_idx})
                _,loss = sess.run([self.train_adv,self.loss_adv],feed_dict={self.u_idx: user_idx, self.v_idx: item_idx, self.neg_idx:neg_item_idx})
                print ('iteration:', epoch, 'loss:',loss)
                self.P = sess.run(self.U)
                self.Q = sess.run(self.V)
                self.ranking_performance()

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        uid = self.data.getId(u,'user')
        return np.divide(self.V.dot(self.U[uid]), self.normalized_U[uid]*self.normalized_V)

