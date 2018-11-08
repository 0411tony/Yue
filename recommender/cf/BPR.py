#coding:utf8
from base.IterativeRecommender import IterativeRecommender
import math
import numpy as np
from tool import qmath
from random import choice
from tool.qmath import sigmoid
from math import log
from collections import defaultdict
import tensorflow as tf
from tensorflow import set_random_seed
import random

set_random_seed(2)

class BPR(IterativeRecommender):

    # BPR：Bayesian Personalized Ranking from Implicit Feedback
    # Steffen Rendle,Christoph Freudenthaler,Zeno Gantner and Lars Schmidt-Thieme

    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(BPR, self).__init__(conf,trainingSet,testSet,fold)

    def initModel(self):
        super(BPR, self).initModel()
        self.m = self.data.getSize('user')
        self.n = self.data.getSize(self.recType)
        self.train_size = len(self.data.trainingData)

    '''
    def buildModel(self):
        userListen = defaultdict(dict)
        for user in self.data.userRecord:
            for item in self.data.userRecord[user]:
                userListen[user][item[self.recType]] = 1
        
        print ('training...')
        iteration = 0
        itemList = list(self.data.name2id[self.recType].keys())
        while iteration < self.maxIter:
            self.loss = 0
            for user in self.data.userRecord:
                u = self.data.getId(user,'user')
                for item in self.data.userRecord[user]:
                    i = self.data.getId(item[self.recType],self.recType)
                    item_j = choice(itemList)
                    while (item_j in userListen[user]):
                        item_j = choice(itemList)
                    j = self.data.getId(item_j,self.recType)
                    s = sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))
                    self.P[u] += self.lRate * (1 - s) * (self.Q[i] - self.Q[j])
                    self.Q[i] += self.lRate * (1 - s) * self.P[u]
                    self.Q[j] -= self.lRate * (1 - s) * self.P[u]

                    self.P[u] -= self.lRate * self.regU * self.P[u]
                    self.Q[i] -= self.lRate * self.regI * self.Q[i]
                    self.Q[j] -= self.lRate * self.regI * self.Q[j]
                    self.loss += -log(s)
            self.loss += self.regU * (self.P * self.P).sum() + self.regI * (self.Q * self.Q).sum()
            iteration += 1
            if self.isConverged(iteration):
                break
    '''

    def next_batch(self):
        batch_idx = np.random.randint(len(self.data.trainingData), size=512)
        users = [self.data.trainingData[idx]['user'] for idx in batch_idx]
        items = [self.data.trainingData[idx]['track'] for idx in batch_idx]
        user_idx,item_idx=[],[]
        neg_item_idx = []
        for i,user in enumerate(users):
            uid = self.data.getId(user, 'user')
            for j in range(100): #negative sampling
                item_j = random.randint(0, self.n- 1)
                while item_j in self.userListen[uid]:
                    item_j = random.randint(0, self.n - 1)
                tid = self.data.getId(items[i], 'track')
                user_idx.append(uid)
                item_idx.append(tid)
                neg_item_idx.append(item_j)
        return user_idx, item_idx, neg_item_idx

    def buildModel(self):
        self.userListen = defaultdict(dict)
        for user in self.data.userRecord:
            uid = self.data.getId(user, 'user')
            for item in self.data.userRecord[user]:
                iid = self.data.getId(item[self.recType], 'track')
                if item[self.recType] not in self.userListen[user]:
                    self.userListen[uid][iid] = 0
                self.userListen[uid][iid] += 1   

        self.u_idx = tf.placeholder(tf.int32, [None], name="u_idx")
        self.v_idx = tf.placeholder(tf.int32, [None], name="v_idx")
        self.neg_idx = tf.placeholder(tf.int32, [None], name="n_idx")

        self.U = tf.Variable(tf.truncated_normal(shape=[self.m, self.k], stddev=0.005), name='U')
        self.V = tf.Variable(tf.truncated_normal(shape=[self.n, self.k], stddev=0.005), name='V')

        self.U_embed = tf.nn.embedding_lookup(self.U, self.u_idx)
        self.V_embed = tf.nn.embedding_lookup(self.V, self.v_idx)
        self.V_neg_embed = tf.nn.embedding_lookup(self.V, self.neg_idx)
        # 构造损失函数 设置优化器
        self.reg_lambda = tf.constant(self.regU, dtype=tf.float32)
      
        error = tf.subtract(tf.reduce_sum(tf.multiply(self.U_embed, self.V_embed), 1), tf.reduce_sum(tf.multiply(self.U_embed, self.V_neg_embed), 1))
        self.loss = tf.reduce_sum(tf.nn.softplus(-error))
        # 构造正则化项 完善损失函数
        self.reg_loss = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.U_embed)),
                               tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V_embed)))
        self.reg_loss = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V_neg_embed)), self.reg_loss)
        self.total_loss = tf.add(self.loss, self.reg_loss)
       
        self.optimizer = tf.train.AdamOptimizer(self.lRate)
        self.train = self.optimizer.minimize(self.total_loss)
        # 初始化会话
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            # 迭代，传递变量
            for epoch in range(self.maxIter):
                # 按批优化
                user_idx, item_idx, neg_item_idx = self.next_batch()
                _,loss = sess.run([self.train,self.total_loss],feed_dict={self.u_idx: user_idx, self.v_idx: item_idx, self.neg_idx:neg_item_idx})
                print ('iteration:', epoch, 'loss:',loss)
                # 输出训练完毕的矩阵
                self.P = sess.run(self.U)
                self.Q = sess.run(self.V)
                self.ranking_performance()
    
    def predict(self, u):
        'invoked to rank all the items for the user'
        u = self.data.getId(u,'user')
        return self.Q.dot(self.P[u])
