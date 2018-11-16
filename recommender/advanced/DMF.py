#coding:utf8
from base.DeepRecommender import DeepRecommender
import math
import numpy as np
from tool import qmath
from tool import config
from random import choice
from random import shuffle
from tool.qmath import sigmoid
from math import log
from collections import defaultdict
from scipy.sparse import *
from scipy import *
import gensim.models.word2vec as w2v
from tool.qmath import cosine
from sklearn.manifold import TSNE
from time import time
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
import gc
import pickle
import tensorflow as tf

from tensorflow import set_random_seed
set_random_seed(2)

### implement by Xue et al., Deep Matrix Factorization Models for Recommender Systems, IJCAI 2017.

class DMF(DeepRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(DMF, self).__init__(conf,trainingSet,testSet,fold)
    
    def initModel(self):
        super(DMF, self).initModel()
        gc.collect()
        
        self.negative_sp = 5

        n_input_u = self.n
        n_input_i = self.m
        self.n_hidden_u=[256,512]
        self.n_hidden_i=[256,512]
        self.input_u = tf.placeholder("float", [None, n_input_u])
        self.input_i = tf.placeholder("float", [None, n_input_i])

        self.userListen = defaultdict(dict)
        for user in self.data.userRecord:
            for item in self.data.userRecord[user]:
                if item[self.recType] not in self.userListen[user]:
                    self.userListen[user][item[self.recType]] = 1
                self.userListen[user][item[self.recType]] += 1   
        print ('training...')

    def readConfiguration(self):
        super(DMF, self).readConfiguration()
        options = config.LineConfig(self.config['DMF'])
        self.alpha = float(options['-alpha'])
        self.topK = int(options['-k'])
        self.negCount = int(options['-neg'])
        
    def buildModel(self):      
        ######################  构建神经网络非线性映射   ####################################
        print ('the tensorflow...')
        initializer = tf.truncated_normal#tf.contrib.layers.xavier_initializer()
        #user net
        user_W1 = tf.Variable(initializer([self.n, self.n_hidden_u[0]],stddev=0.01))
        self.user_out = tf.nn.relu(tf.matmul(self.input_u, user_W1))
        self.regLoss = tf.nn.l2_loss(user_W1)
        for i in range(1, len(self.n_hidden_u)):
            W = tf.Variable(initializer([self.n_hidden_u[i-1], self.n_hidden_u[i]],stddev=0.01))
            b = tf.Variable(initializer([self.n_hidden_u[i]],stddev=0.01))
            self.regLoss = tf.add(self.regLoss, tf.nn.l2_loss(W))
            self.regLoss = tf.add(self.regLoss, tf.nn.l2_loss(b))
            self.user_out = tf.nn.relu(tf.add(tf.matmul(self.user_out, W), b))

        #item net
        item_W1 = tf.Variable(initializer([self.m, self.n_hidden_i[0]],stddev=0.01))
        self.item_out = tf.nn.relu(tf.matmul(self.input_i, item_W1))
        self.regLoss = tf.add(self.regLoss, tf.nn.l2_loss(item_W1))
        for i in range(1, len(self.n_hidden_i)):
            W = tf.Variable(initializer([self.n_hidden_i[i-1], self.n_hidden_i[i]],stddev=0.01))
            b = tf.Variable(initializer([self.n_hidden_i[i]],stddev=0.01))
            self.regLoss = tf.add(self.regLoss, tf.nn.l2_loss(W))
            self.regLoss = tf.add(self.regLoss, tf.nn.l2_loss(b))
            self.item_out = tf.nn.relu(tf.add(tf.matmul(self.item_out, W), b))

        norm_user_output = tf.sqrt(tf.reduce_sum(tf.square(self.user_out), axis=1))
        norm_item_output = tf.sqrt(tf.reduce_sum(tf.square(self.item_out), axis=1))

        self.y_ = tf.reduce_sum(tf.multiply(self.user_out, self.item_out), axis=1) / (
                norm_item_output * norm_user_output)
        self.y_ = tf.maximum(1e-6, self.y_)

        self.loss = self.r*tf.log(self.y_) + (1 - self.r) * tf.log(1 - self.y_)
        #tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_,labels=self.r)
        #self.loss = tf.nn.l2_loss(tf.subtract(self.y_,self.r))
        self.loss = -tf.reduce_sum(self.loss)
        reg_lambda = tf.constant(self.regU, dtype=tf.float32)
        self.regLoss = tf.multiply(reg_lambda,self.regLoss)
        self.loss = tf.add(self.loss,self.regLoss)

        optimizer = tf.train.AdamOptimizer(self.lRate).minimize(self.loss)

        self.U = np.zeros((self.m, self.n_hidden_u[-1]))
        self.V = np.zeros((self.n, self.n_hidden_i[-1]))

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        total_batch = int(len(self.data.trainingData)/ self.batch_size)
        for epoch in range(self.maxIter):
            shuffle(self.data.trainingData)
            for i in range(total_batch):
                users, items, ratings, u_idx, v_idx = self.next_batch(i)

                shuffle_idx = np.random.permutation(range(len(users)))
                users = users[shuffle_idx]
                items = items[shuffle_idx]
                ratings = ratings[shuffle_idx]
                u_idx = u_idx[shuffle_idx]
                v_idx = v_idx[shuffle_idx]

                _,loss= self.sess.run([optimizer, self.loss], feed_dict={self.input_u:users, self.input_i:items, self.r:ratings})
                #print (self.foldInfo, "Epoch:", '%03d' % (epoch + 1), "Batch:", '%03d' % (i + 1), "loss=", "{:.9f}".format(loss))

                U_embedding, V_embedding = self.sess.run([self.user_out, self.item_out], feed_dict={self.input_u:users, self.input_i:items})
                for ue,u in zip(U_embedding,u_idx):
                    self.U[u] = ue            
                for ve,v in zip(V_embedding,v_idx):
                    self.V[v] = ve
            self.normalized_V = np.sqrt(np.sum(self.V * self.V, axis=1))
            self.normalized_U = np.sqrt(np.sum(self.U * self.U, axis=1))
            self.ranking_performance()
        print("Optimization Finished!")    
    
    # return 1xm 的向量
    def row(self, u):
        user = self.data.id2name['user'][u]
        k = self.userListen[user].keys()
        v = self.userListen[user].values()
        vec = np.zeros(self.n)
        for pair in zip(k,v):
            iid = self.data.getId(pair[0], 'track')
            vec[iid] = pair[1]
        return vec

    # return 1xn 的向量
    def col(self,i):
        item = self.data.id2name['track'][i]
        k = self.data.listened['track'][item].keys()
        v = self.data.listened['track'][item].values()
        vec = np.zeros(self.m)
        for pair in zip(k, v):
            uid = self.data.getId(pair[0], 'user')
            vec[uid] = pair[1]
        return vec

    # 返回一个1xk的特征向量
    def col_Shallow(self,i):
        return self.Y[i]   

    def next_batch(self,i):
        rows = np.zeros(((self.negative_sp+1)*self.batch_size,self.n))
        cols = np.zeros(((self.negative_sp+1)*self.batch_size,self.m))
        #cols = np.zeros(((self.negative_sp+1)*self.batch_size,self.m))
        batch_idx = range(self.batch_size*i, self.batch_size*(i+1))

        u_idx = []
        v_idx = []
        ratings = []
        for idx in batch_idx:
            user = self.data.trainingData[idx]['user']
            item = self.data.trainingData[idx]['track']
            rating = 0
            if item in self.userListen[user]:
                rating = self.userListen[user][item]

            u_idx.append(self.data.getId(user, 'user'))
            v_idx.append(self.data.getId(item, 'track'))
            ratings.append(rating)

        for i,u in enumerate(u_idx):
            rows[i] = self.row(u)

        for i,t in enumerate(v_idx):
            cols[i] = self.col(t)
            #cols[i] = self.col_Shallow(t)

        #negative sample
        for i in range(self.negative_sp*self.batch_size):
            u = np.random.randint(self.m)
            v = np.random.randint(self.n)
            while self.data.id2name['track'][v] in self.userListen[self.data.id2name['user'][u]]:
                u = np.random.randint(self.m)
                v = np.random.randint(self.n)

            rows[self.batch_size-1+i]=self.row(u)
            cols[self.batch_size-1+i]=self.col(v)
            #cols[self.batch_size-1+i] = self.col_Shallow(v)
            u_idx.append(u)
            v_idx.append(v)
            ratings.append(0)
        return rows,cols,np.array(ratings),np.array(u_idx),np.array(v_idx)

        
    def predict(self, u):
        'invoked to rank all the items for the user'
        uid = self.data.getId(u,'user')
        return np.divide(self.V.dot(self.U[uid]), self.normalized_U[uid]*self.normalized_V)
