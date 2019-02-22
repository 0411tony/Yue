#coding:utf8
from base.DeepRecommender import DeepRecommender
import math
import numpy as np
from tool import config
from collections import defaultdict
from scipy.sparse import *
from scipy import *
import tensorflow as tf
from random import shuffle

np.random.seed(3)

class RRN(DeepRecommender):

    # Recurrent Recommender Networks. WSDM 2017
    # Chao-Yuan Wu, Amr Ahmed, Alex Beutel, Alexander J. Smola, How Jing
    
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(RRN, self).__init__(conf,trainingSet,testSet,fold)
    
    def initModel(self):
        super(RRN, self).initModel()
       
        self.negative_sp = 5
        self.n_step = 1
        self.userID = tf.placeholder(tf.int32, [None, 1], name='user_onehot')
        self.itemID = tf.placeholder(tf.int32, [None, 1], name='item_onehot')
        self.rating = tf.placeholder(tf.float32, [None,1], name="rating")
        self.dropout = tf.placeholder(tf.float32, name='dropout')

        self.U = np.zeros((self.m, self.k))
        self.V = np.zeros((self.n, self.k))

        self.userListen = defaultdict(dict)
        for user in self.data.userRecord:
            for item in self.data.userRecord[user]:
                if item[self.recType] not in self.userListen[user]:
                    self.userListen[user][item[self.recType]] = 0
                self.userListen[user][item[self.recType]] += 1   
        print ('training...')

    # def readConfiguration(self):
    #     super(RRN, self).readConfiguration()
    #     options = config.LineConfig(self.config['RRN'])
    #     self.alpha = float(options['-alpha'])
    #     self.topK = int(options['-k'])
    #     self.negCount = int(options['-neg'])
        
    def buildModel(self):      
        ######################  构建神经网络非线性映射   ####################################
        print ('the tensorflow...')
        with tf.name_scope("user_embedding"):
            # user id embedding
            uid_onehot = tf.reshape(tf.one_hot(self.userID, self.n), shape=[-1, self.n])
            # uid_onehot_rating = tf.multiply(self.rating, uid_onehot)
            uid_layer = tf.layers.dense(uid_onehot, units=128, activation=tf.nn.relu)
            self.uid_layer = tf.reshape(uid_layer, shape=[-1, self.n_step, 128])

        with tf.name_scope("item_embedding"):
            # movie id embedding
            vid_onehot = tf.reshape(tf.one_hot(self.itemID, self.m), shape=[-1, self.m])
            # mid_onehot_rating = tf.multiply(self.rating, mid_onehot)
            vid_layer = tf.layers.dense(vid_onehot, units=128, activation=tf.nn.relu)
            self.vid_layer = tf.reshape(vid_layer, shape=[-1, self.n_step, 128])
       
        with tf.variable_scope("user3_rnn_cell", reuse=tf.AUTO_REUSE):
            userCell = tf.nn.rnn_cell.GRUCell(num_units=128)
            userInput = tf.transpose(self.vid_layer, [1, 0, 2])
            userOutputs, userStates = tf.nn.dynamic_rnn(userCell, userInput, dtype=tf.float32)
            self.userOutput = userOutputs[-1]
        with tf.variable_scope("item3_rnn_cell", reuse=tf.AUTO_REUSE):
            itemCell = tf.nn.rnn_cell.GRUCell(num_units=128)
            itemInput = tf.transpose(self.uid_layer, [1, 0, 2])
            itemOutputs, itemStates = tf.nn.dynamic_rnn(itemCell, itemInput, dtype=tf.float32)
            self.itemOutput = itemOutputs[-1]

        user_W = tf.Variable(tf.random_normal(shape=[128, self.k], stddev=0.1))
        item_W = tf.Variable(tf.random_normal(shape=[128, self.k], stddev=0.1))
        user_b = tf.Variable(tf.random_normal(shape=[self.k], stddev=0.1))
        item_b = tf.Variable(tf.random_normal(shape=[self.k], stddev=0.1))

        self.U_embedding = tf.add(tf.matmul(self.userOutput, user_W), user_b)
        self.V_embedding = tf.add(tf.matmul(self.itemOutput, item_W), item_b)

        self.pred = tf.reduce_sum(tf.multiply(self.U_embedding, self.V_embedding), axis=1, keep_dims=True)

        loss = tf.losses.mean_squared_error(self.rating, self.pred)
        self.loss = tf.reduce_mean(loss)

        optimizer = tf.train.AdamOptimizer(self.lRate).minimize(self.loss)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        total_batch = int(len(self.data.trainingData)/ self.batch_size)
        for epoch in range(self.maxIter):
            user_idx, item_idx, ratings = self.next_batch()
            _,loss= self.sess.run([optimizer, self.loss], feed_dict={self.userID:user_idx, self.itemID:item_idx, self.rating:ratings, self.dropout:1.})
            print('iteration:', epoch, 'loss:', loss)
            #print (self.foldInfo, "Epoch:", '%03d' % (epoch + 1), "Batch:", '%03d' % (i + 1), "loss=", "{:.9f}".format(loss))
            U_embedding, V_embedding = self.sess.run([self.U_embedding, self.V_embedding], feed_dict={self.userID:user_idx, self.itemID:item_idx})
            for ue,u in zip(U_embedding,user_idx):
                self.U[u] = ue          
            for ve,v in zip(V_embedding,item_idx):
                self.V[v] = ve
            # self.normalized_V = np.sqrt(np.sum(self.V * self.V, axis=1))
            # self.normalized_U = np.sqrt(np.sum(self.U * self.U, axis=1))
            self.ranking_performance()
        print("Optimization Finished!")  

    def next_batch(self):
        batch_idx = np.random.randint(len(self.data.trainingData), size=self.batch_size)
        users = [self.data.trainingData[idx]['user'] for idx in batch_idx]
        items = [self.data.trainingData[idx]['track'] for idx in batch_idx]
        user_idx,item_idx=[],[]
        ratings = []
        for i,user in enumerate(users):
            uid = self.data.getId(user, 'user')
            tid = self.data.getId(items[i], 'track')
            rating = 0
            if items[i] in self.userListen[user]:
                rating = self.userListen[user][items[i]]
            user_idx.append([uid])
            item_idx.append([tid])
            ratings.append([rating])
        return np.array(user_idx), np.array(item_idx), np.array(ratings)
    
    def predict(self, u):
        'invoked to rank all the items for the user'
        uid = self.data.getId(u,'user')
        return self.V.dot(self.U[uid])

