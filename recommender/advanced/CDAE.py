#coding:utf8

from base.IterativeRecommender import IterativeRecommender
import numpy as np
from random import choice,random
from tool import config
import tensorflow as tf
from collections import defaultdict
from tensorflow import set_random_seed
set_random_seed(2)

class CDAE(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(CDAE, self).__init__(conf,trainingSet,testSet,fold)

    def readConfiguration(self):
        super(CDAE, self).readConfiguration()
        eps = config.LineConfig(self.config['CDAE'])
        self.corruption_level = float(eps['-co'])
        self.n_hidden = int(eps['-nh'])
        self.batch_size = int(eps['-batch_size'])

    def initModel(self):
        super(CDAE, self).initModel()
        
        self.n_hidden = 128
        self.num_items = self.data.getSize(self.recType)
        self.num_users = self.data.getSize('user')

        self.negative_sp = 5
        initializer = tf.contrib.layers.xavier_initializer()
        self.X = tf.placeholder(tf.float32, [None, self.num_items])
        self.mask_corruption = tf.placeholder(tf.float32, [None, self.num_items])
        self.sample = tf.placeholder(tf.float32, [None, self.num_items])

        self.U = tf.Variable(initializer([self.num_users, self.n_hidden]))
        self.u_idx =  tf.placeholder(tf.int32, [None], name="u_idx")
        self.U_embed = tf.nn.embedding_lookup(self.U, self.u_idx)

        self.weights = {
            'encoder': tf.Variable(tf.random_normal([self.num_items, self.n_hidden])),
            'decoder': tf.Variable(tf.random_normal([self.n_hidden, self.num_items])),
        }

        self.biases = {
            'encoder': tf.Variable(tf.random_normal([self.n_hidden])),
            'decoder': tf.Variable(tf.random_normal([self.num_items])),
        }

        self.userListen = defaultdict(dict)
        for item in self.data.trainingData:
            uid = self.data.getId(item['user'], 'user')
            tid = self.data.getId(item['track'], 'track')
            if tid not in self.userListen[uid]:
                self.userListen[uid][tid] = 1
            else:
                self.userListen[uid][tid] += 1

    def encoder(self,x,v):
        layer = tf.nn.sigmoid(tf.matmul(x, self.weights['encoder'])+self.biases['encoder']+v)
        return layer

    def decoder(self,x):
        layer = tf.nn.sigmoid(tf.matmul(x, self.weights['decoder'])+self.biases['decoder'])
        return layer

    def row(self, u):
        k = self.userListen[u].keys()
        v = self.userListen[u].values()
        vec = np.zeros(self.num_items)
        for pair in zip(k,v):
            iid = pair[0]
            vec[iid] = pair[1]
        return vec

    def next_batch(self):
        X = np.zeros((self.batch_size, self.num_items))
        uids = []
        sample = np.zeros((self.batch_size, self.num_items))
        userList = list(self.data.name2id['user'].keys())
        itemList = list(self.data.name2id['track'].keys())
        for n in range(self.batch_size):
            user = choice(userList)
            uid = self.data.name2id['user'][user]
            uids.append(uid)
            vec = self.row(uid)
            ratedItems = self.userListen[uid].keys()
            values = self.userListen[uid].values()
            for iid in ratedItems:
                sample[n][iid]=1
            for i in range(self.negative_sp*len(ratedItems)):
                ng = choice(itemList)
                while ng in self.data.userRecord[user]:
                    ng = choice(itemList)
                ng_id = self.data.name2id['track'][ng]
                sample[n][ng_id]=1
            X[n]=vec
        return X, uids, sample

    def buildModel(self):
        self.corruption_input = tf.multiply(self.X, self.mask_corruption)
        self.encoder_op = self.encoder(self.corruption_input, self.U_embed)
        self.decoder_op = self.decoder(self.encoder_op)

        self.y_pred = tf.multiply(self.sample, self.decoder_op)
        y_true = tf.multiply(self.sample, self.corruption_input)
        self.y_pred = tf.maximum(1e-6, self.y_pred)
        
        self.loss = -tf.multiply(y_true,tf.log(self.y_pred))-tf.multiply((1-y_true),tf.log(1-self.y_pred))
        self.reg_loss = self.regU*(tf.nn.l2_loss(self.weights['encoder'])+tf.nn.l2_loss(self.weights['decoder'])+
                                   tf.nn.l2_loss(self.biases['encoder'])+tf.nn.l2_loss(self.biases['decoder']))

        self.reg_loss = self.reg_loss + self.regU*tf.nn.l2_loss(self.U_embed)
        self.loss = self.loss + self.reg_loss
        self.loss = tf.reduce_mean(self.loss)

        optimizer = tf.train.AdamOptimizer(self.lRate).minimize(self.loss)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        for epoch in range(self.maxIter):
            mask = np.random.binomial(1, self.corruption_level, (self.batch_size, self.num_items))
            batch_xs,users,sample = self.next_batch()

            _, loss,y = self.sess.run([optimizer, self.loss, self.y_pred], feed_dict={self.X: batch_xs,self.mask_corruption:mask,self.u_idx:users,self.sample:sample})

            print (self.foldInfo,"Epoch:", '%04d' % (epoch + 1),"loss=", "{:.9f}".format(loss))
        # self.ranking_performance()
        print("Optimization Finished!")


    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u,'user'):
            vec = self.row(u).reshape((1,len(self.data.TrackRecord)))
            uid = [self.data.name2id['user'][u]]
            return self.sess.run(self.decoder_op, feed_dict={self.X:vec,self.v_idx:uid})[0]
        else:
            return [self.data.globalMean] * len(self.data.TrackRecord)
