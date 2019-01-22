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
        self.corruption = float(eps['-co'])
        self.n_hidden = int(eps['-nh'])
        self.batch_size = int(eps['-batch_size'])

    def initModel(self):
        super(CDAE, self).initModel()
        n_input = self.data.getSize(self.recType)
        self.n_hidden = 128
        n_output = self.data.getSize(self.recType)
        self.negative_sp = 5
        self.X = tf.placeholder("float", [None, n_input])
        self.sample = tf.placeholder("bool", [None, n_input])
        self.zeros = np.zeros((self.batch_size, n_input))
        self.V = tf.Variable(tf.random_normal([self.data.getSize('user'), self.n_hidden]))
        self.v_idx = tf.placeholder(tf.int32, [None], name="v_idx")
        self.V_embed = tf.nn.embedding_lookup(self.V, self.v_idx)
        self.n = n_input

        self.weights = {
            'encoder': tf.Variable(tf.random_normal([n_input, self.n_hidden])),
            'decoder': tf.Variable(tf.random_normal([self.n_hidden, n_output])),
        }

        self.biases = {
            'encoder': tf.Variable(tf.random_normal([self.n_hidden])),
            'decoder': tf.Variable(tf.random_normal([n_output])),
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
        layer = tf.nn.sigmoid(tf.add(tf.add(tf.matmul(x, self.weights['encoder']), self.biases['encoder']),v))
        return layer

    def decoder(self,x):
        layer = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder']),self.biases['decoder']))
        return layer

    def row(self, u):
        k = self.userListen[u].keys()
        v = self.userListen[u].values()
        vec = np.zeros(self.getSize(self.recType))
        for pair in zip(k,v):
            iid = pair[0]
            vec[iid] = pair[1]
        return vec

    def next_batch(self):
        X = np.zeros((self.batch_size, self.data.getSize(self.recType)))
        uids = []
        evaluated = np.zeros((self.batch_size, self.data.getSize(self.recType)))>0
        userList = self.data.name2id['user'].keys()
        itemList = self.data.name2id['track'].keys()
        for n in range(self.batch_size):
            sample = []
            user = choice(userList)
            uid = self.data.name2id['user'][user]
            uids.append(uid)
            vec = self.row(uid)
            #corrupt
            ratedItems = self.userListen[uid].keys()
            values = self.userListen[uid].values()
            for iid in ratedItems:
                if random()>self.corruption:
                    vec[iid]=0
                evaluated[n][iid]=True
            for i in range(self.negative_sp*len(ratedItems)):
                ng = choice(itemList)
                while ng in self.data.userRecord[user]:
                    ng = choice(itemList)
                ng = self.data.name2id['track'][ng]
                evaluated[n][ng]=True
            X[n]=vec
        return X, uids, evaluated

    def buildModel(self):

        self.encoder_op = self.encoder(self.X, self.V_embed)
        self.decoder_op = self.decoder(self.encoder_op)

        y_pred = tf.where(self.sample, self.decoder_op, self.zeros)
        y_true = tf.where(self.sample, self.X, self.zeros)
        
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred,labels=y_true)
        self.loss = tf.reduce_mean(self.loss)
        reg_lambda = tf.constant(self.regU, dtype=tf.float32)

        self.reg_loss = tf.add(tf.add(tf.multiply(reg_lambda, tf.nn.l2_loss(self.weights['encoder'])),
                               tf.multiply(reg_lambda, tf.nn.l2_loss(self.weights['decoder']))),
                               tf.add(tf.multiply(reg_lambda, tf.nn.l2_loss(self.biases['encoder'])),
                               tf.multiply(reg_lambda, tf.nn.l2_loss(self.biases['decoder']))))

        self.reg_loss = tf.add(self.reg_loss,tf.multiply(reg_lambda,tf.nn.l2_loss(self.V_embed)))
        self.loss = tf.add(self.loss,self.reg_loss)

        optimizer = tf.train.AdamOptimizer(self.lRate).minimize(self.loss)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        total_batch = int(len(self.data.userRecord) / self.batch_size)

        for epoch in range(self.maxIter):
            for i in range(total_batch):
                mask = np.random.binomial(1, self.corruption, (self.batch_size, self.n))
                batch_xs,users,sample = self.next_batch()

                _, loss = self.sess.run([optimizer, self.loss], feed_dict={self.X: batch_xs,self.mask_corruption:mask,self.v_idx:users,self.sample:sample})

                print (self.foldInfo,"Epoch:", '%04d' % (epoch + 1),"Batch:", '%03d' %(i+1),"loss=", "{:.9f}".format(loss))
            self.ranking_performance()
        print("Optimization Finished!")


    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u,'user'):
            vec = self.row(u).reshape((1,len(self.data.TrackRecord)))
            uid = [self.data.name2id['user'][u]]
            return self.sess.run(self.decoder_op, feed_dict={self.X:vec,self.v_idx:uid})[0]
        else:
            return [self.data.globalMean] * len(self.data.TrackRecord)

