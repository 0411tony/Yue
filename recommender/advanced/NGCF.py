from base.DeepRecommender import DeepRecommender
from random import choice
import tensorflow as tf
import numpy as np
from math import sqrt
from tensorflow import set_random_seed
from collections import defaultdict

set_random_seed(2)

class NGCF(DeepRecommender):

    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(NGCF, self).__init__(conf,trainingSet,testSet,fold)

    def next_batch(self):
        batch_id = 0
        while batch_id < self.train_size:
            if batch_id + self.batch_size <= self.train_size:
                users = [self.data.trainingData[idx]['user'] for idx in range(batch_id, self.batch_size + batch_id)]
                items = [self.data.trainingData[idx]['track'] for idx in range(batch_id, self.batch_size + batch_id)]
                # 可以不断迭代
                batch_id += self.batch_size
            else:
                users = [self.data.trainingData[idx]['user'] for idx in range(batch_id, self.train_size)]
                items = [self.data.trainingData[idx]['track'] for idx in range(batch_id, self.train_size)]
                # 当训练数据小于batch_id时，让batch_id最大为trainingData_id
                batch_id = self.train_size

            u_idx, i_idx, j_idx = [], [], []
            item_list = list(self.data.trackRecord.keys())
            for i, user in enumerate(users):
                u_idx.append(self.data.getId(user, 'user'))
                i_idx.append(self.data.getId(items[i], 'track'))

                neg_item = choice(item_list)
                while neg_item in self.data.userRecord[user]:
                    neg_item = choice(item_list)
                j_idx.append(self.data.getId(neg_item, 'track'))

            yield u_idx, i_idx, j_idx


    def initModel(self):
        super(NGCF, self).initModel()

        # 构建流程图
        self.userListen = defaultdict(dict)
        for entry in self.data.trainingData:
            user = entry['user']
            track = entry['track']
            if track not in self.userListen[user]:
                self.userListen[user][track] = 1
            self.userListen[user][track] += 1
            
        self.isTraining = tf.placeholder(tf.int32)
        self.isTraining = tf.cast(self.isTraining, tf.bool)
        self.user_embeddings = self.U
        self.item_embeddings = self.V
        ego_embeddings = tf.concat([self.user_embeddings,self.item_embeddings], axis=0)

        indices = [[self.data.getId(entry['user'], 'user'), self.m+self.data.getId(entry['track'], 'track')] for entry in self.data.trainingData]
        indices += [[self.m+self.data.getId(entry['user'], 'user'), self.data.getId(entry['track'], 'track')] for entry in self.data.trainingData]
        # 评分(1/sqrt(Nu)/sqrt(Ni))
        values = []
        for entry in self.data.trainingData:
            if len(self.data.userRecord[entry['user']]) == 0 or len(self.data.trackRecord[entry['track']]) == 0:
                values.append(0)
            else:
                values.append(float(self.userListen[entry['user']][entry['track']])/sqrt(len(self.data.userRecord[entry['user']]))/sqrt(len(self.data.trackRecord[entry['track']])))
        values = values*2

        norm_adj = tf.SparseTensor(indices=indices, values=values, dense_shape=[self.m+self.n, self.m+self.n])

        self.weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()
        weight_size = [self.k, self.k, self.k] #can be changed
        weight_size_list = [self.k] + weight_size

        self.n_layers = 3

        #initialize parameters
        for k in range(self.n_layers):
            self.weights['W_%d_1' % k] = tf.Variable(
                initializer([weight_size_list[k], weight_size_list[k + 1]]), name='W_%d_1' % k)
            self.weights['W_%d_2' % k] = tf.Variable(
                initializer([weight_size_list[k], weight_size_list[k + 1]]), name='W_%d_2' % k)

        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            side_embeddings = tf.sparse_tensor_dense_matmul(norm_adj, ego_embeddings)
            sum_embeddings = tf.matmul(side_embeddings + ego_embeddings, self.weights['W_%d_1' % k])
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            bi_embeddings = tf.matmul(bi_embeddings, self.weights['W_%d_2' % k])

            ego_embeddings = tf.nn.leaky_relu(sum_embeddings + bi_embeddings)

            # message dropout.
            def without_dropout():
                return ego_embeddings
            def dropout():
                return tf.nn.dropout(ego_embeddings, keep_prob=0.9)

            ego_embeddings = tf.cond(self.isTraining, lambda:dropout(), lambda:without_dropout())

            # normalize the distribution of embeddings.
            norm_embeddings = tf.nn.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        self.multi_user_embeddings, self.multi_item_embeddings = tf.split(all_embeddings, [self.m, self.n], 0)

        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.neg_item_embedding = tf.nn.embedding_lookup(self.multi_item_embeddings, self.neg_idx)
        self.u_embedding = tf.nn.embedding_lookup(self.multi_user_embeddings, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.multi_item_embeddings, self.v_idx)

        self.test = tf.reduce_sum(tf.multiply(self.u_embedding,self.multi_item_embeddings),1)

    def buildModel(self):

        y = tf.reduce_sum(tf.multiply(self.u_embedding, self.v_embedding), 1) \
            - tf.reduce_sum(tf.multiply(self.u_embedding, self.neg_item_embedding), 1)

        loss = -tf.reduce_sum(tf.log(tf.sigmoid(y))) + self.regU * (tf.nn.l2_loss(self.u_embedding) +
                                                                    tf.nn.l2_loss(self.v_embedding) +
                                                                    tf.nn.l2_loss(self.neg_item_embedding))
        opt = tf.train.AdamOptimizer(self.lRate)

        train = opt.minimize(loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)
        for iteration in range(self.maxIter):
            for n, batch in enumerate(self.next_batch()):
                user_idx, i_idx, j_idx = batch
                _, l = self.sess.run([train, loss],
                                feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx,self.isTraining:1})
                print('training:', iteration + 1, 'batch', n, 'loss:', l)

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if u in self.data.userRecord:
            u = self.data.getId(u, 'user')
            return self.sess.run(self.test,feed_dict={self.u_idx:u,self.isTraining:0})
        else:
            return [self.data.globalMean] * self.n