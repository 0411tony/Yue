#coding:utf8
from base.DeepRecommender import DeepRecommender
import numpy as np
import random
from tool import config
import tensorflow as tf
from tensorflow import set_random_seed
from collections import defaultdict
from scipy.sparse import coo_matrix,hstack

set_random_seed(2)

class DHCF(DeepRecommender):
    # Dual Channel Hypergraph Collaborative Filtering (KDD 2020).
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(DHCF, self).__init__(conf,trainingSet,testSet,fold)

    def buildAdjacencyMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            row += [self.data.getId(pair['user'], 'user')]
            col += [self.data.getId(pair['track'], 'track')]
            entries += [1]
        u_i_adj = coo_matrix((entries, (row, col)), shape=(self.m, self.n),dtype=np.float32)
        return u_i_adj

    def initModel(self):
        super(DHCF, self).initModel()

        self.negativeCount = 5

        self.userListen = defaultdict(dict)
        for entry in self.data.trainingData:
            if entry['track'] not in self.userListen[entry['user']]:
                    self.userListen[entry['user']][entry['track']] = 0
            self.userListen[entry['user']][entry['track']] += 1

        # Build adjacency matrix
        A = self.buildAdjacencyMatrix()
        # Build incidence matrix
        # H_u = hstack([A, A.dot(A.transpose().dot(A))])
        H_u = A
        D_u_v = H_u.sum(axis=1).reshape(1, -1)
        D_u_e = H_u.sum(axis=0).reshape(1, -1)
        temp1 = (H_u.transpose().multiply(np.sqrt(1.0/D_u_v))).transpose()
        temp2 = temp1.transpose()
        A_u = temp1.multiply(1.0/D_u_e).dot(temp2)
        A_u = A_u.tocoo()
        indices = np.mat([A_u.row, A_u.col]).transpose()
        H_u = tf.SparseTensor(indices, A_u.data.astype(np.float32), A_u.shape)

        H_i = A.transpose()
        D_i_v = H_i.sum(axis=1).reshape(1, -1)
        D_i_e = H_i.sum(axis=0).reshape(1, -1)
        temp1 = (H_i.transpose().multiply(np.sqrt(1.0 / D_i_v))).transpose()
        temp2 = temp1.transpose()
        A_i = temp1.multiply(1.0 / D_i_e).dot(temp2)
        A_i = A_i.tocoo()
        indices = np.mat([A_i.row, A_i.col]).transpose()
        H_i = tf.SparseTensor(indices, A_i.data.astype(np.float32), A_i.shape)

        # Build network
        self.isTraining = tf.placeholder(tf.int32)
        self.isTraining = tf.cast(self.isTraining, tf.bool)
        initializer = tf.contrib.layers.xavier_initializer()
        self.n_layer = 2
        self.weights = {}
        for i in range(self.n_layer):
            self.weights['layer_%d' %(i+1)] = tf.Variable(initializer([self.k, self.k]))

        user_embeddings = self.U
        item_embeddings = self.V
        all_user_embeddings = [user_embeddings]
        all_item_embeddings = [item_embeddings]

        def without_dropout(embedding):
            return embedding

        def dropout(embedding):
            return tf.nn.dropout(embedding, keep_prob=0.1)

        for i in range(self.n_layer):
            new_user_embeddings = tf.sparse_tensor_dense_matmul(H_u, self.U)
            new_item_embeddings = tf.sparse_tensor_dense_matmul(H_i, self.V)

            user_embeddings = tf.nn.leaky_relu(tf.matmul(new_user_embeddings, self.weights['layer_%d' %(i+1)])+ user_embeddings)
            item_embeddings = tf.nn.leaky_relu(tf.matmul(new_item_embeddings, self.weights['layer_%d' %(i+1)])+ item_embeddings)

            user_embeddings = tf.cond(self.isTraining, lambda: dropout(user_embeddings),
                                          lambda: without_dropout(user_embeddings))
            item_embeddings = tf.cond(self.isTraining, lambda: dropout(item_embeddings),
                                          lambda: without_dropout(item_embeddings))

            user_embeddings = tf.nn.l2_normalize(user_embeddings,axis=1)
            item_embeddings = tf.nn.l2_normalize(item_embeddings,axis=1)

            all_item_embeddings.append(item_embeddings)
            all_user_embeddings.append(user_embeddings)

        # user_embeddings = tf.reduce_sum(all_user_embeddings,axis=0)/(1+self.n_layer)
        # item_embeddings = tf.reduce_sum(all_item_embeddings, axis=0) / (1 + self.n_layer)
        #
        user_embeddings = tf.concat(all_user_embeddings,axis=1)
        item_embeddings = tf.concat(all_item_embeddings, axis=1)

        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.neg_item_embedding = tf.nn.embedding_lookup(item_embeddings, self.neg_idx)
        self.u_embedding = tf.nn.embedding_lookup(user_embeddings, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(item_embeddings, self.v_idx)
        self.test = tf.reduce_sum(tf.multiply(self.u_embedding,item_embeddings),1)

    def next_batch_pairwise(self):
        batch_id = 0
        while batch_id < self.train_size:
            if batch_id + self.batch_size <= self.train_size:
                users = [self.data.trainingData[idx]['user'] for idx in range(batch_id, self.batch_size + batch_id)]
                items = [self.data.trainingData[idx]['track'] for idx in range(batch_id, self.batch_size + batch_id)]
                batch_id += self.batch_size
            else:
                users = [self.data.trainingData[idx]['user'] for idx in range(batch_id, self.train_size)]
                items = [self.data.trainingData[idx]['track'] for idx in range(batch_id, self.train_size)]
                batch_id = self.train_size

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
            yield user_idx, item_idx, neg_item_idx

    def buildModel(self):
        print ('training...')
        y = tf.reduce_sum(tf.multiply(self.u_embedding, self.v_embedding), 1) \
            - tf.reduce_sum(tf.multiply(self.u_embedding, self.neg_item_embedding), 1)
        reg_loss = self.regU * (tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.v_embedding) +
                                                                    tf.nn.l2_loss(self.neg_item_embedding))
        for i in range(self.n_layer):
            reg_loss+= self.regU*tf.nn.l2_loss(self.weights['layer_%d' %(i+1)])
        loss = -tf.reduce_sum(tf.log(tf.sigmoid(y))) + reg_loss
        opt = tf.train.AdamOptimizer(self.lRate)
        train = opt.minimize(loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for iteration in range(self.maxIter):
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                _, l = self.sess.run([train, loss],
                                feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx, self.isTraining:1})
                print ('training:', iteration + 1, 'batch', n, 'loss:', l)

        
    def predict(self, u):
        'invoked to rank all the items for the user'
        if self.data.contains(u, 'user'):
            uid = self.data.name2id['user'][u]
            return self.sess.run(self.test, feed_dict={self.u_idx: [uid], self.isTraining:0})
        else:
            uid = self.data.getId(u,'user')
            return np.divide(self.V.dot(self.U[uid]), self.normalized_U[uid]*self.normalized_V)
