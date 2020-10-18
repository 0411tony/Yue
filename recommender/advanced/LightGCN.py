from base.DeepRecommender import DeepRecommender
import tensorflow as tf
from math import sqrt
from tensorflow import set_random_seed
from collections import defaultdict
import random

set_random_seed(2)

class LightGCN(DeepRecommender):

    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(LightGCN, self).__init__(conf,trainingSet,testSet,fold)

    def initModel(self):
        super(LightGCN, self).initModel()

        self.negativeCount = 5

        self.userListen = defaultdict(dict)
        for entry in self.data.trainingData:
            if entry['track'] not in self.userListen[entry['user']]:
                    self.userListen[entry['user']][entry['track']] = 0
            self.userListen[entry['user']][entry['track']] += 1
        print('training...')

        ego_embeddings = tf.concat([self.U, self.V], axis=0)

        indices = [[self.data.getId(item['user'], 'user'), self.m + self.data.getId(item['track'], 'track')] for item in self.data.trainingData]
        indices += [[self.m + self.data.getId(item['track'], 'track'), self.data.getId(item['user'], 'user')] for item in self.data.trainingData]
        # values = [float(self.userListen[item['user']][item['track']]) / sqrt(len(self.data.userRecord[item['user']])) / sqrt(len(self.data.trackRecord[item['track']])) for item in self.data.trainingData]*2
        values = [float(self.userListen[item['user']][item['track']]) for item in self.data.trainingData]*2

        norm_adj = tf.SparseTensor(indices=indices, values=values, dense_shape=[self.m+self.n, self.m+self.n])

        self.n_layers = 3

        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            ego_embeddings = tf.sparse_tensor_dense_matmul(norm_adj, ego_embeddings)
            # normalize the distribution of embeddings.
            norm_embeddings = tf.nn.l2_normalize(ego_embeddings, axis=1)
            all_embeddings += [norm_embeddings]

        all_embeddings = tf.reduce_sum(all_embeddings, axis=0)

        self.multi_user_embeddings, self.multi_item_embeddings = tf.split(all_embeddings, [self.m, self.n], 0)

        self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
        self.neg_item_embedding = tf.nn.embedding_lookup(self.multi_item_embeddings, self.neg_idx)
        self.u_embedding = tf.nn.embedding_lookup(self.multi_user_embeddings, self.u_idx)
        self.v_embedding = tf.nn.embedding_lookup(self.multi_item_embeddings, self.v_idx)

        self.test = tf.reduce_sum(tf.multiply(self.u_embedding, self.multi_item_embeddings), 1)

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

            u_idx, i_idx, j_idx = [], [], []

            for i, user in enumerate(users):
                for j in range(self.negativeCount):
                    item_j = random.randint(0, self.n-1)
                    while self.data.id2name['track'][item_j] in self.userListen[user]:
                        item_j = random.randint(0, self.n-1)
                u_idx.append(self.data.getId(user, 'user'))
                i_idx.append(self.data.getId(items[i], 'track'))
                j_idx.append(item_j)

            yield u_idx, i_idx, j_idx


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
            for n, batch in enumerate(self.next_batch_pairwise()):
                user_idx, i_idx, j_idx = batch
                _, l = self.sess.run([train, loss],
                                feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx})
                print('training:', iteration + 1, 'batch', n, 'loss:', l)

    def predict(self, u):
        'invoked to rank all the items for the user'
        if self.data.contains(u, 'user'):
            uid = self.data.name2id['user'][u]
            return self.sess.run(self.test, feed_dict={self.u_idx: [uid]})
        else:
            uid = self.data.getId(u,'user')
            return np.divide(self.V.dot(self.U[uid]), self.normalized_U[uid]*self.normalized_V)
