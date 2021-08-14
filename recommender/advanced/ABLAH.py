#coding:utf8
from base.IterativeRecommender import IterativeRecommender
import numpy as np
import tensorflow as tf
import math
from tool import qmath
from random import choice
from tool.qmath import sigmoid
from math import log
from collections import defaultdict
from tool import config
import pickle
import networkx as nx
import os

# 先找到所有数据，构建词袋
# 构建网络，向网络中添加节点
# 向网络中的两个节点之间添加边的关系，生成路径
# 根据词袋，以及训练集，测试集，获得content和标签
#
class Dataset(object):
    """docstring for ClassName"""
    def __init__(self, datanet, vocab_size, num_category, num_timesteps):
        self._inputs = []
        self._outputs = []
        self._indicator = 0
        self._num_timesteps = num_timesteps
        self.num_classes = num_category
        self.vocab_size = vocab_size
        self._parse_data(datanet)

    def _parse_data(self, datanet):
        self.CUNet = datanet
        self.walks = []
        for uid in self.CUNet:
            if len(self.CUNet[uid]) > 0:
                for i in range(1, 31):
                    self.walks.append(choice(self.CUNet[uid]))

        for line in self.walks:
            id_item = line[-1]
            id_path = line[0:-1]
            padding_number = self._num_timesteps - len(id_path)
            # 用vocab_size填充
            id_path = id_path + [self.vocab_size for i in range(padding_number)]
            self._inputs.append(id_path)
            self._outputs.append(id_item)
        # 转换操作，将列表转换成numpy形式
        self._inputs = np.asarray(self._inputs, dtype = np.int32)
        self._outputs = np.array(self._outputs, dtype = np.int32)
        # 进行随机化
        # self._random_shuffle()
        self._num_examples = len(self._inputs)

    def label_one_hot(self, label_id):
        y = [0] * self.num_classes
        y[label_id] = 1.0
        
        return np.array(y)

    # 随机化函数
    def _random_shuffle(self):
        p = np.random.permutation(len(list(self._inputs)))
        self._inputs = self._inputs[p]
        self._outputs = self._outputs[p]

    def next_batch(self, batch_size):
        end_indicator = self._indicator + batch_size
        if end_indicator > len(self._inputs):
            self._random_shuffle()
            self._indicator = 0
            end_indicator = batch_size
        # batch_size 大于样本总数，提示异常
        if end_indicator > len(self._inputs):
            raise Execption("batch_size: %d is too large" % batch_size)
        
        batchX = np.array(self._inputs[self._indicator: end_indicator])
        batchY = np.array(self._outputs[self._indicator: end_indicator])
        self._indicator = end_indicator
        return batchX, batchY

    def num_examples(self):
        return self._num_examples

class ABLAH(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(ABLAH, self).__init__(conf,trainingSet,testSet,fold)

    def readConfiguration(self):
        super(ABLAH, self).readConfiguration()
        options = config.LineConfig(self.config['ABLAH'])
        self.batch_size = int(options['-batch_size'])
        # the length of sequence: the number of nodes in every path
        self.cutoff = int(options['-cutoff'])

    def initModel(self):
        self.P = np.random.rand(self.data.getSize('user'), self.k).astype(np.float32)/10 # latent user matrix
        self.Q = np.random.rand(self.data.getSize(self.recType), self.k).astype(np.float32)/10 # latent item matrix

        # LSTM的步长，对齐minibatch
        self.num_timesteps = 3
        self.num_lstm_nodes = [self.k, self.k]
        self.num_lstm_layers = 2
        self.num_fc_nodes = self.k
        # LSTM梯度大小,防止梯度爆炸
        self.clip_lstm_grads = 1.0
        self.num_classes =  len(self.data.name2id['track'])
    
    def ListenData(self):
        userListen = defaultdict(dict)
        for item in self.data.trainingData:
            uid = self.data.getId(item['user'], 'user')
            tid = self.data.getId(item['track'], 'track')
            if tid not in userListen[uid]:
                userListen[uid][tid] = 1
            else:
                userListen[uid][tid] += 1
        return userListen

    def _attention(self, H):
        """
        利用Attention机制得到每个路径序列的向量表示
        """
        # 获得最后一层LSTM的神经元数量
        hiddenSize = 32
        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))
        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.tanh(H)
        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))
        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, self.cutoff])
        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)
        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, self.cutoff, 1]))
        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.squeeze(r)
        sentenceRepren = tf.tanh(sequeezeR)
        # 对Attention的输出可以做dropout处理
        output = tf.nn.dropout(sentenceRepren, self.keep_prob)
        return output

    def _Bi_LSTMAttention(self, embeddedWords):
        """
        Bi-LSTM + Attention 的模型结构
        """
        with tf.name_scope("Bi-LSTM"):

            lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(num_units=32, state_is_tuple=True), 
                output_keep_prob=self.keep_prob)

            lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(num_units=32, state_is_tuple=True),
                output_keep_prob=self.keep_prob)

            # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
            # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
            # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
            outputs, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell, 
                                                                          self.embeddedWords, dtype=tf.float32,
                                                                          scope="bi-lstm")
            # 在Bi-LSTM+Attention的论文中，将前向和后向的输出相加
        with tf.name_scope("Attention"):
            H = outputs[0] + outputs[1]

            # 得到Attention的输出
            output = self._attention(H)
            outputSize = 32

         # 全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[outputSize, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            
            outputB= tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="outputB")
            predictions = tf.nn.xw_plus_b(output, outputW, outputB, name="predictions")

        return predictions

    def _normalize(self, wordEmbedding, weights):
        """
        对word embedding 结合权重做标准化处理
        """
        mean = tf.matmul(weights, wordEmbedding)
        print(mean)
        powWordEmbedding = tf.pow(wordEmbedding - mean, 2.)
        
        var = tf.matmul(weights, powWordEmbedding)
        print(var)
        stddev = tf.sqrt(1e-6 + var)
        
        return (wordEmbedding - mean) / stddev

    def _addPerturbation(self, embedded, loss):
        """
        添加波动到word embedding
        """
        grad, = tf.gradients(
            loss,
            embedded,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        grad = tf.stop_gradient(grad)
        perturb = self._scaleL2(grad, 5)
        return embedded + perturb

    def _scaleL2(self, x, norm_length):
        # shape(x) = (batch, num_timesteps, d)
        # Divide x by max(abs(x)) for a numerically stable L2 norm.
        # 2norm(x) = a * 2norm(x/a)
        # Scale over the full sequence, dims (1, 2)
        alpha = tf.reduce_max(tf.abs(x), (1, 2), keepdims=True) + 1e-12
        l2_norm = alpha * tf.sqrt(
            tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keepdims=True) + 1e-6)
        x_unit = x / l2_norm
        return norm_length * x_unit

    def buildModel(self):
        print ('Kind Note: This method will probably take much time.')
        print('Building the HIN ...')
        G = nx.Graph()

        self.userListen = self.ListenData()
                
        # 加入所有数据，包括：用户、歌曲、艺人、专辑等，作为网络中的节点
        if 'user' in self.data.name2id:
            user_list = list(self.data.name2id['user'].values())
            G.add_nodes_from(user_list)
        if 'track' in self.data.name2id:
            item_list = list(self.data.name2id['track'].values())
            result_item = [x + len(self.data.name2id['user']) for x in item_list]
            G.add_nodes_from(result_item)
        if 'artist' in self.data.name2id:
            artist_list = list(self.data.name2id['artist'].values())
            result_artist = [(x + result_item[-1]+1) for x in artist_list]
            G.add_nodes_from(result_artist)
            vocab_size = result_artist[-1]
            # print('result_item:', result_item[-1])
            # print('result_artist:', result_artist)
        if 'album' in self.data.name2id:
            album_list = list(self.data.name2id['album'].values())
            if 'artist' in self.data.name2id:
                result_album = [(x + result_artist[-1]+1) for x in album_list]
            elif ('artist' not in self.data.name2id) and ('track' in self.data.name2id):
                result_album = [(x + result_item[-1]+1) for x in album_list]
            G.add_nodes_from(result_album)
            vocab_size = result_album[-1]
       
        for user in self.data.userRecord:
            uid = self.data.getId(user, 'user')
            for item in self.data.userRecord[user]:
                tid = self.data.getId(item['track'], 'track')
                G.add_edge(uid, tid, weight=1.0)
                if 'artist' in self.data.name2id:
                    aid = self.data.getId(item['artist'], 'artist')
                    G.add_edge(tid, aid)
                if 'album' in self.data.name2id:
                    album_id = self.data.getId(item['album'], 'album')
                    G.add_edge(tid, album_id)
                if ('artist' in self.data.name2id) and ('album' in self.data.name2id):
                    G.add_edge(aid, album_id)   

        print('Generating the all simple paths...')
        
        CUNet = defaultdict(list)
        for uid in self.userListen:
            for tid in self.userListen[uid]:  
                CUNet[uid] = list(nx.all_simple_paths(G, source=uid, target=tid, cutoff=self.cutoff))               
                # self.walks = self.walks+list(nx.all_simple_paths(G, source=uid, target=tid, cutoff=self.cutoff))
       
        test_CUNet = defaultdict(list)
        for user in self.data.testSet:
            uid = self.data.getId(user, 'user')
            for item in self.data.testSet[user]:  
                tid = self.data.getId(item, 'track')
                test_CUNet[uid] = list(nx.all_simple_paths(G, source=uid, target=tid, cutoff=self.cutoff))               

        train_dataset = Dataset(CUNet, vocab_size, self.num_classes, self.cutoff)
        test_dataset = Dataset(test_CUNet, vocab_size, self.num_classes, self.cutoff)
        print('_num_examples:', train_dataset.num_examples())
        print("train_dataset shape: {}".format(train_dataset._inputs.shape))
        print("train label shape: {}".format(train_dataset._outputs.shape))

        #######################################################################################
        with tf.Graph().as_default():
            self.inputs = tf.placeholder(tf.int32, shape=[None, self.cutoff], name='inputs')
            self.outputs = tf.placeholder(tf.int32, shape=[None], name='outputs')
            
            self.keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

            # 保存训练的步数
            self.global_step = tf.Variable(tf.zeros([], tf.int64), name = 'global_step', trainable=False)
            
            # with tf.name_scope("embedding"):

            # 初始化embedding
            embedding_initializer = tf.random_uniform_initializer(-1.0, 1.0)

            with tf.variable_scope(
                'embedding', initializer = embedding_initializer):
                embeddings = tf.get_variable(
                    'embedding',
                    [vocab_size+1, self.k],
                    tf.float32)
                # [1, 10, 7] -> [embeddings[1], embeddings[10], embeddings[7]]
                self.embeddedWords = tf.nn.embedding_lookup(embeddings, self.inputs)
                
                # # 利用词频计算新的词嵌入矩阵
                # normWordEmbedding = self._normalize(tf.cast(wordEmbedding, dtype=tf.float32, name='word2vec'), weights)

                # # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
                # self.embeddedWords = tf.nn.embedding_lookup(normWordEmbedding, self.inputs)


            with tf.name_scope("loss"):
                with tf.variable_scope("Bi-LSTM", reuse=None):
                    self.logits = self._Bi_LSTMAttention(self.embeddedWords)

                    self.predictions = tf.argmax(self.logits, axis=1, name="predictions")
                    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.outputs)

                    loss = tf.reduce_mean(losses)
            print('loss:', loss)

            with tf.name_scope("perturLoss"):
                with tf.variable_scope("Bi-LSTM", reuse=True):
                    perturWordEmbedding = self._addPerturbation(self.embeddedWords, loss)
                    perturPredictions = self._Bi_LSTMAttention(perturWordEmbedding)
                    # perturPredictions = tf.argmax(perturPredictions, axis=1, name='perturPredictions')
                    perturLosses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=perturPredictions, labels=self.outputs)

                    perturLoss = tf.reduce_mean(perturLosses)
            print('perturLoss:', perturLoss)
            self.loss = loss + perturLoss

            # 定义优化函数， 传入学习率参数
            self.optimizer = tf.train.AdamOptimizer(self.lRate)
            # 计算梯度，得到梯度和变量
            self.gradsAndVars = self.optimizer.compute_gradients(self.loss)
            # 将梯度应用到变量下，生成训练器
            self.trainOp = self.optimizer.apply_gradients(self.gradsAndVars, global_step=self.global_step)

            # gradSummaries = []
            # for g, v in self.gradsAndVars:
            #     if g is not None:
            #         tf.summary.histogram("{}/grad/hist".format(v.name), g)
            #         tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))

            # outDir = os.path.abspath(os.path.join(os.path.curdir, "summarys"))
            # print("Writing to {}\n".format(outDir))

            # lossSummary = tf.summary.scalar("loss", self.loss)
            # summaryOp = tf.summary.merge_all()

            init_op = tf.global_variables_initializer()
            self.train_keep_prob_value = 0.8
            with tf.Session() as sess:
                sess.run(init_op)
                for i in range(self.maxIter):
                    batch_inputs, batch_labels = train_dataset.next_batch(self.batch_size)
                    feed_dict = {
                        self.inputs: batch_inputs,
                        self.outputs: batch_labels,
                        self.keep_prob: self.train_keep_prob_value
                    }
                    outputs_val = sess.run([self.trainOp, summaryOp, self.global_step, self.loss, self.predictions],
                                           feed_dict)
                    _, summary, step, loss, predictions = outputs_val
                    self.P = sess.run(self.U)
                    self.Q = sess.run(self.V)
                    if step % 200 == 0:
                        tf.logging.info("Step: %5d, loss: %3.3f" % (step, loss))

    def predict(self, u):
        'invoked to rank all the items for the user'
        uid = self.data.getId(u,'user')
        return self.Q.dot(self.P[u])
