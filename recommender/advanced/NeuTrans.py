#coding:utf8
from base.DeepRecommender import DeepRecommender
import os
import numpy as np
from tool import config
from random import choice
from random import shuffle
from collections import defaultdict
from scipy.sparse import *
from scipy import *
import gensim.models.word2vec as w2v
from tool.qmath import cosine
import tensorflow as tf
import pickle

class NeuTrans(DeepRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(NeuTrans, self).__init__(conf,trainingSet,testSet,fold)


    def readConfiguration(self):
        super(NeuTrans, self).readConfiguration()
        options = config.LineConfig(self.config['NeuTrans'])
        self.alpha = float(options['-alpha'])
        self.topK = int(options['-k'])
        self.negCount = int(options['-neg'])

    def buildNetwork(self):
        self.trainingData = []
        print('Kind Note: This method will take much time')
        # build C-T-NET
        print('Building collaborative track network')
        self.trackNet = {}
        self.filteredListen = defaultdict(list)

        for track in self.data.trackRecord:
            if len(self.data.trackRecord[track]) > 0:
                self.trackNet[track] = self.data.trackRecord[track]
        for track in self.trackNet:
            tid = self.data.getId(track, 'track')
            for item in self.trackNet[track]:
                uid = self.data.getId(item['user'], 'user')
                if self.userListen[uid][tid] >= 0:
                    self.filteredListen[track].append(item['user'])
                    self.trainingData.append(item)
        
        self.CTNet = defaultdict(list)
        i=0
        for track1 in self.filteredListen:
            i += 1
            if i % 200 == 0:
                print (i, '/', len(self.filteredListen))
            s1 = set(self.filteredListen[track1])
            for track2 in self.filteredListen:
                if track1 != track2:
                    s2 = set(self.filteredListen[track2])
                    weight = len(s1.intersection(s2))
                    if weight > 0:
                        self.CTNet[track1] += [track2]*weight
        ########################    歌曲 C-T-N-E-T 构建结束    ############################
        
        print('Genrerating random deep walks...')
        self.T_walks = []
        self.T_visited = defaultdict(dict)
        for track in self.CTNet:
            for t in range(10):
                path = [track]
                lastNode = track
                for i in range(1, 10):
                    nextNode = choice(self.CTNet[lastNode])
                    count = 0
                    #while(nextNode in self.T_visited[lastNode] or nextNode not in self.aSim[lastNode]):
                    while(nextNode in self.T_visited[lastNode]):
                        nextNode = choice(self.CTNet[lastNode])
                        count+=1
                        if count==10:
                            break
                    path.append(nextNode)
                    self.T_visited[track][lastNode] = 1
                    lastNode = nextNode
                self.T_walks.append(path)
        shuffle(self.T_walks)
        ##del self.aSim

        print('Generating track embedding')
        model = w2v.Word2Vec(self.T_walks, size=self.k, window=5, min_count=0, iter=3)
        print('Track embedding generated')

        self.T = np.random.rand(self.data.getSize('track'), self.k)
        
        print ('Constructing similarity matrix...')
        i = 0
        self.nSim = {}
        for track1 in self.CTNet:
            tSim = []
            i += 1
            if i % 1000 == 0:
                print (i, '/', len(self.CTNet))
            vec1 = model.wv[track1]
            tid1 = self.data.getId(track1, 'track')
            for track2 in self.CTNet:
                if track1 != track2:
                    tid2 = self.data.getId(track2, 'track')
                    vec2 = model.wv[track2]
                    sim = max(1e-6, cosine(vec1, vec2))
                    tSim.append((tid2, sim))
                    #self.nSim[t1][t2] = sim
            self.nSim[tid1] = sorted(tSim, key=lambda d: d[1], reverse=True)[:20]
        
        file1 = 'nsim.txt'
        df1 = open(file1, 'wb')
        #df1 = open(file1, 'rb')
        pickle.dump(self.nSim, df1)
        #self.nSim = pickle.load(df1)

    def attributeSim(self):
        ########################    训练属性并计算属性相似度    ############################
        print ('train the attribute...')
        self.attr = {}
        for track in self.CTNet:
            val = []
            artist = self.data.Track2artist[track]
            #album = self.data.Track2album[tid1]
            val.append(artist)
            #val.append(album)
            self.attr[track] = val
        
        ## construct the S matrix
        print ('Constructing the attribute similarity matrix...')
        i = 0
        # self.aSim = defaultdict(dict)
        self.aSim = {}
        for track1 in self.CTNet:
            tSim = []
            i += 1
            if i % 1000 == 0:
                print (i, '/', len(self.CTNet))
            att1 = set(self.attr[track1])
            tid1 = self.data.getId(track1, 'track')
            for track2 in self.CTNet:
                if track1 != track2:
                    tid2 = self.data.getId(track2, 'track')
                    att2 = set(self.attr[track2])
                    num1 = len(list(att1&att2))
                    num2 = len(list(att1|att2))
                    sim = num1/num2
                    tSim.append((tid2, sim))
                    #self.aSim[t1][t2] = sim
            self.aSim[tid1] = sorted(tSim, key=lambda d: d[1], reverse=True)[:20]

        file2 = 'asim.txt'
        df1 = open(file2, 'wb')
        #df1 = open(file2, 'rb')
        pickle.dump(self.aSim, df1)
        #self.aSim = pickle.load(df1)

       
    def initModel(self):
        super(NeuTrans, self).initModel()
        self.userListen = defaultdict(dict)
        for user in self.data.userRecord:
            uid = self.data.getId(user, 'user')
            for item in self.data.userRecord[user]:
                tid = self.data.getId(item['track'], 'track')
                if item['track'] not in self.userListen[user]:
                    self.userListen[uid][tid] = 0
                self.userListen[uid][tid] += 1

        self.buildNetwork()
        self.attributeSim()
        
    def buildModel(self):
        ######################  构建神经网络非线性映射   ####################################
        print ('the tensorflow...')
        self.itemLayer = [64,64,64]
        self.userLayer = [64,64,64]
                
        #self.u_jdx = tf.placeholder(tf.int32, [None], name="u_jdx")
        self.v_jdx = tf.placeholder(tf.int32, [None], name="v_jdx")
        self.netSim = tf.placeholder(tf.float32, [None], name="netSim")
        self.attSim = tf.placeholder(tf.float32, [None], name="attSim")
        self.v_pdx = tf.placeholder(tf.int32, [None], name="v_pdx")
        self.v_qdx = tf.placeholder(tf.int32, [None], name="v_qdx")

        #userj_input = tf.nn.embedding_lookup(self.U, self.u_idx)
        self.V_neg_embed = tf.nn.embedding_lookup(self.V, self.v_jdx)

        self.V_net_pembed = tf.nn.embedding_lookup(self.V, self.v_pdx)
        self.V_net_qembed = tf.nn.embedding_lookup(self.V, self.v_qdx)

        def init_variable(shape, name):
            return tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.01), name=name)
        
        with tf.name_scope("User_Layer"):
            user_W1 = init_variable([self.k, self.userLayer[0]], "user_W1")
            self.U_embed_out = tf.matmul(self.U_embed, user_W1)
            self.regLoss = tf.nn.l2_loss(user_W1)

        with tf.name_scope("Item_Layer"):
            item_W1 = init_variable([self.k, self.itemLayer[0]], "item_W1")
            self.V_embed_out = tf.matmul(self.V_embed, item_W1)
            self.V_neg_embed_out = tf.matmul(self.V_neg_embed, item_W1)
            self.regLoss = tf.add(self.regLoss, tf.nn.l2_loss(item_W1))

        self.reg_lambda = tf.constant(self.regU, dtype=tf.float32)
      
        error = tf.subtract(tf.reduce_sum(tf.multiply(self.U_embed_out, self.V_embed_out), 1), tf.reduce_sum(tf.multiply(self.U_embed_out, self.V_neg_embed_out), 1))
        self.loss = tf.reduce_sum(tf.nn.softplus(-error))
        # 构造正则化项 完善损失函数
        self.regLoss = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.U_embed_out)),self.regLoss)
        self.regLoss = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V_embed_out)),self.regLoss)
        self.regLoss = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V_neg_embed_out)), self.regLoss)
        self.total_loss = tf.add(self.loss, self.regLoss)

        ##### 网络结构损失函数 ####
        error_net = tf.subtract(self.netSim, tf.reduce_sum(tf.multiply(self.V_net_pembed, self.V_net_qembed), 1))
        self.loss_net =  tf.reduce_sum(tf.nn.softplus(-error_net))
        self.reg_loss_net = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V_net_pembed)),
                               tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V_net_qembed)))
        self.total_loss_net = tf.add(self.loss_net, self.reg_loss_net)

        ##### 属性损失函数 ####
        error_att = tf.subtract(self.attSim, tf.reduce_sum(tf.multiply(self.V_net_pembed, self.V_net_qembed), 1))
        self.loss_att =  tf.reduce_sum(tf.nn.softplus(-error_att))
        self.reg_loss_att = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V_net_pembed)),
                               tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V_net_qembed)))
        self.total_loss_att = tf.add(self.loss_att, self.reg_loss_att)

        self.optimizer = tf.train.AdamOptimizer(self.lRate)
        self.train = self.optimizer.minimize(self.total_loss)

        self.optimizer_net = tf.train.AdamOptimizer(self.lRate)
        self.train_net = self.optimizer_net.minimize(self.total_loss_net)

        self.optimizer_att = tf.train.AdamOptimizer(self.lRate)
        self.train_att = self.optimizer_att.minimize(self.total_loss_att)

        self.U = np.zeros((self.m, self.itemLayer[0]))
        self.V = np.zeros((self.n, self.itemLayer[0]))

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for epoch in range(self.maxIter):
                item_pdx, item_qdx, nSim = self._net_next_batch()
                _,loss = sess.run([self.train_net,self.total_loss_net], feed_dict={self.v_pdx:item_pdx, self.v_qdx:item_qdx, self.netSim:nSim})
                print ('iteration:', epoch, 'loss:',loss)
                self.ranking_performance()

            for epoch in range(self.maxIter):
                item_pdx, item_qdx, aSim = self._att_next_batch()
                _,loss = sess.run([self.train_att,self.total_loss_att], feed_dict={self.v_pdx:item_pdx, self.v_qdx:item_qdx, self.attSim:aSim})
                print ('iteration:', epoch, 'loss:',loss)
                self.ranking_performance()

            for epoch in range(self.maxIter):
                user_idx, item_idx, neg_item_idx = self.next_batch()
                _,loss = sess.run([self.train,self.total_loss], feed_dict={self.u_idx:user_idx, self.v_idx:item_idx, self.v_jdx:neg_item_idx})
                print ('iteration:', epoch, 'loss:',loss)
                U_embedding, V_embedding = sess.run([self.U_embed_out,self.V_embed_out], feed_dict={self.u_idx:user_idx, self.v_idx:item_idx, self.v_jdx:neg_item_idx})
                for ue,u in zip(U_embedding,user_idx):
                   self.U[u]=ue
                for ve,v in zip(V_embedding,item_idx):
                   self.V[v]=ve

                self.ranking_performance()


    def next_batch(self):
        batch_idx = np.random.randint(len(self.data.trainingData), size=256)
        users = [self.data.trainingData[idx]['user'] for idx in batch_idx]
        items = [self.data.trainingData[idx]['track'] for idx in batch_idx]
        user_idx,item_idx=[],[]
        neg_item_idx = []
        for i,user in enumerate(users):
            uid = self.data.getId(user, 'user')
            tid = self.data.getId(items[i], 'track')  
            for j in range(100): #negative sampling
                neg_id = random.randint(0, self.n- 1)
                while neg_id in self.userListen[uid]:
                    neg_id = random.randint(0, self.n - 1)      
                user_idx.append(uid)
                item_idx.append(tid)
                neg_item_idx.append(neg_id)
        return user_idx, item_idx, neg_item_idx

    def _net_next_batch(self):
        batch_idx = np.random.randint(len(self.nSim))
        item_pdx, item_qdx=[],[]
        nSim=[]
        for i in range(batch_idx):
            t1 = choice(list(self.nSim))
            for t2 in self.nSim[t1]:
                item_pdx.append(t1)
                item_qdx.append(t2[0])
                nSim.append(t2[1])
        return item_pdx, item_qdx, nSim

    def _att_next_batch(self):
        batch_idx = np.random.randint(len(self.aSim))
        item_pdx, item_qdx=[],[]
        aSim=[]
        for i in range(batch_idx):
            t1 = choice(list(self.aSim))
            for t2 in self.aSim[t1]:
                item_pdx.append(t1)
                item_qdx.append(t2[0])
                aSim.append(t2[1])
        return item_pdx, item_qdx, aSim

    def predict(self, u):
        'invoked to rank all the items for the user'
        u = self.data.getId(u,'user')
        return self.V.dot(self.U[u])
        #pass