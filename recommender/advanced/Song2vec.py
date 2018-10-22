#coding:utf8
from base.IterativeRecommender import IterativeRecommender
import math
import numpy as np
from tool import qmath
from tool import config
from random import choice
from tool.qmath import sigmoid
from math import log
from collections import defaultdict
from scipy.sparse import *
from scipy import *
import gensim.models.word2vec as w2v
from tool.qmath import cosine
class Song2vec(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(Song2vec, self).__init__(conf,trainingSet,testSet,fold)

    def initModel(self):
        super(Song2vec, self).initModel()
        self.X=self.P*10
        self.Y=self.Q*10
        self.m = self.data.getSize('user')
        self.n = self.data.getSize(self.recType)

    def readConfiguration(self):
        super(Song2vec, self).readConfiguration()
        options = config.LineConfig(self.config['Song2vec'])
        self.alpha = float(options['-alpha'])
        self.topK = int(options['-k'])

    def buildModel(self):
        self.T = np.random.rand(self.data.getSize('track'),self.k)
        sentences = []
        self.listenTrack = set()
        self.user = defaultdict(list)
        for user in self.data.userRecord:
            playList = []
            if len(self.data.userRecord[user]) > 10:
                self.user[user] = self.data.userRecord[user]
                for item in self.data.userRecord[user]:
                    playList.append(item['track'])
                    self.listenTrack.add(item['track'])
                sentences.append(playList)
        # print('the sentences:', self.data.userRecord)
        model = w2v.Word2Vec(sentences, size=self.k, window=5, min_count=0, iter=10)
        for track in self.listenTrack:
            tid = self.data.getId(track, 'track')
            self.T[tid] = model.wv[track]
        print ('song embedding generated.')

        print ('Constructing similarity matrix...')
        i = 0
        self.topKSim = {}
        for track1 in self.listenTrack:
            tSim = []
            i += 1
            if i % 200 == 0:
                print (i, '/', len(self.listenTrack))
            vec1 = self.T[self.data.getId(track1, 'track')]
            for track2 in self.listenTrack:
                if track1 != track2:
                    vec2 = self.T[self.data.getId(track2, 'track')]
                    sim = cosine(vec1, vec2)
                    tSim.append((track2, sim))

            self.topKSim[track1] = sorted(tSim, key=lambda d: d[1], reverse=True)[:self.topK]

        userListen = defaultdict(dict)
        for user in self.user:
            for item in self.user[user]:
                if item[self.recType] not in userListen[user]:
                    userListen[user][item[self.recType]] = 0
                userListen[user][item[self.recType]] += 1
        print ('training...')
        
        
        iteration = 0
        itemList = list(self.data.name2id[self.recType].keys())
        while iteration < self.maxIter:
            self.loss = 0
            
            YtY = self.Y.T.dot(self.Y)
            I = np.ones(self.n)
            for user in self.data.name2id['user']:
                #C_u = np.ones(self.data.getSize(self.recType))
                H = np.ones(self.n)
                val = []
                pos = []
                P_u = np.zeros(self.n)
                uid = self.data.getId(user,'user')
                for item in userListen[user]:
                    iid = self.data.getId(item,self.recType)
                    r_ui = userListen[user][item]
                    pos.append(iid)
                    val.append(10*r_ui)
                    H[iid]+=10*r_ui
                    P_u[iid]=1
                    error = (P_u[iid]-self.X[uid].dot(self.Y[iid]))
                    self.loss+=pow(error,2)
                #sparse matrix
                C_u = coo_matrix((val,(pos,pos)),shape=(self.n,self.n))
                A = (YtY+np.dot(self.Y.T,C_u.dot(self.Y))+self.regU*np.eye(self.k))
                self.X[uid] = np.dot(np.linalg.inv(A),(self.Y.T*H).dot(P_u))


            XtX = self.X.T.dot(self.X)
            I = np.ones(self.m)
            for item in self.data.name2id[self.recType]:
                P_i = np.zeros(self.m)
                iid = self.data.getId(item, self.recType)
                H = np.ones(self.m)
                val = []
                pos = []
                for user in self.data.listened[self.recType][item]:
                    uid = self.data.getId(user, 'user')
                    r_ui = self.data.listened[self.recType][item][user]
                    pos.append(uid)
                    val.append(10*r_ui)
                    H[uid] += 10*r_ui
                    P_i[uid] = 1
                # sparse matrix
                C_i = coo_matrix((val, (pos, pos)),shape=(self.m,self.m))
                A = (XtX+np.dot(self.X.T,C_i.dot(self.X))+self.regU*np.eye(self.k))
                self.Y[iid]=np.dot(np.linalg.inv(A), (self.X.T*H).dot(P_i))
            
            for user in self.user:
                u = self.data.getId(user,'user')
                for item in self.user[user]:
                    i = self.data.getId(item[self.recType],self.recType)
                    for ind in range(3):
                        item_j = choice(itemList)
                        while (item_j in userListen[user]):
                            item_j = choice(itemList)
                        j = self.data.getId(item_j,self.recType)
                        s = sigmoid(self.X[u].dot(self.Y[i]) - self.X[u].dot(self.Y[j]))
                        self.X[u] += self.lRate * (1 - s) * (self.Y[i] - self.Y[j])
                        self.Y[i] += self.lRate * (1 - s) * self.X[u]
                        self.Y[j] -= self.lRate * (1 - s) * self.X[u]

                        self.X[u] -= self.lRate * self.regU * self.X[u]
                        self.Y[i] -= self.lRate * self.regI * self.Y[i]
                        self.Y[j] -= self.lRate * self.regI * self.Y[j]
                        self.loss += -log(s)

            for t1 in self.topKSim:
                tid1 = self.data.getId(t1,'track')
                for t2 in self.topKSim[t1]:
                    tid2 = self.data.getId(t2[0],'track')
                    sim = t2[1]
                    error = (sim-self.Y[tid1].dot(self.Y[tid2]))
                    self.loss+=error**2
                    self.Y[tid1]+=0.5*self.alpha*self.lRate*(error)*self.Y[tid2]
                    self.Y[tid2]+=0.5*self.alpha*self.lRate*(error)*self.Y[tid1]
         
            self.loss += (self.X * self.X).sum() + (self.Y * self.Y).sum()
            iteration += 1
            print ('iteration:',iteration,'loss:',self.loss)
            # if self.isConverged(iteration):
            #     break


    def predict(self, u):
        'invoked to rank all the items for the user'
        u = self.data.getId(u,'user')
        return self.Y.dot(self.X[u])
