#coding:utf8
from base.IterativeRecommender import IterativeRecommender
import math
import numpy as np
from tool import qmath
from random import choice
from tool.qmath import sigmoid
from math import log
from collections import defaultdict
from scipy.sparse import *
from scipy import *
import gensim.models.word2vec as w2v
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
        for user in self.data.userRecord:
            playList = []
            for item in self.data.userRecord[user]:
                playList.append(item['track'])
            sentences.append(playList)
        model = w2v.Word2Vec(sentences, size=self.k, window=5, min_count=0, iter=10)
        for track in self.data.listened['track']:
            tid = self.data.getId(track, 'track')
            self.T[tid] = model.wv[track]
        print 'song embedding generated.'

        print 'Constructing similarity matrix...'
        i = 0
        self.topKSim = 0
        for track1 in self.data.listened['track']:
            tSim = []
            i += 1
            if i % 200 == 0:
                print i, '/', len(self.data.listened['track'])
            vec1 = self.T[self.data.getId(track1, 'track')]
            for track2 in self.data.listened['track']:
                if track1 <> track2:
                    vec2 = self.T[self.data.getId(track2, 'track')]
                    sim = cosine(vec1, vec2)
                    tSim.append((track2, sim))

            self.topKSim[track1] = sorted(tSim, key=lambda d: d[1], reverse=True)[:self.topK]


        userListen = defaultdict(dict)
        for user in self.data.userRecord:
            for item in self.data.userRecord[user]:
                if not userListen[user].has_key(item[self.recType]):
                    userListen[user][item[self.recType]] = 0
                userListen[user][item[self.recType]] += 1
        print 'training...'
        iteration = 0
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

            for t1 in self.data.listened['track']:
                tid1 = self.data.getId(t1,'track')
                for t2 in self.topKSim:
                    tid2 = self.data.getId(t2[0],'track')
                    sim = t2[1]
                    error = (sim-self.Y[tid1].dot(self.Y[tid2]))
                    self.loss+=error**2
                    self.Y[tid1]+=0.5*self.alpha*self.lRate*(error)*self.Y[tid2]
                    self.Y[tid2]+=0.5*self.alpha*self.lRate*(error)*self.Y[tid1]


            #self.loss += (self.X * self.X).sum() + (self.Y * self.Y).sum()
            iteration += 1
            print 'iteration:',iteration,'loss:',self.loss
            # if self.isConverged(iteration):
            #     break


    def predict(self, u):
        'invoked to rank all the items for the user'
        u = self.data.getId(u,'user')
        return self.Y.dot(self.X[u])



