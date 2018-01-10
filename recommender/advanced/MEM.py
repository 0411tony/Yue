#coding:utf8
from base.IterativeRecommender import IterativeRecommender
from random import choice
from tool.qmath import cosine
import numpy as np
from math import log,exp
from sklearn.metrics.pairwise import pairwise_distances
from tool.config import LineConfig
import gensim.models.word2vec as w2v
import scipy.spatial.distance as distance

class MEM(IterativeRecommender):
    # BPR：Bayesian Personalized Ranking from Implicit Feedback
    # Steffen Rendle,Christoph Freudenthaler,Zeno Gantner and Lars Schmidt-Thieme

    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(MEM, self).__init__(conf,trainingSet,testSet,fold)

    def initModel(self):
        super(MEM, self).initModel()

    def readConfiguration(self):
        super(MEM, self).readConfiguration()
        MEMConfig = LineConfig(self.config['MEM'])
        self.epoch = int(MEMConfig['-epoch'])
        self.winSize = int(MEMConfig['-winSize'])
        self.negCount = int(MEMConfig['-negCount'])
        self.beta = float(MEMConfig['-beta'])

    def buildModel(self):
        #build a list for weighted negative sampling
        negCandidate = []
        # for track in self.data.trackListened:
        #     count = sum(self.data.trackListened[track].values())
        #     id = self.data.getId(track,'track')
        #     negCandidate+=[id]*count
        # print 'learning music embedding...'
        # iteration = 0
        # while iteration < self.epoch:
        #     loss = 0
        #     for user in self.data.userRecord:
        #         u = self.data.getId(user, 'user')
        #         #global user preference
        #         global_uv = np.zeros(self.k)
        #         for event in self.data.userRecord[user]:
        #             id = self.data.getId(event['track'], 'track')
        #             global_uv += self.Q[id]
        #         global_uv /= len(self.data.userRecord[user])
        #
        #         #song embedding
        #         for i in range(len(self.data.userRecord[user])):
        #             start = max(0,i-self.winSize/2)
        #             end = min(i+self.winSize/2,len(self.data.userRecord[user])-1)
        #             local = self.data.userRecord[user][start:i]+self.data.userRecord[user][i+1:end+1]
        #             local_v = np.zeros(self.k)
        #             for event in local:
        #                 id = self.data.getId(event['track'],'track')
        #                 local_v+=self.Q[id]
        #             v_hat = (global_uv+local_v)/(end-start+1)
        #             center_id = self.data.getId(self.data.userRecord[user][i]['track'],'track')
        #             center_v = self.Q[center_id]
        #             gradient = self.lRate*(1-sigmoid(v_hat.dot(center_v)))*v_hat
        #             gradient2 = self.lRate*(1-sigmoid(v_hat.dot(center_v)))*center_v
        #             self.Q[center_id]+=gradient
        #             global_uv+=gradient/len(self.data.userRecord[user])
        #             global_uv+=gradient2/len(self.data.userRecord[user])*(end-start)
        #             for event in local:
        #                 id = self.data.getId(event['track'],'track')
        #                 self.Q[id]+=gradient2/(end-start+1)
        #             loss+= -log(sigmoid(v_hat.dot(center_v)))
        #             #negative sampling
        #             for j in range(self.negCount):
        #                 neg_id = choice(negCandidate)
        #                 while neg_id==center_id:
        #                     neg_id = choice(negCandidate)
        #                 neg_v = self.Q[neg_id]
        #                 gradient = -self.lRate * (1 - sigmoid(v_hat.dot(neg_v))) * v_hat
        #                 gradient2 = -self.lRate * (1 - sigmoid(v_hat.dot(neg_v))) * neg_v
        #                 self.Q[center_id]+=gradient
        #                 for event in local:
        #                     id = self.data.getId(event['track'], 'track')
        #                     self.Q[id] += gradient2 / (end - start + 1)
        #                 loss+=-(log(1-sigmoid(neg_v.dot(v_hat))))

        sentences = []
        for user in self.data.userRecord:
            playList = []
            for item in self.data.userRecord[user]:
                playList.append(item['track'])
            sentences.append(playList)
        model = w2v.Word2Vec(sentences,size=self.k,window=5,min_count=0,iter=10,sg=1)
        for track in self.data.trackListened:
            tid = self.data.getId(track,'track')
            self.Q[tid]=model.wv[track]

        # #regularization
        # for album in self.data.album2Track:
        #     for track1 in self.data.album2Track[album]:
        #         for track2 in self.data.album2Track[album]:
        #             t1 = self.data.getId(track1,'track')
        #             t2 = self.data.getId(track2,'track')
        #             v1 = self.Q[t1]
        #             v2 = self.Q[t2]
        #             self.Q[t1]+=self.lRate*(exp(v1.dot(v2))*v2)
        #             self.Q[t2] += self.lRate*(exp(v1.dot(v2))*v1)

            #
            #         #print 'window %d finished' %(i)
            #     #print 'user %s finished.' %(user)
            # iteration+=1
            # print 'iteration %d, loss %.4f' %(iteration,loss)


        #preference embedding
        self.R = np.zeros((self.data.getSize('user'), self.k))
        for user in self.data.userRecord['user']:
            uid = self.data.getId(user,'user')
            global_uv = np.zeros(self.k)
            local_uv = np.zeros(self.k)
            for event in self.data.userRecord[user]:
                tid = self.data.getId(event['track'],'track')
                global_uv +=self.Q[tid]
            self.P[uid] = global_uv/len(self.data.userRecord['user'])
            recent = max(0,len(self.data.userRecord[user])-20)
            for event in self.data.userRecord[user][recent:]:
                tid = self.data.getId(event['track'],'track')
                local_uv +=self.Q[tid]
            self.R[uid] = local_uv/recent

        for t1 in self.data.trackListened:
            if len(self.data.trackListened[t1])<200:
                continue
            xiangsi=''
            m = 0
            s = ''
            n =''
            mi = 10000
            s1 = set(self.data.trackListened[t1].keys())
            for t2 in self.data.trackListened:
                if t1!=t2:
                    s2 = set(self.data.trackListened[t2].keys())
                    l = len(s1.intersection(s2))
                    if l>m and l>50:
                        m = l
                        s = t2
                    if l<mi:
                        mi = l
                        n = t2

            print t1,s,cosine(self.Q[self.data.getId(t1,'track')],self.Q[self.data.getId(s,'track')]),m
            print t1, n, cosine(self.Q[self.data.getId(t1, 'track')], self.Q[self.data.getId(n, 'track')]),mi
            break



    def predict(self, u):
        'invoked to rank all the items for the user'
        u = self.data.getId(u,'user')
        #using Euclidean Distance instead
        return pairwise_distances(self.Q,[self.R[u]])



