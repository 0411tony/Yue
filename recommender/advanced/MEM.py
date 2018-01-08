#coding:utf8
from base.IterativeRecommender import IterativeRecommender
from random import choice
from tool.qmath import sigmoid
import numpy as np
from math import log,exp
from collections import defaultdict
from tool.config import LineConfig

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
        for track in self.data.trackListened:
            count = sum(self.data.trackListened[track].values())
            id = self.data.getId(track,'track')
            negCandidate+=[id]*count
        print 'learning music embedding...'
        iteration = 0
        while iteration < self.epoch:
            loss = 0
            for user in self.data.userRecord:
                u = self.data.getId(user, 'user')
                #global user preference
                global_uv = np.zeros(self.k)
                for event in self.data.userRecord[user]:
                    id = self.data.getId(event['track'], 'track')
                    global_uv += self.Q[id]
                    global_uv /= len(self.data.userRecord[user])

                #song embedding
                for i in range(len(self.data.userRecord[user])):
                    start = max(0,i-self.winSize/2)
                    end = min(i+self.winSize/2,len(self.data.userRecord[user])-1)
                    local = self.data.userRecord[user][start:i]+self.data.userRecord[user][i+1:end+1]
                    local_v = np.zeros(self.k)
                    for event in local:
                        id = self.data.getId(event['track'],'track')
                        local_v+=self.Q[id]
                    v_hat = (global_uv+local_v)/(end-start+1)
                    center_id = self.data.getId(self.data.userRecord[user][i]['track'],'track')
                    center_v = self.Q[center_id]
                    gradient = self.lRate*(1-sigmoid(v_hat.dot(center_v)))*v_hat
                    gradient2 = self.lRate*(1-sigmoid(v_hat.dot(center_v)))*center_v
                    self.Q[center_id]+=gradient
                    global_uv+=gradient/len(self.data.userRecord[user])
                    global_uv+=gradient2/len(self.data.userRecord[user])
                    for event in local:
                        id = self.data.getId(event['track'],'track')
                        self.Q[id]+=gradient2/(end-start+1)
                    loss+= -log(sigmoid(v_hat.dot(center_v)))
                    #negative sampling
                    for j in range(self.negCount):
                        neg_id = choice(negCandidate)
                        while neg_id==center_id:
                            neg_id = choice(negCandidate)
                        neg_v = self.Q[neg_id]
                        gradient = -self.lRate * (1 - sigmoid(v_hat.dot(neg_v))) * v_hat
                        gradient2 = -self.lRate * (1 - sigmoid(v_hat.dot(neg_v))) * neg_v
                        self.Q[center_id]+=gradient
                        for event in local:
                            id = self.data.getId(event['track'], 'track')
                            self.Q[id] += gradient2 / (end - start + 1)
                        loss+=-(log(1-sigmoid(neg_v.dot(v_hat))))

            #regularization
            for album in self.data.album2Track:
                for track1 in self.data.album2Track[album]:
                    for track2 in self.data.album2Track[album]:
                        t1 = self.data.getId(track1,'track')
                        t2 = self.data.getId(track2,'track')
                        v1 = self.Q[t1]
                        v2 = self.Q[t2]
                        self.Q[t1]+=self.lRate*(exp(v1.dot(v2))*v2)
                        self.Q[t2] += self.lRate*(exp(v1.dot(v2))*v1)


                    #print 'window %d finished' %(i)
                #print 'user %s finished.' %(user)
            print 'iteration %d, loss %.4f' %(iteration,loss)

        #recent playlist embedding
        self.R = np.zeros((self.data.getSize('user'), self.k))
        for user in self.data.userRecord['user']:
            uid = self.data.getId(user,'user')
            global_uv = np.zeros(self.k)
            local_uv = np.zeros(self.k)
            for event in self.data.userRecord[user]:
                tid = self.data.getId(event['track'],'track')
                global_uv +=self.Q[tid]
            self.Q[uid] = global_uv/len(self.data.userRecord['user'])
            recent = max(0,len(self.data.userRecord[user])-10)
            for event in self.data.userRecord[user][recent:]:
                tid = self.data.getId(event['track'],'track')
                local_uv +=self.Q[tid]
            self.R[uid] = local_uv/recent



    def predict(self, u):
        'invoked to rank all the items for the user'
        u = self.data.getId(u,'user')
        #using Euclidean Distance instead
        return 1/(((self.Q-self.P[u])*(self.Q-self.P[u])).sum(axis=0)+((self.R-self.P[u])*(self.R-self.P[u])).sum(axis=0))



