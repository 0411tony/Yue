#coding:utf8
from base.IterativeRecommender import IterativeRecommender
from random import choice
from tool.qmath import sigmoid
from math import log
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
        MEMConfig = LineConfig(self.config['MEM'])
        self.epoch = int(MEMConfig['-epoch'])
        self.winSize = int(MEMConfig['-windowSize'])

    def buildModel(self):

        print 'learning music embedding...'
        #build a list for weighted negative sampling
        negCandidate = []
        for track in self.data.trackListened:
            count = sum(self.data.trackListened[track].values())
            negCandidate+=[track]*count


        iteration = 0
        while iteration < self.epoch:
        for user in self.data.userRecord:
            u = self.data.getId(user, 'user')
            for item in self.data.userRecord[user]:



        print 'training...'
        iteration = 0

        while iteration < self.maxIter:
            self.loss = 0
            for user in self.data.userRecord:
                u = self.data.getId(user,'user')
                for item in self.data.userRecord[user]:
                    i = self.data.getId(item[self.recType],self.recType)
                    item_j = choice(itemList)
                    while (userListened[user].has_key(item_j)):
                        item_j = choice(itemList)
                    j = self.data.getId(item_j,self.recType)
                    s = sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))
                    self.P[u] += self.lRate * (1 - s) * (self.Q[i] - self.Q[j])
                    self.Q[i] += self.lRate * (1 - s) * self.P[u]
                    self.Q[j] -= self.lRate * (1 - s) * self.P[u]

                    self.P[u] -= self.lRate * self.regU * self.P[u]
                    self.Q[i] -= self.lRate * self.regI * self.Q[i]
                    self.Q[j] -= self.lRate * self.regI * self.Q[j]
                    self.loss += -log(s)
            self.loss += self.regU * (self.P * self.P).sum() + self.regI * (self.Q * self.Q).sum()
            iteration += 1
            if self.isConverged(iteration):
                break


    def predict(self, u):
        'invoked to rank all the items for the user'
        u = self.data.getId(u,'user')
        return self.Q.dot(self.P[u])



