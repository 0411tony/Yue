#coding:utf8
from base.IterativeRecommender import IterativeRecommender
import math
import numpy as np
from tool import qmath
from random import choice
from tool.qmath import sigmoid
from math import log
from collections import defaultdict
class BPR(IterativeRecommender):

    # BPR：Bayesian Personalized Ranking from Implicit Feedback
    # Steffen Rendle,Christoph Freudenthaler,Zeno Gantner and Lars Schmidt-Thieme

    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(BPR, self).__init__(conf,trainingSet,testSet,fold)

    def initModel(self):
        super(BPR, self).initModel()


    def buildModel(self):
        userListened = defaultdict(dict)
        for user in self.dao.userRecord:
            for item in self.dao.userRecord[user]:
                userListened[user][item[self.recType]] = 1

        print 'training...'
        iteration = 0
        itemList = self.dao.name2id[self.recType].keys()
        while iteration < self.maxIter:
            self.loss = 0
            for user in self.dao.userRecord:
                u = self.dao.getId(user,'user')
                for item in self.dao.userRecord[user]:
                    i = self.dao.getId(item[self.recType],self.recType)
                    item_j = choice(itemList)
                    while (userListened[user].has_key(item_j)):
                        item_j = choice(itemList)
                    j = self.dao.getId(item_j,self.recType)
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
        u = self.dao.getId(u,'user')
        return self.Q.dot(self.P[u])



