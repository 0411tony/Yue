#coding:utf8
from base.recommender import Recommender
from random import choice
import numpy as np
class Rand(Recommender):

    # Recommend items for every user at random

    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(Rand, self).__init__(conf,trainingSet,testSet,fold)



    def predict(self, u):
        'invoked to rank all the items for the user'
        N = int(self.ranking['-topN'])
        self.recommendation = []
        self.candidates = []
        if self.recType == 'track':
            candidates = self.dao.trackListened.keys()
        elif self.recType == 'artist':
            candidates = self.dao.artistListened.keys()
        else:
            candidates = self.dao.albumListened.keys()
        for i in range(N):
            item = choice(candidates)
            self.recommendation.append(item)
        return self.recommendation


