#coding:utf8
from base.recommender import Recommender
from random import shuffle
import numpy as np
class Rand(Recommender):

    # Recommend items for every user at random

    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(Rand, self).__init__(conf,trainingSet,testSet,fold)



    def predict(self, u):
        'invoked to rank all the items for the user'
        self.candidates = []
        candidates = self.data.listened[self.recType].keys()
        shuffle(candidates)
        return self.candidates


