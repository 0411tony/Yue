#coding:utf8
from base.recommender import Recommender
import numpy as np
class MostPop(Recommender):

    # Recommend the most popular items for every user

    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(MostPop, self).__init__(conf,trainingSet,testSet,fold)

    # def readConfiguration(self):
    #     super(BPR, self).readConfiguration()

    def buildModel(self):

        self.recommendation = []
        if self.recType=='track':
            self.recommendation = sorted(self.data.trackListened.iteritems(),key=lambda d:len(d[1]), reverse=True)
        elif self.recType=='artist':
            self.recommendation = sorted(self.data.artistListened.iteritems(),key=lambda d:len(d[1]), reverse=True)
        else:
            self.recommendation = sorted(self.data.albumListened.iteritems(), key=lambda d: len(d[1]), reverse=True)
        self.recommendation = [item[0] for item in self.recommendation]



    def predict(self, u):
        'invoked to rank all the items for the user'
        return self.recommendation


