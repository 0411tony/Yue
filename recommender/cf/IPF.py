#coding:utf8
from base.recommender import Recommender
from tool.config import LineConfig
from random import choice
from collections import defaultdict
class IPF(Recommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(IPF, self).__init__(conf,trainingSet,testSet,fold)

    def initModel(self):
        super(IPF, self).initModel()
        print 'initializing STG...'
        self.STG = {}
        self.STG['user'] = self.data.userRecord
        self.sessionNodes = {}
        for user in self.data.userRecord:
            t = max(0,len(self.data.userRecord[user])-10)
            self.sessionNodes[user]=self.data.userRecord[t:]
        item2session=defaultdict(list)
        for user in self.sessionNodes:
            for item in self.sessionNodes[user]:
                item2session[item].append(user)
        
        self.STG['session'] = self.sessionNodes
        item2user = {}
        if self.recType=='track':
            for item in self.data.trackListened:
                item2user[item]=self.data.trackListened[item].keys()

        elif self.recType== 'artist':
            for item in self.data.artistListened:
                item2user[item]=self.data.artistListened[item].keys()
        else:
            for item in self.data.albumListened:
                item2user[item]=self.data.albumListened[item].keys()
        self.STG['item2user'] = item2user
        self.STG['item2session'] = item2session
        self.path = [['user','item','user','item'],
                     ['user','item','session','item'],
                     ['session','item','user','item'],
                     ['session','item','session','item']]


    def readConfiguration(self):
        super(IPF, self).readConfiguration()
        self.rho = int(LineConfig(self.config['IPF'])['-rho'])
        if self.rho<0 or self.rho>1:
            self.rho=0.5
        self.beta = float(LineConfig(self.config['IPF'])['-beta'])


    def predict(self, user):
        'invoked to rank all the items for the user'
        #I think the pseudo code in the paper sucks, so I re-implement the algorithm based on my design
        rank = []
        for p in self.path:
            visited = {}
            queue = []
            queue.append((user, p[1]))

            v,vType = queue.pop()
            if visited.has_key([v]) and vType=='item':
                continue
            visited[vType+v]=1
            for nextNode in self.STG[vType][v]:
                if not visited.has_key(vType+nextNode):
                    distance[vType+nextNode]=distance[vType+v]+1
                    queue.append((vType,nextNode))
                if distance[vType]




