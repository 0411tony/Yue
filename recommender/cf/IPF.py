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
        userListened = defaultdict(list)
        self.sessionNodes = {}
        for user in self.data.userRecord:
            for item in self.data.userRecord[user]:
                userListened[user].append(item[self.recType])
        for user in userListened:
            t = max(0, len(userListened[user]) - 10)
            self.sessionNodes[user] = userListened[user][t:]
        self.STG = {}
        self.STG['user'] = userListened
        self.STG['session'] = self.sessionNodes

        item2session=defaultdict(list)
        for user in self.sessionNodes:
            for item in self.sessionNodes[user]:
                item2session[item].append(user)
        item2user = {}

        item2user[item]=self.data.listened[self.recType][item].keys()


        self.STG['item2user'] = item2user
        self.STG['item2session'] = item2session
        self.path = [['user','item','user','item'],
                     ['user','item','session','item'],
                     ['session','item','user','item'],
                     ['session','item','session','item']]


    def readConfiguration(self):
        super(IPF, self).readConfiguration()
        self.rho = float(LineConfig(self.config['IPF'])['-rho'])
        if self.rho<0 or self.rho>1:
            self.rho=0.5
        self.beta = float(LineConfig(self.config['IPF'])['-beta'])
        self.eta = float(LineConfig(self.config['IPF'])['-eta'])

    def probability(self,v1,v2):
        if v1[0]=='user' or v1[0]=='session' and v2[0]=='item':
            return 1.0/pow(len(self.STG[v1[0]][v1[1]]),self.rho)
        elif v1[0]=='item' and v2[0]=='user':
            return pow(self.eta/(self.eta*len(self.STG['item2user'][v1[1]])+
                                 len(self.STG['item2session'][v1[1]])),self.rho)
        elif v1[0]=='item' and v2[0]=='session':
            return pow(1/(self.eta*len(self.STG['item2user'][v1[1]])+
                                 len(self.STG['item2session'][v1[1]])),self.rho)

    def predict(self, user):
        #I think the pseudo code in the paper sucks, so I re-implement the algorithm based on my design
        rank = {}
        visited = {}
        for p in self.path:
            queue = []
            queue.append((p[0],user))
            distance = {}
            distance[p[0]+user]=0
            if p[0]=='user':
                rank[p[0]+user]=self.beta
            else:
                rank[p[0] + user] = 1-self.beta
            while len(queue)>0:
                vType,v = queue.pop()
                if visited.has_key(vType+v):
                    continue
                visited[vType+v]=1
                for nextNode in self.STG[p[distance[vType+v]]][v]:
                    nextType = p[distance[vType+v]+1]
                    if not visited.has_key(nextType+nextNode):
                        distance[nextType+nextNode]=distance[vType+v]+1
                        queue.append((nextType,nextNode))
                        visited[nextType+nextNode]=1
                    else:
                        continue
                    if distance[vType+v]< distance[p[distance[vType+v]+1]+nextNode]:
                        if not rank.has_key(nextType+nextNode):
                            rank[nextType+nextNode]=0
                        rank[nextType+nextNode]+=rank[vType+v]*self.probability((vType,v),(nextType,nextNode))
        recommendedList = [(key[4:],value) for key,value in rank.iteritems() if key[0:4]=='item']
        recommendedList = sorted(recommendedList,key=lambda d:d[1],reverse=True)
        recommendedList = [item[0] for item in recommendedList]

        return recommendedList




