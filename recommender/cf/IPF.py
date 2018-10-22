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
        print ('initializing STG...')
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
        for item in self.data.listened[self.recType]:
            item2user[item]=self.data.listened[self.recType][item].keys()


        self.STG['item2user'] = item2user
        self.STG['item2session'] = item2session
        self.path = [['user','item2user','user','item'],
                     ['user','item2session','session','item'],
                     ['session','item2user','user','item'],
                     ['session','item2session','session','item']]


    def readConfiguration(self):
        super(IPF, self).readConfiguration()
        self.rho = float(LineConfig(self.config['IPF'])['-rho'])
        if self.rho<0 or self.rho>1:
            self.rho=0.5
        self.beta = float(LineConfig(self.config['IPF'])['-beta'])
        self.eta = float(LineConfig(self.config['IPF'])['-eta'])

    def probability(self,v1,v2):
        if (v1[0]=='user' and v2[0]=='item2user') or (v1[0]=='session' and v2[0]=='item2session') or \
                (v1[0] == 'user' and v2[0] == 'item2session') or (v1[0]=='session'and v2[0] =='item2user') or \
            (v1[0] == 'user' and v2[0] == 'item') or  (v1[0] == 'session' and v2[0] == 'item'):
            return 1.0/pow(len(self.STG[v1[0]][v1[1]]),self.rho)
        elif v1[0]=='item2user' and v2[0]=='user':
            return pow(self.eta/(self.eta*len(self.STG['item2user'][v1[1]])+
                                 len(self.STG['item2session'][v1[1]])),self.rho)
        elif v1[0]=='item2session' and v2[0]=='session':
            return pow(1/(self.eta*len(self.STG['item2user'][v1[1]])+
                                 len(self.STG['item2session'][v1[1]])),self.rho)


    def predict(self, user):
        rank = {}
        for p in self.path:
            visited = {}
            queue = []
            queue.append((p[0],user))
            distance = {}
            distance[p[0]+user]=0
            if p[0]=='user':
                rank[p[0]+'_'+user]=self.beta
            else:
                rank[p[0]+'_'+user] = 1-self.beta
            while len(queue)>0:
                vType,v = queue.pop()
                if (vType+v) in visited and visited[vType+v]==1:
                    continue
                visited[vType+v]=1
                if vType=='item':
                    continue
                for nextNode in self.STG[p[distance[vType+v]]][v]:
                    nextType = p[distance[vType+v]+1]
                    if (nextType+nextNode) not in visited:
                        distance[nextType+nextNode]=distance[vType+v]+1
                        queue.append((nextType,nextNode))
                        visited[nextType+nextNode]=0
                    else:
                        continue
                    if distance[vType+v]< distance[p[distance[vType+v]+1]+nextNode]:
                        if (nextType+'_'+nextNode) not in rank:
                            rank[nextType+'_'+nextNode]=0
                        rank[nextType+'_'+nextNode]+=rank[vType+'_'+v]*self.probability((vType,v),(nextType,nextNode))
        recommendedList = [(key[5:],value) for key,value in rank.items() if key[0:5]=='item_']
        recommendedList = sorted(recommendedList,key=lambda d:d[1],reverse=True)
        recommendedList = [item[0] for item in recommendedList]
        #print ('user',user,'finished')
        return recommendedList
