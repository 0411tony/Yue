from base.IterativeRecommender import IterativeRecommender
from tool import config
from random import randint
from random import shuffle,choice
from collections import defaultdict
import numpy as np
from tool.qmath import sigmoid,cosine
from math import log
import gensim.models.word2vec as w2v

class CUNE(IterativeRecommender):
    def __init__(self,conf,trainingSet=None,testSet=None,fold='[1]'):
        super(CUNE, self).__init__(conf,trainingSet,testSet,fold)        

    def readConfiguration(self):
        super(CUNE, self).readConfiguration()
        options = config.LineConfig(self.config['CUNE'])
        self.walkCount = int(options['-T'])
        self.walkLength = int(options['-L'])
        self.walkDim = int(options['-l'])
        self.winSize = int(options['-w'])
        self.topK = int(options['-k'])
        self.s = float(options['-s'])
        self.epoch = int(options['-ep'])

    def printAlgorConfig(self):
        super(CUNE, self).printAlgorConfig()
        print ('Specified Arguments of', self.config['recommender'] + ':')
        print ('Walks count per user', self.walkCount)
        print ('Length of each walk', self.walkLength)
        print ('Dimension of user embedding', self.walkDim)
        print ('='*80)

    def buildModel(self):
        print ('Kind Note: This method will probably take much time.')
        #build C-U-NET
        print ('Building collaborative user network...')

        userListen = defaultdict(dict)
        for user in self.data.userRecord:
            for item in self.data.userRecord[user]:
                userListen[user][item[self.recType]] = 1
        self.CUNet = defaultdict(list)

        for user1 in userListen:
            s1 = set(userListen[user1].keys())
            for user2 in userListen:
                if user1 != user2:
                    s2 = set(userListen[user2].keys())
                    weight = len(s1.intersection(s2))
                    if weight > 0:
                        self.CUNet[user1]+=[user2]*weight

        print ('Generating random deep walks...')
        self.walks = []
        self.visited = defaultdict(dict)
        for user in self.CUNet:
            for t in range(self.walkCount):
                path = [user]
                lastNode = user
                for i in range(1,self.walkLength):
                    nextNode = choice(self.CUNet[lastNode])
                    count=0
                    while(nextNode in self.visited[lastNode]):
                        nextNode = choice(self.CUNet[lastNode])
                        #break infinite loop
                        count+=1
                        if count==10:
                            break
                    path.append(nextNode)
                    self.visited[user][nextNode] = 1
                    lastNode = nextNode
                self.walks.append(path)
        shuffle(self.walks)

        #Training get top-k friends
        print ('Generating user embedding...')       
        model = w2v.Word2Vec(self.walks, size=self.walkDim, window=self.winSize, min_count=0, iter=self.epoch)
        print ('User embedding generated.')

        print ('Constructing similarity matrix...')
        self.W = np.random.rand(self.data.getSize('user'), self.k) / 10  # global user preference
        self.topKSim = {}
        i = 0
        for user in self.CUNet:
            u = self.data.getId(user,'user')
            self.W[u] = model.wv[user]
        for user1 in self.CUNet:
            sims = []
            u1 = self.data.getId(user1,'user')
            for user2 in self.CUNet:
                if user1 != user2:
                    u2 = self.data.getId(user2,'user')
                    sims.append((user2,cosine(self.W[u1],self.W[u2])))
            self.topKSim[user1] = sorted(sims, key=lambda d: d[1], reverse=True)[:self.topK]
            i += 1
            if i % 200 == 0:
                print ('progress:', i, '/', len(self.CUNet))
        print ('Similarity matrix finished.')
        #print self.topKSim

        #prepare Pu set, IPu set, and Nu set
        print ('Preparing item sets...')
        self.PositiveSet = defaultdict(list)
        self.IPositiveSet = defaultdict(list)
        #self.NegativeSet = defaultdict(list)
        for user in self.data.userRecord:
            for event in self.data.userRecord[user]:
                self.PositiveSet[user].append(event[self.recType])


        for user in self.CUNet:
            for friend in self.topKSim[user]:
                self.IPositiveSet[user] += list(set(self.PositiveSet[friend[0]]).difference(self.PositiveSet[user]))



        print ('Training...')
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            itemList = list(self.data.name2id[self.recType].keys())
            for user in self.PositiveSet:
                u = self.data.getId(user,'user')

                for item in self.PositiveSet[user]:
                    i = self.data.getId(item,self.recType)
                    for n in range(3):
                        if len(self.IPositiveSet[user]) > 0:
                            item_k = choice(self.IPositiveSet[user])

                            k = self.data.getId(item_k,self.recType)
                            self.P[u] += self.lRate * (1 - sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[k]))) * (
                            self.Q[i] - self.Q[k])
                            self.Q[i] += self.lRate * (1 - sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[k]))) * \
                                        self.P[u]
                            self.Q[k] -= self.lRate * (1 - sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[k]))) * \
                                        self.P[u]

                            item_j = ''
                            # if len(self.NegativeSet[user])>0:
                            #     item_j = choice(self.NegativeSet[user])
                            # else:
                            item_j = choice(itemList)
                            while (user in self.data.listened[self.recType][item_j]):
                                item_j = choice(itemList)
                            j = self.data.getId(item_j,self.recType)
                            self.P[u] += (1 / self.s) * self.lRate * (
                            1 - sigmoid((1 / self.s) * (self.P[u].dot(self.Q[k]) - self.P[u].dot(self.Q[j])))) * (
                                        self.Q[k] - self.Q[j])
                            self.Q[k] += (1 / self.s) * self.lRate * (
                            1 - sigmoid((1 / self.s) * (self.P[u].dot(self.Q[k]) - self.P[u].dot(self.Q[j])))) * self.P[u]
                            self.Q[j] -= (1 / self.s) * self.lRate * (
                            1 - sigmoid((1 / self.s) * (self.P[u].dot(self.Q[k]) - self.P[u].dot(self.Q[j])))) * self.P[u]

                            self.P[u] -= self.lRate * self.regU * self.P[u]
                            self.Q[i] -= self.lRate * self.regI * self.Q[i]
                            self.Q[j] -= self.lRate * self.regI * self.Q[j]
                            self.Q[k] -= self.lRate * self.regI * self.Q[k]

                            self.loss += -log(sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[k]))) - \
                                        log(sigmoid((1 / self.s) * (self.P[u].dot(self.Q[k]) - self.P[u].dot(self.Q[j]))))
                        else:
                            item_j = choice(itemList)
                            while (user in self.data.listened[self.recType][item_j]):
                                item_j = choice(itemList)
                            j = self.data.getId(item_j,self.recType)
                            self.P[u] += self.lRate * (1 - sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))) * (self.Q[i] - self.Q[j])
                            self.Q[i] += self.lRate * (1 - sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))) * self.P[u]
                            self.Q[j] -= self.lRate * (1 - sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))) * self.P[u]

                            self.loss += -log(sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j])))


                self.loss += self.regU*(self.P*self.P).sum() + self.regI*(self.Q*self.Q).sum()
            iteration += 1
            if self.isConverged(iteration):
                break


    def predict(self, u):
        'invoked to rank all the items for the user'
        u = self.data.getId(u, 'user')
        return self.Q.dot(self.P[u])
