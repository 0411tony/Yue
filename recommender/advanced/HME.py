from base.IterativeRecommender import IterativeRecommender
from tool import config
from random import randint
from random import shuffle, choice
from collections import defaultdict
import numpy as np
from tool.qmath import sigmoid, cosine, cosine_sp
from math import log
import gensim.models.word2vec as w2v
from sklearn.metrics.pairwise import pairwise_distances

class HME(IterativeRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, fold='[1]'):
        super(HME, self).__init__(conf, trainingSet, testSet, fold)

    def readConfiguration(self):
        super(HME, self).readConfiguration()
        options = config.LineConfig(self.config['HME'])
        self.walkCount = int(options['-T'])
        self.walkLength = int(options['-L'])
        self.winSize = int(options['-w'])
        self.alpha = float(options['-alpha'])
        self.beta = float(options['-beta'])
        self.epoch = int(options['-ep'])


    def printAlgorConfig(self):
        super(HME, self).printAlgorConfig()
        print 'Specified Arguments of', self.config['recommender'] + ':'
        print 'Walks count per user', self.walkCount
        print 'Length of each walk', self.walkLength
        print 'alpha:',self.alpha,' beta:',self.beta
        print '=' * 80

    def buildModel(self):

        # data clean

        # li = self.sao.followees.keys()
        #
        print 'Kind Note: This method will probably take much time.'
        # build U-F-NET
        print 'Building weighted user-friend network...'
        # filter isolated nodes and low ratings
        # Definition of Meta-Path
        self.I = np.random.rand(self.data.getSize('track'),self.k)/10 #item characteristics
        self.G = np.random.rand(self.data.getSize('user'), self.k) / 10 #global user preference
        self.R = np.random.rand(self.data.getSize('user'), self.k) / 10 #recent user preference

        self.user2track = defaultdict(list)
        self.user2artist = defaultdict(list)
        self.user2album = defaultdict(list)

        self.r_user2track = defaultdict(list) #recent
        self.r_user2artist = defaultdict(list)
        self.r_user2album = defaultdict(list)
        
        self.track2user = defaultdict(list)
        self.artist2user = defaultdict(list)
        self.album2user = defaultdict(list)
        self.artist2track = defaultdict(list)
        self.artist2album = defaultdict(list)
        self.album2track = defaultdict(list)
        self.album2artist = {}
        self.track2artst = {}
        self.track2album = {}

        for user in self.data.userRecord:
            for item in self.data.userRecord[user]:
                self.user2track[user].append(item['track'])
                self.user2artist[user].append(item['artist'])
                if self.data.columns.has_key('album'):
                    self.user2album[user].append(item['album'])

            recent = max(0, len(self.data.userRecord[user]) - 20)
            for item in self.data.userRecord[user][recent:]:
                self.r_user2track[user].append(item['track'])
                self.r_user2artist[user].append(item['artist'])
                if self.data.columns.has_key('album'):
                    self.user2album[user].append(item['album'])

        for artist in self.data.listened['artist']:
            for user in self.data.listened['artist'][artist]:
                self.artist2user[artist] += [user] * self.data.listened['artist'][artist][user]

        for track in self.data.listened['track']:
            for user in self.data.listened['track'][track]:
                self.track2user[track] += [user] * self.data.listened['track'][track][user]

        if self.data.columns.has_key('album'):
            for album in self.data.listened['album']:
                for user in self.data.listened['album'][album]:
                    self.album2user[album] += [user] * self.data.listened['album'][album][user]

        for artist in self.data.artist2Track:
            self.artist2track[artist] = self.data.artist2Track[artist].keys()
            for key in self.data.artist2Track[artist]:
                self.track2artst[key] = artist
        if self.data.columns.has_key('album'):
            for album in self.data.album2Track:
                self.album2track[album] = self.data.album2Track[album].keys()
                for key in self.data.album2Track[album]:
                    self.track2album[key] = album

            for artist in self.data.artist2Album:
                self.artist2album[artist] = self.data.artist2Album[artist].keys()
                for key in self.data.artist2Album[artist]:
                    self.album2artist[key] = artist

        print 'Generating random meta-path random walks...'

        #global walks
        self.walks = []
        #recent walks
        self.r_walks = []
        p1 = 'UTU'
        p2 = 'UAU'
        p3 = 'UZU'
        p4 = 'UTATU'
        p5 = 'UTZTU'
        p6 = 'UTZAZTU'

        mPaths = []
        if self.data.columns.has_key('album'):
            mPaths = [p1, p2, p3, p4, p5, p6]
        else:
            mPaths = [p1, p2, p4]

        for user in self.data.userRecord:

            for mp in mPaths:
                for t in range(self.walkCount):

                    path = [user]
                    lastNode = user
                    nextNode = user
                    lastType = 'U'
                    for i in range(self.walkLength / len(mp[1:])):
                        for tp in mp[1:]:
                            try:
                                if tp == 'T' and lastType == 'U':
                                    nextNode = choice(self.user2track[lastNode])
                                elif tp == 'T' and lastType == 'A':
                                    nextNode = choice(self.artist2track[lastNode])
                                elif tp == 'T' and lastType == 'Z':
                                    nextNode = choice(self.album2track[lastNode])
                                elif tp == 'A' and lastType == 'T':
                                    nextNode = self.track2artst[lastNode]
                                elif tp == 'A' and lastType == 'Z':
                                    nextNode = self.album2artist[lastNode]
                                elif tp == 'A' and lastType == 'U':
                                    nextNode = choice(self.user2artist[lastNode])

                                elif tp == 'Z' and lastType == 'U':
                                    nextNode = choice(self.user2album[lastNode])
                                elif tp == 'Z' and lastType == 'A':
                                    nextNode = choice(self.artist2album[lastNode])
                                elif tp == 'Z' and lastType == 'T':
                                    nextNode = self.track2album[lastNode]

                                elif tp == 'U':
                                    if lastType == 'T':
                                        nextNode = choice(self.track2user[lastNode])
                                    elif lastType == 'Z':
                                        nextNode = choice(self.album2user[lastNode])
                                    elif lastType == 'A':
                                        nextNode = choice(self.artist2user[lastNode])

                                path.append(nextNode)
                                lastNode = nextNode
                                lastType = tp

                            except (KeyError, IndexError):
                                path = []
                                break

                    if path:
                        self.walks.append(path)
                        # for node in path:
                        #     if node[1] == 'U' or node[1] == 'F':
                        #         self.usercovered[node[0]] = 1
                        # print path
                        # if mp == 'UFIU':
                        # pass
        shuffle(self.walks)

        #recent random walks
        for user in self.data.userRecord:

            for mp in mPaths:
                for t in range(self.walkCount/2):

                    path = [user]
                    lastNode = user
                    nextNode = user
                    lastType = 'U'
                    for i in range(self.walkLength / len(mp[1:])):
                        for tp in mp[1:]:
                            try:
                                if tp == 'T' and lastType == 'U':
                                    nextNode = choice(self.r_user2track[lastNode])
                                elif tp == 'T' and lastType == 'A':
                                    nextNode = choice(self.artist2track[lastNode])
                                elif tp == 'T' and lastType == 'Z':
                                    nextNode = choice(self.album2track[lastNode])
                                elif tp == 'A' and lastType == 'T':
                                    nextNode = self.track2artst[lastNode]
                                elif tp == 'A' and lastType == 'Z':
                                    nextNode = self.album2artist[lastNode]
                                elif tp == 'A' and lastType == 'U':
                                    nextNode = choice(self.r_user2artist[lastNode])

                                elif tp == 'Z' and lastType == 'U':
                                    nextNode = choice(self.r_user2album[lastNode])
                                elif tp == 'Z' and lastType == 'A':
                                    nextNode = choice(self.artist2album[lastNode])
                                elif tp == 'Z' and lastType == 'T':
                                    nextNode = self.track2album[lastNode]

                                elif tp == 'U':
                                    if lastType == 'T':
                                        nextNode = choice(self.track2user[lastNode])
                                    elif lastType == 'Z':
                                        nextNode = choice(self.album2user[lastNode])
                                    elif lastType == 'A':
                                        nextNode = choice(self.artist2user[lastNode])

                                path.append(nextNode)
                                lastNode = nextNode
                                lastType = tp

                            except (KeyError, IndexError):
                                path = []
                                break

                    if path:
                        self.r_walks.append(path)
                        # for node in path:
                        #     if node[1] == 'U' or node[1] == 'F':
                        #         self.usercovered[node[0]] = 1
                        # print path
                        # if mp == 'UFIU':
                        # pass
        shuffle(self.r_walks)


        #local Preference

        print 'walks:', len(self.walks)
        print 'recent walks',len(self.r_walks)
        # Training get top-k friends
        print 'Generating user embedding...'


        # for user in self.data.userRecord:
        #     playList = []
        #     for item in self.data.userRecord[user]:
        #         playList.append(item['track'])
        #     self.walks.append(playList)
        g_model = w2v.Word2Vec()
        #g_model = w2v.Word2Vec(self.walks, size=self.k, window=self.winSize, min_count=0, iter=self.epoch)
        # for track in self.data.listened['track']:
        #     tid = self.data.getId(track, 'track')
        #     try:
        #         self.Q[tid] = model.wv[track]
        #     except KeyError:
        #         pass

        #self.R = np.zeros((self.data.getSize('user'), self.k))
        for user in self.data.userRecord:
            uid = self.data.getId(user,'user')
            try:
                self.G[uid] = g_model.wv[user]
            except KeyError:
                pass
        for item in self.data.name2id['track']:
            iid = self.data.getId(item,'track')
            try:
                self.I[iid] = g_model.wv[item]
            except KeyError:
                pass
        r_model = w2v.Word2Vec(self.r_walks, size=self.k, window=self.winSize, min_count=0, iter=self.epoch)
        for user in self.data.userRecord:
            uid = self.data.getId(user,'user')
            self.R[uid] = r_model.wv[user]

        for item in self.data.listened[self.recType]:
            iid = self.data.getId(item,self.recType)
            try:
                self.Q[iid] = g_model.wv[item]
            except KeyError:
                pass

        print 'User embedding generated.'
        #
        userListened = defaultdict(dict)
        for user in self.data.userRecord:
            for item in self.data.userRecord[user]:
                userListened[user][item['track']] = 1

        print 'training...'
        iteration = 0
        itemList = self.data.name2id['track'].keys()
        while iteration < self.maxIter:
            self.loss = 0
            for user in self.data.userRecord:

                u = self.data.getId(user, 'user')
                for item in self.data.userRecord[user]:
                    i = self.data.getId(item['track'], 'track')
                    item_j = choice(itemList)
                    while (userListened[user].has_key(item_j)):
                        item_j = choice(itemList)
                    j = self.data.getId(item_j, 'track')
                    s = sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))
                    self.P[u] += self.lRate * (1 - s) * (self.Q[i] - self.Q[j])
                    self.Q[i] += self.lRate * (1 - s) * self.P[u]
                    self.Q[j] -= self.lRate * (1 - s) * self.P[u]
                    self.P[u] -= self.lRate * self.alpha*(self.beta*(self.P[u]-self.G[u])+(1-self.beta)*(self.P[u]-self.R[u]))
                    self.P[u] -= self.lRate * self.regU * self.P[u]
                    #self.Q[i] -= self.lRate * self.alpha*(self.Q[i]-self.I[i])
                    self.Q[i] -= self.lRate * self.regI * self.Q[i]
                    self.Q[j] -= self.lRate * self.regI * self.Q[j]
                    #self.Q[j] -= self.lRate * self.alpha * (self.Q[j] - self.I[j])
                    self.loss += -log(s)
            self.loss += self.regU * (self.P * self.P).sum() + self.regI * (self.Q * self.Q).sum()\
                         +self.alpha*((1-self.beta)*((self.P-self.R)*(self.P-self.R)).sum()+
                         self.beta*((self.P-self.G)*(self.P-self.G)).sum())#+\
                         #self.alpha*((self.Q-self.I)*(self.Q-self.I)).sum()
            iteration += 1
            if self.isConverged(iteration):
                break



    def predict(self, u):
        'invoked to rank all the items for the user'
        u = self.data.getId(u, 'user')
        #return pairwise_distances(self.Q,[self.P[u]])
        return self.Q.dot(self.P[u])#+0.4*pairwise_distances(self.Q,[self.R[u]])