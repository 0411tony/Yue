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
        self.walkDim = int(options['-l'])
        self.winSize = int(options['-w'])
        self.topK = int(options['-k'])
        self.alpha = float(options['-a'])
        self.epoch = int(options['-ep'])
        self.neg = int(options['-neg'])
        self.rate = float(options['-r'])

    def printAlgorConfig(self):
        super(HME, self).printAlgorConfig()
        print 'Specified Arguments of', self.config['recommender'] + ':'
        print 'Walks count per user', self.walkCount
        print 'Length of each walk', self.walkLength
        print 'Dimension of user embedding', self.walkDim
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

        self.b = np.random.random(self.data.getSize(self.recType))
        self.G = np.random.rand(self.data.getSize(self.recType), self.k) / 10
        self.W = np.random.rand(self.data.getSize('user'), self.k) / 10

        self.user2track = defaultdict(list)
        self.user2artist = defaultdict(list)
        self.user2album = defaultdict(list)
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
                self.user2track[user].append(item[self.recType])
                self.user2artist[user].append(item['artist'])
                if self.data.columns.has_key('album'):
                    self.user2album[user].append(item['album'])

        for artist in self.data.artistListened:
            for user in self.data.artistListened[artist]:
                self.artist2user[artist] += [user] * self.data.artistListened[artist][user]

        for track in self.data.trackListened:
            for user in self.data.trackListened[track]:
                self.track2user[track] += [user] * self.data.trackListened[track][user]

        if self.data.columns.has_key('album'):
            for album in self.data.albumListened:
                for user in self.data.albumListened[album]:
                    self.album2user[album] += [user] * self.data.albumListened[album][user]

        for artist in self.data.artist2Track:
            self.artist2track = self.data.artist2Track[artist].keys()
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

        #global Preference
        self.walks = []
        # self.usercovered = {}
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


        #local Preference

        print 'walks:', len(self.walks)
        # Training get top-k friends
        print 'Generating user embedding...'

        self.topKSim = {}


        # for user in self.data.userRecord:
        #     playList = []
        #     for item in self.data.userRecord[user]:
        #         playList.append(item['track'])
        #     self.walks.append(playList)
        shuffle(self.walks)
        model = w2v.Word2Vec(self.walks, size=self.k, window=5, min_count=0, iter=self.epoch)
        for track in self.data.trackListened:
            tid = self.data.getId(track, 'track')
            try:
                self.Q[tid] = model.wv[track]
            except KeyError:
                pass

        self.R = np.zeros((self.data.getSize('user'), self.k))
        for user in self.data.userRecord:
            uid = self.data.getId(user,'user')
            # global_uv = np.zeros(self.k)
            # local_uv = np.zeros(self.k)
            # for event in self.data.userRecord[user]:
            #     tid = self.data.getId(event['track'],'track')
            #     global_uv +=self.Q[tid]
            # self.P[uid] = global_uv/len(self.data.userRecord[user])
            # recent = max(0,len(self.data.userRecord[user])-20)
            # for event in self.data.userRecord[user][recent:]:
            #     try:
            #         local_uv +=model.wv[event['track']]
            #     except KeyError:
            #         recent-=1
            # self.R[uid] = local_uv/recent
            self.R[uid] = model.wv[user]

        print 'User embedding generated.'
        #
        userListened = defaultdict(dict)
        for user in self.data.userRecord:
            for item in self.data.userRecord[user]:
                userListened[user][item[self.recType]] = 1

        print 'training...'
        iteration = 0
        itemList = self.data.name2id[self.recType].keys()
        while iteration < self.maxIter:
            self.loss = 0
            for user in self.data.userRecord:

                u = self.data.getId(user, 'user')
                for item in self.data.userRecord[user]:
                    i = self.data.getId(item[self.recType], self.recType)
                    item_j = choice(itemList)
                    while (userListened[user].has_key(item_j)):
                        item_j = choice(itemList)
                    j = self.data.getId(item_j, self.recType)
                    s = sigmoid(self.P[u].dot(self.Q[i]) - self.P[u].dot(self.Q[j]))
                    self.P[u] += self.lRate * (1 - s) * (self.Q[i] - self.Q[j])
                    self.Q[i] += self.lRate * (1 - s) * self.P[u]
                    self.Q[j] -= self.lRate * (1 - s) * self.P[u]
                    self.P[u] -= self.lRate * 0.5*(self.P[u]-self.R[u])
                    self.P[u] -= self.lRate * self.regU * self.P[u]
                    self.Q[i] -= self.lRate * self.regI * self.Q[i]
                    self.Q[j] -= self.lRate * self.regI * self.Q[j]
                    self.loss += -log(s)+0.5*(self.P[u]-self.R[u]).dot(self.P[u]-self.R[u])
            self.loss += self.regU * (self.P * self.P).sum() + self.regI * (self.Q * self.Q).sum()
            iteration += 1
            if self.isConverged(iteration):
                break



    def predict(self, u):
        'invoked to rank all the items for the user'
        u = self.data.getId(u, 'user')
        #return pairwise_distances(self.Q,[self.P[u]])
        return self.Q.dot(self.P[u])#+0.4*pairwise_distances(self.Q,[self.R[u]])