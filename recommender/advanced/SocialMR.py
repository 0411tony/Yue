from base.IterativeRecommender import IterativeRecommender
from tool import config
from random import randint
from random import shuffle, choice
from collections import defaultdict
import numpy as np
from tool.qmath import sigmoid, cosine, cosine_sp
from math import log
import gensim.models.word2vec as w2v


class SocialMR(IterativeRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, fold='[1]'):
        super(SocialMR, self).__init__(conf, trainingSet, testSet, fold)

    def readConfiguration(self):
        super(SocialMR, self).readConfiguration()
        options = config.LineConfig(self.config['SocialMR'])
        self.walkCount = int(options['-T'])
        self.walkLength = int(options['-L'])
        self.walkDim = int(options['-l'])
        self.winSize = int(options['-w'])
        self.topK = int(options['-k'])
        self.alpha = float(options['-a'])
        self.epoch = int(options['-ep'])


    def printAlgorConfig(self):
        super(SocialMR, self).printAlgorConfig()
        print 'Specified Arguments of', self.config['recommender'] + ':'
        print 'Walks count per user', self.walkCount
        print 'Length of each walk', self.walkLength
        print 'Dimension of user embedding', self.walkDim
        print '=' * 80

    def buildModel(self):

        #data clean

        # li = self.sao.followees.keys()
        #
        print 'Kind Note: This method will probably take much time.'
        # build U-F-NET
        print 'Building weighted user-friend network...'
        # filter isolated nodes and low ratings
        # Definition of Meta-Path

        self.W = np.random.rand(self.data.getSize('user'), self.walkDim) / 10

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
        
        for artist in self.data.listened['artist']:
            for user in self.data.listened['artist'][artist]:
                self.artist2user[artist]+=[user]*self.data.listened['artist'][artist][user]
        
        for track in self.data.listened['track']:
            for user in self.data.listened['track'][track]:
                self.track2user[track]+=[user]*self.data.listened['track'][track][user]

        if self.data.columns.has_key('album'):
            for album in self.data.listened['album']:
                for user in self.data.listened['album'][album]:
                    self.album2user[album]+=[user]*self.data.listened['album'][album][user]

        for artist in self.data.artist2Track:
            self.artist2track[artist]=self.data.artist2Track[artist].keys()
            for key in self.data.artist2Track[artist]:
                self.track2artst[key] = artist
        if self.data.columns.has_key('album'):
            for album in self.data.album2Track:
                self.album2track[album] = self.data.album2Track[album].keys()
                for key in self.data.album2Track[album]:
                    self.track2album[key]=album

            for artist in self.data.artist2Album:
                self.artist2album[artist] = self.data.artist2Album[artist].keys()
                for key in self.data.artist2Album[artist]:
                    self.album2artist[key]=artist


        print 'Generating random meta-path random walks...'
        self.walks = []
        #self.usercovered = {}
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
                                if tp == 'T' and lastType=='U':
                                    nextNode = choice(self.user2track[lastNode])
                                elif tp == 'T' and lastType=='A':
                                    nextNode = choice(self.artist2track[lastNode])
                                elif tp=='T' and lastType=='Z':
                                    nextNode = choice(self.album2track[lastNode])
                                elif tp=='A' and lastType=='T':
                                    nextNode = self.track2artst[lastNode]
                                elif tp=='A' and lastType=='Z':
                                    nextNode = self.album2artist[lastNode]
                                elif tp == 'A' and lastType=='U':
                                    nextNode = choice(self.user2artist[lastNode])
                                
                                elif tp=='Z' and lastType=='U':
                                    nextNode = choice(self.user2album[lastNode])
                                elif tp=='Z' and lastType=='A':
                                    nextNode = choice(self.artist2album[lastNode])
                                elif tp=='Z' and lastType=='T':
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
        print 'walks:', len(self.walks)
        # Training get top-k friends
        print 'Generating user embedding...'

        self.topKSim = {}

        model = w2v.Word2Vec(self.walks, size=self.walkDim, window=5, min_count=0, iter=self.epoch)

        for user in self.data.userRecord:
            uid = self.data.getId(user,'user')
            self.W[uid] = model.wv[user]
        print 'User embedding generated.'

        print 'Constructing similarity matrix...'
        i = 0


        for user1 in self.data.userRecord:
            uSim = []
            i+=1
            if i%200==0:
                print i,'/',len(self.data.userRecord)
            vec1 = self.W[self.data.getId(user1,'user')]
            for user2 in self.data.userRecord:
                if user1 <> user2:
                    vec2 = self.W[self.data.getId(user2,'user')]
                    sim = cosine(vec1, vec2)
                    uSim.append((user2,sim))

            self.topKSim[user1] = sorted(uSim, key=lambda d: d[1], reverse=True)[:self.topK]


        # print 'Similarity matrix finished.'
        # # # #print self.topKSim
        #import pickle
        # # # #
        # # # #recordTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        # similarity = open('SocialMR-lastfm-sim'+self.foldInfo+'.pkl', 'wb')
        # vectors = open('SocialMR-lastfm-vec'+self.foldInfo+'.pkl', 'wb')
        # #Pickle dictionary using protocol 0.
        #
        # pickle.dump(self.topKSim, similarity)
        # pickle.dump((self.W,self.G),vectors)
        # similarity.close()
        # vectors.close()

        # matrix decomposition
        #pkl_file = open('SocialMR-lastfm-sim' + self.foldInfo + '.pkl', 'rb')

        #self.topKSim = pickle.load(pkl_file)


#        self.F = np.random.rand(self.data.trainingSize()[0], self.k) / 10
        # prepare Pu set, IPu set, and Nu set

        print 'Preparing item sets...'
        self.PositiveSet = defaultdict(dict)
        self.pSet = defaultdict(list)
        self.IPositiveSet = defaultdict(dict)
        self.ipSet = defaultdict(list)
        # self.NegativeSet = defaultdict(list)

        for user in self.data.userRecord:
            for item in self.data.userRecord[user]:
                self.PositiveSet[user][item['track']] = 1
                self.pSet[user].append(item['track'])

            for friend,sim in self.topKSim[user]:
                    for item in self.data.userRecord[friend]:
                        if not self.PositiveSet[user].has_key(item['track']):
                            self.IPositiveSet[user][item['track']] = 1
                            self.ipSet[user].append(item['track'])

        Suk = 0.5
        print 'Training...'
        iteration = 0
        while iteration < self.maxIter:
            self.loss = 0
            itemList = self.data.name2id['track'].keys()
            for user in self.pSet:
                u = self.data.getId(user,'user')
                kItems = self.ipSet[user]
                for item in self.pSet[user]:
                    i = self.data.getId(item,'track')
                    if len(self.ipSet[user]) > 0:
                        item_k = choice(kItems)
                        k = self.data.getId(item_k,'track')
                        s1 = sigmoid((self.P[u].dot(self.Q[i])- self.P[u].dot(self.Q[k])) / (Suk + 1))
                        self.P[u] += 1 / (Suk + 1) * self.lRate * (1 - s1) * (self.Q[i] - self.Q[k])
                        self.Q[i] += 1 / (Suk + 1) * self.lRate * (1 - s1) * self.P[u]
                        self.Q[k] -= 1 / (Suk + 1) * self.lRate * (1 - s1) * self.P[u]
                        item_j = ''
                        # if len(self.NegativeSet[user])>0:
                        #     item_j = choice(self.NegativeSet[user])
                        # else:
                        item_j = choice(itemList)
                        while (self.PositiveSet[user].has_key(item_j) or self.IPositiveSet[user].has_key(item_j)):
                            item_j = choice(itemList)
                        j = self.data.getId(item_j,'track')
                        s2 = sigmoid(self.P[u].dot(self.Q[k]) - self.P[u].dot(self.Q[j]) )
                        self.P[u] += self.lRate * (1 - s2) * (self.Q[k] - self.Q[j])
                        self.Q[k] += self.lRate * (1 - s2) * self.P[u]
                        self.Q[j] -= self.lRate * (1 - s2) * self.P[u]


                        self.P[u] -= self.lRate * self.regU * self.P[u]
                        self.Q[i] -= self.lRate * self.regI * self.Q[i]
                        self.Q[j] -= self.lRate * self.regI * self.Q[j]
                        self.Q[k] -= self.lRate * self.regI * self.Q[k]

                        self.loss += -log(s1)-log(s2)

                    else:
                        item_j = choice(itemList)
                        while (self.PositiveSet[user].has_key(item_j)):
                            item_j = choice(itemList)
                        j = self.data.getId(item_j,'track')
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
        u = self.data.getId(u,'user')
        return self.Q.dot(self.P[u])