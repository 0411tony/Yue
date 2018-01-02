import numpy as np
from tool.config import Config,LineConfig
from tool.qmath import normalize
from tool.dataSplit import DataSplit
import os.path
from re import split
from collections import defaultdict
class RatingDAO(object):
    'data access control'
    def __init__(self,config,trainingSet,testSet):
        self.config = config
        self.recordConfig = LineConfig(config['record.setup'])
        self.
        self.users = {} #store the id of users
        self.artists = {} #store the id of artists
        self.albums = {} #store the id of albums
        self.tracks = {} #store the id of tracks
        self.artistsListened = defaultdict(dict) #key:user id, value:{artist id1:count, artist id2:count, ...}
        self.albumsListened = defaultdict(dict) #key:user id, value:{album id1:count, album id2:count, ...}
        self.tracksListened = defaultdict(dict) #key:user id, value:{track id1:count, track id2:count, ...}
        self.artist2Albums = defaultdict(dict) #key:artist id, value:{album id1:1, album id2:1 ...}
        self.albums2Tracks = defaultdict(dict) #
        self.artist2Tracks = defaultdict(dict) #
        self.userRecords = defaultdict(list) #user data in training set. form: {user:{record1,record2}}
        self.testSet = defaultdict(dict) #user data in test set. form: {user:{record1,record2}}

        self.preprocess(trainingSet,testSet)



    def preprocess(self,trainingSet, testSet):





        for i,entry in enumerate(self.trainingData):
            userName,itemName,rating = entry
            # makes the rating within the range [0, 1].
            rating = normalize(float(rating), self.rScale[-1], self.rScale[0])
            self.trainingData[i][2] = rating
            # order the user
            if not self.user.has_key(userName):
                self.user[userName] = len(self.user)
                self.id2user[self.user[userName]] = userName
            # order the item
            if not self.item.has_key(itemName):
                self.item[itemName] = len(self.item)
                self.id2item[self.item[itemName]] = itemName
                # userList.append
            self.trainSet_u[userName][itemName] = rating
            self.trainSet_i[itemName][userName] = rating

        self.all_User.update(self.user)
        self.all_Item.update(self.item)
        for entry in self.testData:
            userName, itemName, rating = entry
            # order the user
            if not self.user.has_key(userName):
                self.all_User[userName] = len(self.all_User)
            # order the item
            if not self.item.has_key(itemName):
                self.all_Item[itemName] = len(self.all_Item)

            self.testSet_u[userName][itemName] = rating
            self.testSet_i[itemName][userName] = rating




    def trainingSize(self):
        return (len(self.user),len(self.item),len(self.trainingData))

    def testSize(self):
        return (len(self.testSet_u),len(self.testSet_i),len(self.testData))

    def contains(self,u,i):
        'whether user u rated item i'
        if self.user.has_key(u) and self.trainSet_u[u].has_key(i):
            return True
        else:
            return False



    def sRow(self,u):
        return self.trainSet_u[u]

    def sCol(self,c):
        return self.trainSet_i[c]

    def rating(self,u,c):
        if self.contains(u,c):
            return self.trainSet_u[u][c]
        return -1

    def ratingScale(self):
        return (self.rScale[0],self.rScale[1])

    def elemCount(self):
        return len(self.trainingData)
