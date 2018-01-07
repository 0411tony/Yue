import numpy as np
from tool.config import Config,LineConfig
from tool.qmath import normalize
from tool.dataSplit import DataSplit
import os.path
from re import split
from collections import defaultdict
class Record(object):
    'data access control'
    def __init__(self,config,trainingSet,testSet):
        self.config = config
        self.recordConfig = LineConfig(config['record.setup'])
        self.evalConfig = LineConfig(config['evaluation.setup'])
        self.id = defaultdict(dict)
        self.name2id = defaultdict(dict)
        self.artistListened = defaultdict(dict) #key:aritst id, value:{user id1:count, user id2:count, ...}
        self.albumListened = defaultdict(dict) #key:album id, value:{user id1:count, user id2:count, ...}
        self.trackListened = defaultdict(dict) #key:track id, value:{user id1:count, user id2:count, ...}
        self.artist2Album = defaultdict(dict) #key:artist id, value:{album id1:1, album id2:1 ...}
        self.album2Track = defaultdict(dict) #
        self.artist2Track = defaultdict(dict) #
        self.userRecord = defaultdict(list) #user data in training set. form: {user:[record1,record2]}
        self.testSet = defaultdict(dict) #user data in test set. form: {user:{recommenedObject1:1,recommendedObject:1}}

        self.preprocess(trainingSet,testSet)


    def preprocess(self,trainingSet,testSet):
        for entry in trainingSet:
            for key in entry:
                if key!='time':
                    if not self.id[key].has_key(entry[key]):
                        self.id[key][entry[key]] = len(self.id[key])
                        self.name2id[key][len(self.name2id[key])] = entry[key]

                if key=='user':
                    self.userRecord[entry['user']].append(entry)
                    if entry.has_key('artist'):
                        if not self.artistListened[entry['artist']].has_key(entry[key]):
                            self.artistListened[entry['artist']][entry[key]] = 0
                        else:
                            self.artistListened[entry['artist']][entry[key]] += 1
                    if  entry.has_key('album'):
                        if not self.albumListened[entry['album']].has_key(entry[key]):
                            self.albumListened[entry['album']][entry[key]] = 0
                        else:
                            self.albumListened[entry['album']][entry[key]] += 1
                    if entry.has_key('track'):
                        if not self.trackListened[entry['track']].has_key(entry[key]):
                            self.trackListened[entry['track']][entry[key]] = 0
                        else:
                            self.trackListened[entry['track']][entry[key]] += 1
                if key == 'artist' and entry.has_key('album'):
                        self.artist2Track[entry[key]][entry['album']] = 1
                if key == 'album' and entry.has_key('track'):
                        self.album2Track[entry[key]][entry['track']] = 1
                if key == 'artist' and entry.has_key('track'):
                    self.artist2Track[entry[key]][entry['artist']] = 1


        recType = self.evalConfig['-target']
        for entry in testSet:
            for key in entry:
                if key != 'time':
                    if not self.id[key].has_key(entry[key]):
                        self.id[key][entry[key]] = len(self.id[key])
                        self.name2id[key][len(self.name2id[key])] = entry[key]
                if key=='user':
                    if entry.has_key(recType):
                        self.testSet[entry['user']][entry[recType]]=1


    def printTrainingSize(self):
        if self.id.has_key('user'):
            print 'user count:',len(self.id['user'])
        if self.id.has_key('artist'):
            print 'artist count:',len(self.id['artist'])
        if self.id.has_key('album'):
            print 'album count:',len(self.id['album'])
        if self.id.has_key('track'):
            print 'track count:', len(self.id['track'])
        print 'Training set size:',len(self.userRecord)





