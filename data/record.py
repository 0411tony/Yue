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
        self.artistListened = defaultdict(dict) #key:user id, value:{artist id1:count, artist id2:count, ...}
        self.albumListened = defaultdict(dict) #key:user id, value:{album id1:count, album id2:count, ...}
        self.trackListened = defaultdict(dict) #key:user id, value:{track id1:count, track id2:count, ...}
        self.artist2Album = defaultdict(dict) #key:artist id, value:{album id1:1, album id2:1 ...}
        self.album2Track = defaultdict(dict) #
        self.artist2Track = defaultdict(dict) #
        self.userRecord = defaultdict(dict) #user data in training set. form: {user:{record1,record2}}
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
                        if not self.artistListened[entry[key]].has_key(entry['artist']):
                            self.artistListened[entry[key]][entry['artist']] = 0
                        else:
                            self.artistListened[entry[key]][entry['artist']] += 1
                    if  entry.has_key('album'):
                        if not self.albumListened[entry[key]].has_key(entry['album']):
                            self.albumListened[entry[key]][entry['album']] = 0
                        else:
                            self.albumListened[entry[key]][entry['album']] += 1
                    if entry.has_key('track'):
                        if not self.trackListened[entry[key]].has_key(entry['track']):
                            self.trackListened[entry[key]][entry['track']] = 0
                        else:
                            self.trackListened[entry[key]][entry['track']] += 1
                if key == 'artist' and entry.has_key('album'):
                        self.artist2Track[entry[key]][entry['artist']] = 1
                if key == 'album' and entry.has_key('track'):
                        self.album2Track[entry[key]][entry['artist']] = 1
                if key == 'artist' and entry.has_key('track'):
                    self.artist2Track[entry[key]][entry['artist']] = 1


        recommendedType = self.evalConfig['-target']
        for entry in testSet:
            for key in entry:
                if key != 'time':
                    if not self.id[key].has_key(entry[key]):
                        self.id[key][entry[key]] = len(self.id[key])
                        self.name2id[key][len(self.name2id[key])] = entry[key]
                if key=='user':
                    if entry.has_key(recommendedType):
                        testSet[entry['user']][entry[recommendedType]]=1




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





