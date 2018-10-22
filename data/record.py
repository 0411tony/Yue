import numpy as np
import time
from tool.config import Config,LineConfig
from tool.qmath import normalize
from tool.dataSplit import DataSplit
import os.path
from re import split
from collections import defaultdict
import time
import pickle
class Record(object):
    'data access control'
    def __init__(self,config,trainingSet,testSet):
        self.config = config
        self.recordConfig = LineConfig(config['record.setup'])
        self.evalConfig = LineConfig(config['evaluation.setup'])
        self.name2id = defaultdict(dict)
        self.id2name = defaultdict(dict)
        self.listened = {}
        self.listened['artist']=defaultdict(dict)
        self.listened['track']=defaultdict(dict)
        self.listened['album']=defaultdict(dict)
        self.artist2Album = defaultdict(dict) #key:artist id, value:{album id1:1, album id2:1 ...}
        self.album2Track = defaultdict(dict) #
        self.artist2Track = defaultdict(dict) #
        self.Track2artist = defaultdict(dict) #
        self.Track2album = defaultdict(dict) #
        self.userRecord = defaultdict(list) #user data in training set. form: {user:[record1,record2]}
        self.trackRecord = defaultdict(list) # track data in training set. form: {track:[record1, record2]}
        self.testSet = defaultdict(dict) #user data in test set. form: {user:{recommenedObject1:1,recommendedObject:1}}
        self.recordCount = 0
        self.columns = {}
        self.globalMean = 0
        self.userMeans = {} #used to store the mean values of users's listen tims
        self.trackListen = {}

        self.trainingData = trainingSet

        self.computeUserMean()
        self.globalAverage()
        self.PopTrack = {}

        labels = self.recordConfig['-columns'].split(',')
        for col in labels:
            label = col.split(':')
            self.columns[label[0]] = int(label[1])
        if self.evalConfig.contains('-byTime'):
            trainingSet,testSet = self.splitDataByTime(trainingSet)

        self.preprocess(trainingSet,testSet)


        self.computePop(trainingSet)
    
    def globalAverage(self):
        total = sum(self.userMeans.values())
        if total==0:
            self.globalMean = 0
        else:
            self.globalMean = total/len(self.userMeans)

    def computeUserMean(self):
        for user in self.userRecord:
            for item in self.userRecord[user]:
                userSum += self.listened['track'][item].values()
            
            self.userMeans[user] = userSum/float(len(self.userRecord[user]))
        
    ''' 
    def splitDataByTime(self,dataset):
        trainingSet = []
        testSet = []
        listened = {}
        ratio = float(self.evalConfig['-byTime'])
        records = defaultdict(list)
        for event in dataset:
            records[event['user']].append(event)
            if event['user'] not in listened:
                listened[event['user']] = 1
            else:
                listened[event['user']] += 1
        orderlist = sorted(listened.items(), key=lambda item:item[1], reverse=True)
        dellist = orderlist[:int(len(orderlist)*ratio)]
        for i in range(len(dellist)):
            if dellist[i][0] in records:
                del records[dellist[i][0]]

        #print('The amount of data after deletion:', len(records))

        for user in records:
            orderedList = sorted(records[user],key=lambda d:d['time'])
            training = orderedList[0:int(len(orderedList)*(1-ratio))]
            test = orderedList[int(len(orderedList)*(1-ratio)):]
            trainingSet += training
            testSet += test

        #print ('the type1 :', type(trainingSet), type(testSet))
        #file_train = 'trainset.txt'
        #file_test = 'testset.txt'
        #trainf = open(file_train, 'wb')
        #testf = open(file_test, 'wb')
        #pickle.dump(trainingSet, trainf, 2)
        #pickle.dump(testSet, testf, 2)
        #trainf.close()
        #testf.close()
        return trainingSet,testSet
    '''
    def splitDataByTime(self,dataset):
        trainingSet = []
        testSet = []
        ratio = float(self.evalConfig['-byTime'])
        records = defaultdict(list)
        for event in dataset:
            records[event['user']].append(event)

        for user in records:
            orderedList = sorted(records[user],key=lambda d:d['time'])
            training = orderedList[0:int(len(orderedList)*(1-ratio))]
            test = orderedList[int(len(orderedList)*(1-ratio)):]
            trainingSet += training
            testSet += test

        return trainingSet,testSet

    def computePop(self, dataset):
        print('computePop...')
        for event in dataset:
            total = 0
            for value in self.listened['track'][event['track']].values():
                total += value
                if value > 0:
                    self.PopTrack[event['track']] = total
            
        print('computePop is finished...')
        print('PopTrack', len(self.PopTrack))
        

    def preprocess(self,trainingSet,testSet):
        for entry in trainingSet:
            self.recordCount+=1
            for key in entry:
                if key!='time':
                    if entry[key] not in self.name2id[key]:
                        self.name2id[key][entry[key]] = len(self.name2id[key])
                        self.id2name[key][len(self.id2name[key])] = entry[key]

                if key=='user':
                    self.userRecord[entry['user']].append(entry)
                    if 'artist' in entry:
                        if entry[key] not in self.listened['artist'][entry['artist']]:
                            self.listened['artist'][entry['artist']][entry[key]] = 1
                        else:
                            self.listened['artist'][entry['artist']][entry[key]] += 1
                    if  'album' in entry:
                        if entry[key] not in self.listened['album'][entry['album']]:
                            self.listened['album'][entry['album']][entry[key]] = 1
                        else:
                            self.listened['album'][entry['album']][entry[key]] += 1
                    if 'track' in entry:
                        if entry[key] not in self.listened['track'][entry['track']]:
                            self.listened['track'][entry['track']][entry[key]] = 1
                        else:
                            self.listened['track'][entry['track']][entry[key]] += 1
                
                if key == 'artist' and 'album' in entry:
                    self.artist2Album[entry[key]][entry['album']] = 1

                if key == 'album' and 'track' in entry:
                    self.album2Track[entry[key]] = self.name2id['track'][entry['track']]
                    self.Track2album[entry['track']] = self.name2id[key][entry[key]]
                
                if key == 'artist' and 'track' in entry:
                    self.artist2Track[entry[key]] = self.name2id['track'][entry['track']]
                    self.Track2artist[entry['track']] = self.name2id[key][entry[key]]
                
                if key == 'track':
                    self.trackRecord[entry['track']].append(entry)



        recType = self.evalConfig['-target']
        for entry in testSet:
            for key in entry:
                if key != 'time':
                    if entry[key] not in self.name2id[key]:
                        self.name2id[key][entry[key]] = len(self.name2id[key])
                        self.id2name[key][len(self.id2name[key])] = entry[key]
                if key=='user':
                    if recType in entry and entry[recType] not in self.testSet[entry['user']]:
                        self.testSet[entry['user']][entry[recType]]=1
                    else:
                        self.testSet[entry['user']][entry[recType]]+=1

        #remove items appearing in the training set from the test set
        for item in self.listened[recType]:
            for user in self.listened[recType][item]:
                try:
                    del self.testSet[user][item]
                except KeyError:
                    pass
                if user in self.testSet and len(self.testSet[user])==0:
                    del self.testSet[user]
        


    def printTrainingSize(self):
        if 'user' in self.name2id:
            print ('user count:',len(self.name2id['user']))
        if 'artist' in self.name2id:
            print ('artist count:',len(self.name2id['artist']))
        if 'album' in self.name2id:
            print ('album count:',len(self.name2id['album']))
        if 'track' in self.name2id:
            print ('track count:', len(self.name2id['track']))
        print ('Training set size:',self.recordCount)


    def getId(self,obj,t):
        if obj in self.name2id[t]:
            return self.name2id[t][obj]
        else:
            print ('No '+t+' '+obj+' exists!')
            exit(-1)

    def getSize(self,t):
        return len(self.name2id[t])

    def contains(self, obj, t):
        'whether the recType t is in trainging set'
        if obj in self.name2id[t]:
            return True
        else:
            return False

    

