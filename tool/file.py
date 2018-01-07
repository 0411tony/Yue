import os.path
from os import makedirs,remove
from re import compile,findall,split
from config import LineConfig
class FileIO(object):
    def __init__(self):
        pass

    @staticmethod
    def writeFile(dir,file,content,op = 'w'):
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(dir+file,op) as f:
            f.writelines(content)

    @staticmethod
    def deleteFile(filePath):
        if os.path.exists(filePath):
            remove(filePath)


    @staticmethod
    def loadDataSet(file, columns,binarized = False, threshold = 3, delim='' ):
        print 'load dataset...'
        record = []
        colNames = columns.keys()
        if len(colNames) < 2:
            print 'The dataset needs more information or the record.setup setting has some problems...'
            exit(-1)
        index = [int(item) for item in columns.values()]
        delimiter=',| |\t'
        if delim != '':
            delimiter = delim
        with open(file) as f:
            lineNo = 0
            for line in f:
                lineNo += 1
                try:
                    items = split(delimiter, line.strip())
                    event = {}
                    for column,ind in zip(colNames,index):
                        event[column] = items[ind]
                        if binarized and event.has_key('play'):
                            if int(event['play']) >= threshold:
                                event['play'] = 1
                            else:
                                event['play'] = 0
                    record.append(event)
                except IndexError:
                    print 'The record file is not in a correct format. Error Location: Line num %d' % lineNo
                    exit(-1)
        return record





