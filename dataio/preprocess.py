"""
From a list of contents(self.ques), generate vocabs which is vocab of the contens, generate inputsData
which is a matrix of word indexes, and provide labels.
"""
from __future__ import print_function
import csv
import numpy as np
from compiler.ast import flatten
import sys
import gzip
from nntools.utils import clean_str
import re

class Dataset_oi(object):

    def __init__(self, datadir, rnd_seed,select_columns = 'level',oned_flag = False):
        '''
            'GetPay', 'AssoPay', 'WorkTime', 'WorkPlace', 'JobRel', 'DiseRel', 'OutForPub',
            'OnOff', 'InjIden', 'EndLabor', 'LaborContr', 'ConfrmLevel', 'Level', 'Insurance', 'HaveMedicalFee'],
 
        '''

        self.oned_flag = oned_flag
        self.rnd_seed = rnd_seed
        self.train_path = datadir+'/train.txt'
        self.train_label_path = datadir+'/train_label.txt'
        self.valid_path = datadir+'/valid.txt'
        self.valid_label_path = datadir+'/valid_label.txt'
        self.test_path = datadir+'/test.txt'
        self.test_label_path = datadir+'/test_label.txt'
        self.train_docs, self.train_labels = self.readfile(self.train_path,\
                self.train_label_path)
        self.valid_docs, self.valid_labels = self.readfile(self.valid_path,\
                self.valid_label_path)
        self.test_docs, self.test_labels = self.readfile(self.test_path,\
                self.test_label_path)


        self.vocab = self.genVocab(self, self.train_docs)
        self.train_data = self.word2index( self.train_docs, self.vocab)
        self.valid_data = self.word2index(self.valid_docs, self.vocab)
        self.test_data = self.word2index(self.test_docs, self.vocab)
        
        print(select_columns)
        if select_columns == 'level':
            slt_index = [12]
        elif select_columns == 'ii':
            slt_index = [8]
        if select_columns == 'all':
            slt_index = [i for i in range(15)]


        self.train_labels = self.train_labels[:,slt_index]
        self.valid_labels = self.valid_labels[:,slt_index]
        self.test_labels = self.test_labels[:,slt_index]
        
        print('loaded data {} labels {}'.format(len(self.train_data), self.train_labels.shape))

    @staticmethod
    def readfile(datapath, labelpath):
        source = open(datapath,'r').read().decode('UTF-8')  
        lines = source.split('\n')
        docs = [line for line in lines if len(line)>0]

        source = open(labelpath,'r').read()
        source = source.replace('\xef\xbb\xbf','')
        lines_label = source.split('\n')
        labels = [[int(x) for x in label.split()] for label in lines_label if len(label)>0]

        labels = np.array(labels, dtype='int')
        assert len(docs)==len(labels)
        assert labels.shape[1] == 15
        
        return docs, labels

    @staticmethod
    def genVocab(self,lines, maskid=0):
        """generate vocabulary from contents"""
        lines = [' '.join(line) for line in lines]
        wordset = set(item for line in lines for item in line.strip().split())
        word2index = {word: index + 1 for index, word in enumerate(wordset)}
        word2index['<mask>'] = maskid
        word2index['unk'] = len(word2index)
        return word2index

    def word2index(self, docs, vocab):
        docs = [' '.join(line) for line in docs]
        index_docs = [[self.getIndex(char) for char in doc.strip().split()] for doc in docs]
        # max_len = max([len(doc) for doc in index_docs])
        # index_docs = [doc+[vocab['mask']]*(max_len - len(doc)) for doc in index_docs]
        # index_docs = np.array(index_docs)
        return index_docs

    def getIndex(self,word):
        if self.vocab.has_key(word):
            return self.vocab[word]
        else:
            return self.vocab['unk']

    def getInputs(self):
        return self.index_docs

    def getLabels(self):
        return self.labels

    def getVocab(self):
        return self.vocab

    def reshape_split(self,docs, is_train = True):
        '''
        todo: add a char # at the back of the each punctuation
        :return: (batch__size,num_subparam,num_char)
        '''

        docs_inputs = []
        docs = [' '.join(line) for line in docs]
        max_subp_len  = 0
        max_doc_len = 0
        n_len = 0
        for doc in docs:
            doc_1 = doc.replace(u'\uff0c',u'\uff0c'+' #')
            doc_1 = doc_1.replace(u'\uff01', u'\uff01'+' #')
            doc_1 = doc_1.replace(u'\uff1f', u'\uff1f' + ' #')
            doc_1 = doc_1.replace(u'\u3002', u'\u3002'+' #')
            doc_1 = doc_1.replace(u'\uff1b', u'\uff1b'+' #')
            doc_1 = doc_1.replace(u',', u',' + ' #')
            doc_1 = doc_1.replace(u'!', u'!' + ' #')
            doc_1 = doc_1.replace(u'?', u'?' + ' #')
            doc_1 = doc_1.replace(u'.', u'.' + ' #')
            doc_1 = doc_1.replace(u';', u';' + ' #')
            docs_inputs.append(doc_1.strip(' #'))

        if is_train:
            self.vocab = self.genVocab(self,docs_inputs)

        docs1 = []
        for doc in docs_inputs:
            subparam_inputs = [[self.getIndex(char) for char in subparam.strip().split()] for subparam in doc.strip().split('#')]

            #if len(subparam_inputs)>max_doc_len:
            #    max_doc_len  = len(subparam_inputs)
            docs1.append(subparam_inputs)
        
            n_len += len(subparam_inputs)
        return docs1

    def get3Ddata(self):
        train_num = len(self.train_docs)
        self.vocab['#'] = len(self.vocab)
        self.data = self.reshape_split(self.train_docs)
        self.labels = self.train_labels
        self.valid_data = self.reshape_split(self.valid_docs, False)
        self.test_data = self.reshape_split(self.test_docs, False)
        print('loaded vocab size {} '.format(len(self.vocab)))
        print('loaded training data {} label {}'.format(len(self.data), self.labels.shape))
        return self.data, self.labels, self.valid_data,  self.valid_labels,\
                self.test_data, self.test_labels

    def getBowf(self):
        # bow feature
        self.bowdata = []
        for dt in self.train_data:
            sample = [0]*len(self.vocab)
            for dt_id  in dt:
                sample[dt_id] += 1
            self.bowdata.append(sample)
        self.labels = self.train_labels

        self.valid_bowdata = []
        for dt in self.valid_data:
            sample = [0]*len(self.vocab)
            for dt_id  in dt:
                sample[dt_id] += 1
            self.valid_bowdata.append(sample)

        self.test_bowdata = []
        for dt in self.test_data:
            sample = [0]*len(self.vocab)
            for dt_id  in dt:
                sample[dt_id] += 1
            self.test_bowdata.append(sample)



        print('loaded vocab size {} '.format(len(self.vocab)))
        print('loaded training data {} label {}'.format(len(self.bowdata), self.labels.shape))
        return self.bowdata, self.labels, self.valid_bowdata,  self.valid_labels,\
            self.test_bowdata, self.test_labels

    def getData(self):
        return self.train_data, self.train_labels, self.valid_data, self.valid_labels,\
            self.test_data, self.test_labels


