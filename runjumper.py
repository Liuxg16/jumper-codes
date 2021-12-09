# coding=utf-8
import sys
import time
import os
import json
import random
import numpy as np
import math
from os.path import dirname, join
import cPickle as pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import nntools.utils as utils
import nntools.nntools as nntools
from nntools.utils import say,mylog,MyRecorder
import  options
from dataio.preprocess import *
from smodel import *

def main(args,kwargs):

    assert args.model != None
    assert args.prate != None
    assert args.drate != None

    kwargs['model'] = args.model
    kwargs['prate'] = args.prate
    kwargs['n_hids'] = args.n_hids
    kwargs['n_filter'] = args.n_filter
    kwargs['embsize'] = args.embsize
    kwargs['drate'] = args.drate
    kwargs['mark'] = args.mark
    kwargs['optim'] = args.optim
    kwargs['feed_flag'] = not args.nofeed
    kwargs['task'] = args.task
    kwargs['parallel'] = args.parallel
    kwargs['n_sample'] = args.n_sample
    kwargs['batch_size'] = args.batch_size
    kwargs['lr'] = args.lr
    seed  = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_source, valid_source, test_source, vocab = gen_data_task(args.task)
    train_labels = train_source[1]
    # print 'unk id:{}'.format(vocab['unk'])
    # print 'size of vocab'
    # print len(vocab)

    tmodel = TrainingModel(model_name= Jumper_std,vocab = vocab,category_list=(train_labels.max(axis=0)+1).tolist(), **kwargs)

    if args.save is  not None: 
        with open(args.save, 'rb') as f:
            tmodel.model.load_state_dict(torch.load(f))
            # tmodel.model = torch.load(f)
 

    # list (n,matrix_tensor)
    if not args.eval:
        print('Load the model.....')
        print('Test performance:')
        # tmodel.loop_train(train_source,valid_source, test_source)
        test_data = tmodel.create_buckets_batches(test_source[0], test_source[1], vocab['<mask>'])
        ans = tmodel.validation(test_data)
        print ans[1]
    else:
        tmodel.eval = True
        # tmodel.show_error([test_data,test_labels],vocab)
        test_data = tmodel.create_buckets_batches(test_source[0], test_source[1], vocab['<mask>'])

        tmodel.evaluate(vocab)
        # print tmodel.oitest_nonjump_error(test_data,vocab)

        # print tmodel.test_error(test_data,vocab)
        #print tmodel.validation(test_data)
        # print tmodel.jump_annotation(test_data,vocab)
        # print tmodel.test_nonjump_error(test_data,vocab)

class TrainingModel(object):

    def __init__(self, model_name,vocab, category_list, **kwargs):
        self.vocab = vocab
        self.lr = kwargs['lr']
        self.eval = False
        ntokens = len(vocab)
        n_task = len(category_list)
        self.kwargs = kwargs
        self.kwargs['category_list'] = category_list
        ###############################################################################
        # Build the model
        ###############################################################################
        self.device_ids = self.kwargs['device']
        self.kwargs['mask_id'] = self.vocab['<mask>']
        self.model_pre = model_name( len(self.vocab), **self.kwargs)
        if self.kwargs['parallel'] :
            self.model = nn.DataParallel(self.model_pre, self.device_ids)
        else:
            self.model = self.model_pre
        self.model.cuda()
        if self.kwargs['optim']== 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr = self.lr)
        elif self.kwargs['optim'] == 'adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters(), lr = self.lr)
        self.l1loss = nn.L1Loss()
        self.l2loss = nn.MSELoss()

    def validation(self, data_source):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        total_loss = 0.0
        acc = 0
        edge_acc = 0
        last_edge_acc = 0
        reward = 0
        n_batches = len(data_source[0])
        acc_vec = []
        num_samples = 0
        for i in range(n_batches):
            data, targets = self.get_batch(data_source, i, evaluation=True)
            batch_size = data.size()[0]
            loss_t, acc_t, reward_t, lensss, result = self.model(data, targets)
            loss_t = torch.mean(loss_t)
            acc_t = torch.mean(acc_t)
            reward_t = torch.mean(reward_t)
            
            acc += acc_t.data[0]*batch_size
            reward += reward_t.data[0]*batch_size

            total_loss += loss_t.data[0]*batch_size
            acc_vec.append(result)
            num_samples += batch_size
        # acc_all = torch.cat(acc_vec,0)
        acc_all = acc_vec
        return total_loss /num_samples,acc/num_samples, reward/num_samples, reward/num_samples

    def evaluate(self, word2id):
        self.model.testjump = True
        self.model.eval()
        text = '去年10月21日和同事去苏州出差，同事骑电瓶车带我撞到公路护栏上，导致小腿粉碎骨折，现在已经做完工伤认定，11月份做完伤残鉴定，由于去年9月21日刚到公司，10月就出事，未转正，现在不想去公司上班了，公司大约能赔多少钱，劳动能力鉴定结果已经出来为9级，下一步去哪个部门申请进行索赔？'.decode('UTF-8') 
        text_sents = text.split('，'.decode('UTF-8'))
        text2ids = [[self.gen_word2id(word2id,word) for word in sent] for sent in text_sents]
        print('raw text:')
        for sent in text_sents:
            print sent
        
        mask_id = 0
        max_l =  max(len(sent) for sent in text2ids)
        testids = [sent+[mask_id]*(max_l-len(sent)) for sent in text2ids]
        data_ts = torch.LongTensor(testids)
        data =Variable( data_ts.unsqueeze(0)).cuda()
        target_ts = Variable(torch.LongTensor([1]).view(1,1)).cuda()

        result = self.model(data,target_ts)
        preds = result[-1]-1
        print('decision process: (\'-1\' means not jumping)')
        print (preds).data[0].cpu().tolist()

    def training(self,train_data, test_data):
        # Turn on training mode which enables dropout.
        total_loss = 0
        total_acc = 0
        total_reward = 0
        start_time = time.time()
        n_batches = len(train_data[0])
        order = [i for i in xrange(n_batches)]
        # shuffle the order
        random.shuffle(order)
        for i in range(n_batches):
            self.model.train()
            self.optimizer.zero_grad()
            data, targets1  = self.get_batch(train_data, order[i])

            batch_size = data.size()[0]
            self.model.zero_grad() # important
            loss_t, acc_t, reward_t ,lens, res= self.model(data, targets1)
            params = list(self.model.parameters())

            loss_factor = 1e-6
            reg_loss = torch.sum(loss_t)
            acc_t = torch.mean(acc_t)
            reward_t = torch.mean(reward_t)
            # regularization
            #for param in self.model.parameters():
            #    reg_loss += self.l1loss(param,nntools.repackage_var(param,False)*0)* loss_factor 
            #    reg_loss += self.l2loss(param,nntools.repackage_var(param,False)*0)* 10 * loss_factor 
                
            reg_loss.backward()
            accuracy = acc_t.data[0]
            reward = reward_t.data[0]

            ## update parameters
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            # torch.nn.utils.clip_grad_norm(self.model.parameters(), kwargs['clip_grad'])
            self.optimizer.step()
            # for p in self.model.parameters():
            #     p.data.add_(-self.lr, p.grad.data)
            for module_i in self.model_pre.module_list:
                if (module_i.weight.norm() > self.kwargs['MAX_NORM']).data.all():
                    module_i.weight.data =( module_i.weight.data *  self.kwargs['MAX_NORM']) /module_i.weight.data.norm()

            total_loss += reg_loss.data[0]
            total_acc += accuracy
            total_reward += reward
            if (i+1) % args.interval == 0:
                cur_loss = total_loss / args.interval
                cur_acc = total_acc / args.interval
                cur_reward = total_reward / args.interval
                elapsed = time.time() - start_time
                self.logg.logging('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.3f} | reward {:5.2} | acc {:5.3f} |'.format(
                    self.epoch, i+1, n_batches, self.lr, elapsed * 1000 / args.interval,\
                            cur_loss, cur_reward, cur_acc))
                total_loss = 0
                total_acc = 0
                total_reward = 0
                start_time = time.time()
                test_acc,jump_abu,jump_acc, zero_jump_acc, final_jump_acc, impulacc = self.test_error(test_data,self.vocab)
                self.logg.logging('| {} batches | acc:{} | jump acc{}| zero acc {}| final acc {}'.\
                        format(self.epoch*n_batches+i-n_batches, test_acc, jump_abu, zero_jump_acc,\
                        final_jump_acc))
                acc_dict = {'test_acc':test_acc}
                jump_dict = {'test_jump_acc':jump_abu}
                zjump_dict = {'zero_jump_acc':zero_jump_acc}
                final_acc_dict = {'final_jump_acc':final_jump_acc}
                self.recorder.add(acc_dict)
                self.recorder.add(jump_dict)
                self.recorder.add(zjump_dict)
                self.recorder.add(final_acc_dict)

    def get_batch(self, source, i, evaluation=False):
        data_ts = torch.LongTensor(source[0][i])
        target_ts = torch.LongTensor(source[1][i])
        if args.cuda:
            data_ts = data_ts.cuda()
            target_ts = target_ts.cuda()

        data = Variable(data_ts, volatile=evaluation)
        target = Variable(target_ts)
        return data, target

    def gen_batches(self,data_tuple):
        # suitable for abitrary length of data_tuple
        batches = [[] for i in xrange(len(data_tuple))]
        for i in xrange(len(data_tuple)-1):
            assert len(data_tuple[i]) == len(data_tuple[i+1])
        bs = self.args.batch_size
        # for i in xrange(int(np.ceil(len(data_tuple[0]) / float(bs)))):
        for i in xrange(int(np.floor(len(data_tuple[0]) / float(bs)))):  # delete the last batch

            for j in xrange(len(data_tuple)):
                batches[j].append(data_tuple[j][i * bs:i * bs + bs])

        return batches

    def create_buckets_batches(self,lstx,lsty,mask_id=0):
        assert min(len(x) for x in lstx) > 0
        batches_x, batches_y = [], []
        assert len(lstx) == len(lsty)
        bs = args.batch_size
         
        for i in xrange(int(np.ceil(len(lstx) / float(bs)))):
            max_subp_len = 0
            bucket_list_x = lstx[i*bs:i*bs+bs]            
            mx_num_subsent = max(len(x) for x in bucket_list_x)
            for subparam_inputs in bucket_list_x:
                max_subp_len1 = max([len(subp) for subp in subparam_inputs])+6
                if max_subp_len1 > max_subp_len:
                    max_subp_len = max_subp_len1
            bucket_docs = [[[mask_id]*3 + subp + [mask_id] * (3+max_subp_len - len(subp)) for subp in doc] +
                    [[mask_id] * (max_subp_len+6)] * (mx_num_subsent - len(doc)) for doc in bucket_list_x]
            bucket_docs_np = np.array(bucket_docs)
            batches_x.append(bucket_docs_np)
            bucket_list_y = np.array(lsty[i*bs:i*bs+bs])
            batches_y.append(bucket_list_y)
        return batches_x,batches_y

    def gen_idx2word(self, vocab, id):
        idx2word = dict((word,idx) for idx,word in vocab.items())
        if idx2word.has_key(id):
            return idx2word[id]
        else:
            return 'unk'

    def gen_word2id(self, vocab, word):
        if vocab.has_key(word):
            return vocab[word]
        else:
            return vocab['unk']

def gen_data_task(task_name, bow = False):
    if task_name=='oi-level':
        kwargs['num_categories'] =  12
        dataset = Dataset_oi('data/OI-dataset', kwargs['rnd'], 'level')
        vocab = dataset.getVocab()
        train_data,train_labels, valid_data, valid_labels,\
                 test_data, test_labels = dataset.get3Ddata()
        vocab = dataset.getVocab()
        train_labels = train_labels+1
        valid_labels = valid_labels +1
        test_labels = test_labels +1
        print min(train_labels)
        print max(train_labels)

    return [train_data, train_labels],[valid_data, valid_labels], [test_data,\
            test_labels], vocab

if __name__ == "__main__":

    kwargs = {
        "rnd": 123,
        "max_epochs":10000,
        "max_unchange_epochs":400,
        "MAX_NORM":3,
        "n_repeat":5,
        "size_filter":5,
        "use_emb": True,
        "n_hids":20,
        "n_filter":50,
        "device":[0,1],
        "embsize": 300,
        "zero_reward": 0.05,
        "yita": 0.9,
        "feed_flag": True,
        "is_print": False,
        'add_mean_reward':True,
        'use_sm':0,
        'clip_grad':0.5
    }

    args = options.load_arguments()
    if args.mode==0:
        main(args,kwargs)
    elif args.mode==1:
        args.eval = True



