import torch.nn as nn
from torch.autograd import Variable
import torch
import nntools.nntools as nntools
import torch.nn.functional as F


class Jumper_std(nn.Module):

    def __init__(self, vocab_size, **kwargs):
        '''
        :param hid_size:
        :param category_list:
        :param n_task:
        :param embsize:
        :param smi_size:  Symbolic memory input size
        :param vocab_size:
        :param x:
        :param y:
        :param kargs:
        '''
        super(Jumper_std, self).__init__()
        self.mask_id =  kwargs['mask_id'] 
        self.feed_flag = kwargs['feed_flag']
        self.n_d_hids = kwargs['n_hids']
        self.embsize = kwargs['embsize']
        self.num_categories = kwargs['num_categories']
        self.prate = kwargs['prate']
        self.drate = kwargs['drate']
        rnd = kwargs['rnd']
        self.n_filter = kwargs['n_filter']
        self.size_filter = kwargs['size_filter']

        self.embedding_layer = nn.Embedding(vocab_size, self.embsize)
        self.cnn = nntools.KimCNN(self.size_filter,self.n_filter,self.embsize)

        self.n_repeat = kwargs['n_repeat']
        hidden_size = self.n_d_hids
        self.hidden_size = hidden_size
        input_size = self.cnn.n_out
        self.input_size = input_size

        self.xdz = nn.Linear(input_size, hidden_size)
        self.ddz = nn.Linear(hidden_size, hidden_size)

        self.xdr = nn.Linear(input_size, hidden_size)
        self.ddr = nn.Linear(hidden_size, hidden_size)

        self.xd = nn.Linear(input_size, hidden_size)
        self.dd = nn.Linear(hidden_size, hidden_size)

        self.xs = nn.Linear(input_size, self.num_categories)
        self.ds = nn.Linear(hidden_size, self.num_categories)
        # self.fc = nn.Linear(hidden_size, self.num_categories)
        self.module_list = nn.ModuleList([self.xs, self.ds])

        self.drop = nn.Dropout(self.drate)
        self.l1loss = nn.L1Loss()
        self.l2loss = nn.MSELoss()


    def forward(self, x, y):
        '''
        inputs:  (b_s, l, k)
        targets: (b_s, n_task)
        '''
        # 3 dimensions -> 2 dimensions
        batches_size = x.size()[0]  # (batch,l,K)
        l = x.size()[1]  # the number of sub paragragh
        k = x.size()[2]
        x_reshape = x.view(batches_size,l*k)  # (batch_size,l*k)

        mask_x = 1- torch.eq(x, self.mask_id).float()  # #(batch_size, l,k)
        mask_x_2d = 1- torch.eq(x_reshape, self.mask_id).float()  # #(batch_size, l,k)
        
        # length* batch * embsize
        word_vectors = self.embedding_layer(x_reshape)  # #(batch_size,l*k,embsize)
        feed_in = word_vectors.view(batches_size,l,k,-1)
     
        inputs = feed_in
        targets = y
        inputs_mask = mask_x

        inputs_r = inputs.repeat(self.n_repeat,1,1,1) # (b_s*5,l,k,emb_size)
        targets_r = targets.repeat(self.n_repeat,1) #

        shape = inputs_r.size()
        self.n_sent = shape[1]
        self.b_s = shape[0]
        self.n_subsent = shape[2]
        steps = self.n_sent
        self.ones = Variable(torch.zeros(self.b_s, 1).cuda())

        # for sampling
        self.switch_m = Variable(torch.FloatTensor(self.b_s, 2).fill_(self.prate)).cuda()
        self.switch_m[:,1] = 1- self.prate
        self.action_m = Variable(torch.FloatTensor(self.b_s,self.num_categories).fill_(1.0/self.num_categories)).cuda()

        if inputs_mask is None:
            inputs_mask = Variable(torch.ones(inputs.size()[:-1]).cuda())
        else:
            inputs_mask = inputs_mask.repeat(self.n_repeat, 1,1)
        

        st = Variable(torch.zeros(self.b_s, 1).cuda()).long()
        mask = Variable(torch.ones(self.b_s, 1).long().cuda())
        dt = Variable(torch.zeros(self.b_s, self.hidden_size).cuda())
        pi_matrix = Variable(torch.zeros(self.b_s, steps).cuda())
        mask_matrix = Variable(torch.zeros(self.b_s, steps).cuda())
        st_matrix = Variable(torch.zeros(self.b_s, steps).cuda())

        for i in range(steps):
            input = inputs_r[:,i,:,:] # (b_s,k,emsize)
            input_mask = inputs_mask[:,i,:]#(b_s,k)
            pi, dt, st, mask= self.step(input, input_mask, dt, st, mask)
            pi_matrix[:,i] = pi 
            mask_matrix[:,i] = mask
            st_matrix[:,i] = st
        
        acc = torch.eq(st_matrix[:,steps-1:steps].long(), targets_r).float() #(b_s,n_task)
        reward = acc
        returns = 2*reward-1
        mean_returns_pre = torch.mean(returns.view(-1,self.b_s/self.n_repeat,1),0).view(-1,1)
        mean_returns = mean_returns_pre.repeat(self.n_repeat, 1)
        real_returns = nn.ReLU()(returns-mean_returns)

        log_pi = torch.log(pi_matrix.clamp(1e-6,1)) * mask_matrix
        sum_pi = torch.mean(log_pi,1).view(-1,1)
        # sum_pi = torch.sum(log_pi,1).view(-1,1)
        rl_cost = - (sum_pi) * real_returns

        # slot0 =  mask_matrix * torch.eq(st_matrix, 0).float()
        # backup0 = self.get_backup(slot0, 0.1, 0.1)  # bs,step
        # backup0_reshape = backup0.view(-1,self.b_s/self.n_repeat, steps)
        # backup0_mean = torch.mean(backup0_reshape,0).repeat(self.n_repeat,1)
        # backup0_real = backup0 - backup0_mean
        # log_pi_0 =  torch.log(pi_matrix.clamp(1e-6,1)) * backup0_real
        # loss0  = - torch.sum(log_pi_0)

        # loss =  torch.sum(rl_cost)
        loss =  torch.sum(rl_cost)
        self.st_matrix = st_matrix

        # regularization
        # loss_factor = 1e-6
        # for param in self.parameters():
        #     loss += self.l1loss(param,nntools.repackage_var(param,False)*0)* loss_factor 
        #     loss += self.l2loss(param,nntools.repackage_var(param,False)*0)* 10 * loss_factor 

        self.loss = loss
        self.accuracy = torch.mean(acc)
        self.accuracy_vec = torch.mean(acc,1)
        self.reward = torch.mean(reward)
        # print acc.size()
        # return self.loss, self.accuracy, self.reward, st_matrix # (b_s, len, n_task)
        return self.loss, self.accuracy, self.reward, torch.sum(mask_matrix,1),st_matrix # (b_s, len, n_task)
        # return self.loss, self.accuracy, self.reward, acc.view(-1,1) # (b_s, len, n_task)

    def step(self, inp_rnn, input_mask_rnn,  d_tm1, s_tm1_pre, mask_tm1):
        '''
        input_ids: (b_s,k)
        inp_rnn: (b_s, k, emb_size)
        input_mask_rnn: ( b_s, k)
        tag_target:(b_s,k)
        '''

		# onehot for symbolic memory
        cnn_f = self.cnn(inp_rnn) #(b_s,k,embsize) -> (b_s,250)
        # (b_s,k, hids_rnn)  
        x_content = cnn_f
        z_dt = nn.Sigmoid()(self.xdz(x_content)+ self.ddz(d_tm1))
        r_dt = nn.Sigmoid()(self.xdr(x_content)+ self.ddr(d_tm1) )
        can_dt = nn.Tanh()(self.xd(x_content) + r_dt* self.dd(d_tm1))
        dt = z_dt * can_dt + (1-z_dt)* d_tm1
        # feed_fc = (self.xs(x_content)+self.ds(dt))
        # # fc1_feed = self.drop(feed_fc)
        # energe_s = nn.Softmax()(feed_fc) # (b_s, n_sinter)
        x_content = self.drop(x_content)
        dtd = self.drop(dt)
        energe_s = nn.Softmax(1)(self.xs(x_content)+self.ds(dtd)) # (b_s, n_sinter)

        if self.training:
            action_exploit = energe_s.multinomial()  # as edge_predict
            explorate_flag = self.switch_m.multinomial()  # as edge_predict
            action_explorate = self.action_m.multinomial()
            action = nntools.repackage_var(explorate_flag*action_exploit +\
                    (1-explorate_flag.float().float()).long() * action_explorate)
        else:
            values,action = torch.max(energe_s,1)
         
        s_t = action.view(-1,1)
        pi = torch.gather(energe_s, 1, s_t) # (b_s,1), \pi for i,j
        
        valid_mask = torch.eq(s_tm1_pre,0).long()  *(torch.gt(torch.sum(input_mask_rnn,1),0).view(-1,1).long())
        valid_mask = valid_mask * mask_tm1
        s_t_valid = s_t *valid_mask + s_tm1_pre
        
        return pi, dt, s_t_valid, valid_mask

    def set_prate(self, prate):
        self.slstm.prate = prate
    
    def set_objective(self):
        self.object_flag = True

    def get_backup(self, slot, yita, im_reward):
        steps = slot.size()[1]
        backup0 = Variable(torch.zeros(slot.size()).cuda())
        tmp = Variable(torch.zeros(slot.size()[0],1).cuda())
        for i in xrange(steps-1,-1,-1):
            tp = torch.gt(slot[:,i:i+1],0).float() * im_reward +yita*tmp
            backup0[:,i:i+1] = tp
            tmp = tp
        return  backup0

