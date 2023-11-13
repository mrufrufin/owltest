import torch
from torch import nn

# Abdoli, S., Cardinal, P., Koerich, A.L. (2019). End-to-End Environmental Sound Classification using a 1D Convolutional Neural Network. https://arxiv.org/abs/1904.08990

class Abdoli(nn.Module):
    def __init__(self, mtype = 0, mono=True, isize = 1600, num_cl=5):
        # (idx, ipt_size) = (4, 50999), (2 o 3, 16000) , (1, 80000), (0, 1600)
        self.mtype = max(0, min(mtype,5))
        self.in_ch = 2 if mono == False else 1
        self.mono = mono
        self.num_cl = num_cl
        self.dropout = 0.25
        conv_outsz = 0
        out_ch = 128
        if self.mtype == 0:
            #1600
            # (l_in + 2 x pad - dilation x (ksize - 1) - 1)/stride + 1
            #in (n, c, isize), out = (n, c, outsize)
            self.cl1 = nn.Conv1d(self.in_ch, 16, 32, stride=2, padding=0, dilation=1)
            self.cl1_act = nn.ReLU()
            cl1out_sz = self.calc_outsize(isize, 32, stride=2)
            self.cl1_bn = nn.BatchNorm1d(16)
            self.pl1 = nn.MaxPool1d(2, stride=2, padding=0, dilation=1)
            pl1out_sz = self.calc_outsize(cl1out_sz, 2, stride=2)
            self.cl2 = nn.Conv1d(16, 32, stride=2, padding=0, dilation=1)
            self.cl2_act = nn.ReLU()
            cl2out_sz = self.calc_outsize(pl1out_sz, 16, stride=2)
            self.cl2_bn = nn.BatchNorm1d(32)
            self.pl2 = nn.MaxPool1d(2, stride=2, padding=0, dilation=1)
            pl2out_sz = self.calc_outsize(cl2out_sz, 2, stride=2)
            self.cl3 =  nn.Conv1d(32,64, 8, stride=2, padding=0, dilation=1)
            self.cl3_act = nn.ReLU()
            self.cl3_bn = nn.BatchNorm1d(64)
            cl3out_sz = self.calc_outsize(pl2out_sz, 8, stride=2)
            self.cl3_act = nn.ReLU()
            conv_outsz = cl3out_sz
            out_ch = 64
        if self.mtype == 1:
            #8000
            self.cl1 = nn.Conv1d(self.in_ch, 16, 64, stride=2, padding=0, dilation=1)
            self.cl1_act = nn.ReLU()
            cl1out_sz = self.calc_outsize(isize, 64, stride=2)
            self.cl1_bn = nn.BatchNorm1d(16)
            self.pl1 = nn.MaxPool1d(8, stride=8, padding=0, dilation=1)
            pl1out_sz - self.calc_outsize(cl1out_sz, 8, stride=8)
            self.cl2 = nn.Conv1d(16, 32, 32, stride=2, padding=0, dilation=1)
            self.cl2_act = nn.ReLU()
            cl2out_sz = self.calc_outsize(pl1out_sz, 32, stride=2)
            self.cl2_bn = nn.BatchNorm1d(32)
            self.pl2 = nn.MaxPool1d(8, stride=8, padding=0, dilation=1)
            pl2out_sz = self.calc_outsize(cl2out_sz, 8, stride=8)
            self.cl3 = nn.Conv1d(32, 64, 16, stride=2, padding=0, dilation=1)
            self.cl3_act = nn.ReLU()
            cl3out_sz = self.calc_outsize(pl2out_sz, 16, stride=2)
            self.cl3_bn = nn.BatchNorm1d(64)
            conv_outsz = cl3out_sz
            out_ch = 64
        if self.mtype == 2 or self.mtype == 3:
            # 16000 (skip 16000G)
            self.cl1 = nn.Conv1d(self.in_ch, 16, 64, stride=2, padding=0, dilation=1)
            self.cl1_act = nn.ReLU()
            cl1out_sz = self.conv_outsize(isize, 64, stride=2)
            self.cl1_bn = nn.BatchNorm1d(16)
            self.pl1 = nn.MaxPool1d(8, stride= 8, padding=0, dilation=1)
            pl1out_sz = self.conv_outsize(cl1out_sz, 8, stride=8)
            self.cl2 = nn.Conv1d(16,32, 32, stride=2, padding=0, dilation=1)
            self.cl2_act = nn.ReLU()
            cl2out_sz = self.conv_outsize(pl1out_sz, 32, stride=2)
            self.cl2_bn = nn.BatchNorm1d(32)
            self.pl2 = nn.MaxPool1d(8,stride=8, padding=0, dilation=1)
            pl2out_sz = self.conv_outsize(cl2out_sz, 8, stride=8)
            self.cl3 = nn.Conv1d(32, 64, 16, stride=2, padding=0, dilation=1)
            self.cl3_act = nn.ReLU()
            cl3out_sz = self.conv_outsize(pl2out_sz, 16, stride=2)
            self.cl3_bn = nn.BatchNorm1d(64)
            self.cl4 = nn.Conv1d(64,128, 8, stride=2, padding=0, dilation=1)
            self.cl4_act = nn.ReLU()
            cl4out_sz - self.conv_outsize(cl3out_sz, 8, stride=2)
            self.cl4_bn = nn.BatchNorm1d(64)
            conv_outsz = cl4out_sz
            out_ch = 128
        else:
            # 32000, 50999
            self.cl1 = nn.Conv1d(self.in_ch, 16, 64, stride=2)
            self.cl1_act = nn.ReLU()
            cl1out_sz = self.conv_outsz(isize, 64, stride=2)
            self.cl1_bn = nn.BatchNorm1d(16)
            self.pl1 = nn.MaxPool1d(8, stride=8)
            pl1out_sz = self.conv_outsz(cl1out_sz, 8, stride=8)
            self.cl2 = nn.Conv1d(16,32, 32, stride=2)
            self.cl2_act = nn.ReLU()
            cl2out_sz = self.conv_outsz(pl1out_sz, 32, stride=2)
            self.cl2_bn = nn.BatchNorm1d(32)
            self.pl2 = nn.MaxPool1d(8, stride=8)
            pl2out_sz = self.conv_outsz(cl2out_sz, 8, stride=8)
            self.cl3 = nn.Conv1d(32,64, 16, stride=2)
            self.cl3_act = nn.ReLU()
            cl3out_sz = self.conv_outsz(pl2out_sz, 16, stride=2)
            self.cl3_bn = nn.BatchNorm1d(64)
            self.cl4 = nn.Conv1d(64,128, 8, stride=2)
            self.cl4_act = nn.ReLU()
            cl4out_sz = self.conv_outsz(cl3out_sz, 8, stride=2)
            self.cl4_bn = nn.BatchNorm1d(128)
            self.cl5 = nn.Conv1d(128,256, 4, stride=2)
            self.cl5_act = nn.ReLU()
            cl5out_sz = self.conv_outsz(cl4out_sz, 4, stride=2)
            self.cl5_bn = nn.BatchNorm1d(256)
            self.pl3 = nn.MaxPool1d(4, stride=4)
            pl3out_sz = self.conv_outsz(cl5out_sz, 4, stride=4)
            conv_outsz = pl3out_sz
            out_ch = 256
        fc_indim = out_ch * conv_outsz
        self.flat = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(fc_indim,128,bias=True)
        self.fc1_act = nn.ReLU()
        self.fc1_drop = nn.Dropout(p=self.dropout)
        self.fc2 = nn.Linear(128, 64, bias=True)
        self.fc2_act = nn.ReLU()
        self.fc2_drop = nn.Dropout(p=self.dropout)
        self.fc3 = nn.Linear(64, self.num_cl, bias=True)
      




    def calc_outsize(self, insize, ksize, stride = 1, pad = 0, dilation = 1):
        ret =  int((insize + (2 * pad) - (dilation * (ksize - 1)) - 1)/stride) + 1
        return ret

    def forward(self, cur_ipt):

        
        out_cl1 = self.cl1(cur_ipt)
        out_clr1 = self.cl1_act(out_cl1)
        out_bn1 = self.cl1_bn(out_clr1)
        out_mp1 = self.pl1(out_bn1)
        out_cl2 = self.cl2(out_mp1)
        out_clr2 = self.cl2_act(out_cl2)
        out_bn2 = self.cl2_bn(out_clr2)
        out_mp2 = self.pl2(out_bn2)
        out_cl3 = self.cl3(out_mp2)
        out_clr3  = self.cl3_act(out_cl3)
        conv_out  = self.cl3_bn(out_clr3)
        if self.mtype >= 2:
            out_cl4 = self.cl4(conv_out)
            out_clr4 = self.cl4_act(out_cl4)
            conv_out = self.cl4_bn(out_clr4)
        if self.mtype >= 4:
            out_cl5 = self.cl5(conv_out)
            out_clr5 = self.cl5_act(out_cl5)
            out_bn5 = self.cl5_bn(out_clr5)
            conv_out = self.pl3(out_bn5)
        out_flat = self.flat(conv_out)
        out_fc1 = self.fc1(out_flat)
        out_fcr1 = self.fc1_act(out_fc1)
        out_d1 = self.fc1_drop(out_fcr1)
        out_fc2 = self.fc2(out_d1)
        out_fcr2 = self.fc2_act(out_fc2)
        out_d2 = self.fc2_drop(out_fcr2)
        net_out = self.fc3(out_d2)
        return net_out


