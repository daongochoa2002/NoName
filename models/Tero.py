import torch
import numpy as np
import torch.nn as nn
from torch.nn.init import xavier_normal_
from torch.nn import functional as F
from torch.autograd import Variable
from numpy.random import RandomState

class Tero(torch.nn.Module):
    def __init__(self, config):
        super(Tero, self).__init__()
        self.config = config
        self.n_ent = config.n_ent
        self.n_rel = config.n_rel
        self.s_emb_dim = config.s_emb_dim
        self.t_emb_dim = config.t_emb_dim
        self.d_model = config.d_model
        self.dropout_rate = config.dropout

        self.emb_E_real = torch.nn.Embedding(self.n_ent, self.d_model, padding_idx=0)
        self.emb_E_img = torch.nn.Embedding(self.n_ent, self.d_model, padding_idx=0)
        self.emb_R_real = torch.nn.Embedding(self.n_rel, self.d_model, padding_idx=0)
        self.emb_R_img = torch.nn.Embedding(self.n_rel, self.d_model, padding_idx=0)
        self.w1 = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.d_model))).float(), requires_grad=True)
        self.w2 = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.d_model))).float(), requires_grad=True)
        # Initialization
        r = 6 / np.sqrt(self.d_model)
        self.emb_E_real.weight.data.uniform_(-r, r)
        self.emb_E_img.weight.data.uniform_(-r, r)
        self.emb_R_real.weight.data.uniform_(-r, r)
        self.emb_R_img.weight.data.uniform_(-r, r)
        # self.emb_T_img.weight.data.uniform_(-r, r)
        self.lp_loss_fn = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def train_forward(self, sub, rel, obj, year, month, day):
        bs = sub.shape[0]
        neg = torch.randint(0, self.n_ent, (bs, 500)).to(sub.device)
        ent_type = torch.cat([obj.unsqueeze(1),neg],dim = 1)
        #pi = 3.14159265358979323846
        #d_img = torch.sin(self.emb_Time(day).view(-1, self.embedding_dim))#/(6 / np.sqrt(self.embedding_dim)/pi))
        d_img = torch.sin(self.w1.view(1, -1) * day.unsqueeze(1))
        #d_real = torch.cos(self.emb_Time(day).view(-1, self.embedding_dim))#/(6 / np.sqrt(self.embedding_dim)/pi))
        d_real = torch.cos(self.w2.view(1, -1) * day.unsqueeze(1))
        h_real = self.emb_E_real(sub) *d_real - self.emb_E_img(sub) *d_img
        t_real = self.emb_E_real(ent_type) * d_real.unsqueeze(1) - self.emb_E_img(ent_type) * d_img.unsqueeze(1)
        r_real = self.emb_R_real(rel)
        h_img = self.emb_E_real(sub) *d_img + self.emb_E_img(sub) *d_real
        t_img = self.emb_E_real(ent_type) *d_img.unsqueeze(1) + self.emb_E_img(ent_type) *d_real.unsqueeze(1)
        r_img = self.emb_R_img(rel)
        #if self.L == 'L1':
            #out_real = torch.sum(torch.abs(h_real + r_real - t_real), 1)
        out_real = torch.abs((h_real + r_real).unsqueeze(1) - t_real).sum(dim=-1)
        out_img = torch.abs((h_img + r_img).unsqueeze(1) + t_img).sum(dim=-1)
        #out_img = torch.sum(torch.abs(h_img + r_img + t_img), 1)
        type_intes = out_real + out_img
        #else:
        #    out_real = torch.sum((h_real + r_real + day - t_real) ** 2, 1)
        #    out_img = torch.sum((h_img + r_img + day + t_real) ** 2, 1)
        #    out = torch.sqrt(out_img + out_real)

        loss_lp = self.link_prediction_loss(type_intes, ent_type, torch.zeros_like(obj))
        return loss_lp.mean()

    def test_forward(self, sub, rel, obj, year, month, day):
        bs = sub.shape[0]
        with torch.no_grad():
            ent_type =  torch.arange(self.n_ent, device=sub.device).unsqueeze(0).repeat(sub.size(0), 1)
            d_img = torch.sin(self.w1.view(1, -1) * day.unsqueeze(1))
            d_real = torch.cos(self.w2.view(1, -1) * day.unsqueeze(1))
            h_real = self.emb_E_real(sub) *d_real - self.emb_E_img(sub) *d_img
            t_real = self.emb_E_real(ent_type) * d_real.unsqueeze(1) - self.emb_E_img(ent_type) * d_img.unsqueeze(1)
            r_real = self.emb_R_real(rel)
            h_img = self.emb_E_real(sub) *d_img + self.emb_E_img(sub) *d_real
            t_img = self.emb_E_real(ent_type) *d_img.unsqueeze(1) + self.emb_E_img(ent_type) *d_real.unsqueeze(1)
            r_img = self.emb_R_img(rel)
            out_real = torch.abs((h_real + r_real).unsqueeze(1) - t_real).sum(dim=-1)
            out_img = torch.abs((h_img + r_img).unsqueeze(1) + t_img).sum(dim=-1)
            scores = out_real + out_img
        return scores
    def link_prediction_loss(self, intens, type, answers):
        loss = self.lp_loss_fn(intens, answers)
        return loss
