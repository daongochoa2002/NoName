# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from params import Params
#from dataset import Dataset

class DE_DistMult(torch.nn.Module):
    def __init__(self, config):
        super(DE_DistMult, self).__init__()
        self.config = config
        self.n_ent = config.n_ent
        self.n_rel = config.n_rel
        self.s_emb_dim = config.s_emb_dim
        self.t_emb_dim = config.t_emb_dim
        self.dropout_rate = config.dropout
        self.dropout = nn.Dropout(self.dropout_rate)
        self.lp_loss_fn = nn.CrossEntropyLoss()
        # Creating static embeddings.
        self.ent_embs      = nn.Embedding(self.n_ent, self.s_emb_dim)
        self.rel_embs      = nn.Embedding(self.n_rel, self.s_emb_dim + self.t_emb_dim)

        # Creating and initializing the temporal embeddings for the entities
        self.create_time_embedds()

        # Setting the non-linearity to be used for temporal part of the embedding
        self.time_nl = torch.sin

        nn.init.xavier_uniform_(self.ent_embs.weight)
        nn.init.xavier_uniform_(self.rel_embs.weight)


    def create_time_embedds(self):

        # frequency embeddings for the entities
        self.m_freq = nn.Embedding(self.n_ent, self.t_emb_dim)
        self.d_freq = nn.Embedding(self.n_ent, self.t_emb_dim)
        self.y_freq = nn.Embedding(self.n_ent, self.t_emb_dim)

        nn.init.xavier_uniform_(self.m_freq.weight)
        nn.init.xavier_uniform_(self.d_freq.weight)
        nn.init.xavier_uniform_(self.y_freq.weight)

        # phi embeddings for the entities
        self.m_phi = nn.Embedding(self.n_ent, self.t_emb_dim)
        self.d_phi = nn.Embedding(self.n_ent, self.t_emb_dim)
        self.y_phi = nn.Embedding(self.n_ent, self.t_emb_dim)
        nn.init.xavier_uniform_(self.m_phi.weight)
        nn.init.xavier_uniform_(self.d_phi.weight)
        nn.init.xavier_uniform_(self.y_phi.weight)

        # amplitude embeddings for the entities
        self.m_amp = nn.Embedding(self.n_ent, self.t_emb_dim)
        self.d_amp = nn.Embedding(self.n_ent, self.t_emb_dim)
        self.y_amp = nn.Embedding(self.n_ent, self.t_emb_dim)

        nn.init.xavier_uniform_(self.m_amp.weight)
        nn.init.xavier_uniform_(self.d_amp.weight)
        nn.init.xavier_uniform_(self.y_amp.weight)

    def get_time_embedd(self, entities, year, month, day):
        y = self.y_amp(entities)*self.time_nl(self.y_freq(entities)*year.unsqueeze(1) + self.y_phi(entities))
        m = self.m_amp(entities)*self.time_nl(self.m_freq(entities)*month.unsqueeze(1) + self.m_phi(entities))
        d = self.d_amp(entities)*self.time_nl(self.d_freq(entities)*day.unsqueeze(1) + self.d_phi(entities))
        return y + m + d

    def getEmbeddings(self, heads, rels, tails, years, months, days, intervals = None):

        h,r,t = self.ent_embs(heads), self.rel_embs(rels), self.ent_embs(tails)
        h_t = self.get_time_embedd(heads, years, months, days)
        t_t = self.get_time_embedd(tails, years, months, days)

        h = torch.cat([h,h_t], -1)
        t = torch.cat([t,t_t], -1)
        return h,r,t
    def forward(self, heads, rels, tails, years, months, days):
        h_embs, r_embs, t_embs = self.getEmbeddings(heads, rels, tails, years, months, days)

        scores = (h_embs * r_embs) * t_embs
        scores = F.dropout(scores, p=self.params.dropout, training=self.training)
        scores = torch.sum(scores, dim=1)

        return scores
    def train_forward(self, sub, rel, obj, year, month, day):
        bs = sub.shape[0]
        neg = torch.randint(0, self.n_ent, (bs, 500)).to(sub.device)
        ent_type = torch.cat([obj.unsqueeze(1),neg],dim = 1)
        h_embs, r_embs, t_embs = self.getEmbeddings(sub.unsqueeze(1), rel.unsqueeze(1), ent_type, year.unsqueeze(1), month.unsqueeze(1), day.unsqueeze(1))
        type_intes = self.dropout((h_embs * r_embs) * t_embs ).sum(dim= -1)

        loss_lp = self.link_prediction_loss(type_intes, ent_type, torch.zeros_like(obj))

        return loss_lp.mean()
    def test_forward(self, sub, rel, obj, year, month, day):
        bs = sub.shape[0]
        with torch.no_grad():
            ent_type =  torch.arange(self.n_ent, device=sub.device).unsqueeze(0).repeat(sub.size(0), 1)
            ent_type = torch.cat([obj.unsqueeze(1),ent_type],dim = 1)
            h_embs, r_embs, t_embs = self.getEmbeddings(sub.unsqueeze(1), rel.unsqueeze(1), ent_type, year.unsqueeze(1), month.unsqueeze(1), day.unsqueeze(1))
            scores =self.dropout((h_embs * r_embs) * t_embs ).sum(dim= -1)

        return scores
    def link_prediction_loss(self, intens, type, answers):
        loss = self.lp_loss_fn(intens, answers)
        return loss
