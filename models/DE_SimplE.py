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

class DE_SimplE(torch.nn.Module):
    def __init__(self, config):
        super(DE_SimplE, self).__init__()
        self.config = config
        self.n_ent = config.n_ent
        self.n_rel = config.n_rel
        self.s_emb_dim = config.s_emb_dim
        self.t_emb_dim = config.t_emb_dim
        self.dropout_rate = config.dropout


        self.ent_embs_h = nn.Embedding(self.n_ent, self.s_emb_dim)
        self.ent_embs_t = nn.Embedding(self.n_ent, self.s_emb_dim)
        self.rel_embs_f = nn.Embedding(self.n_rel, self.s_emb_dim + self.t_emb_dim)
        self.rel_embs_i = nn.Embedding(self.n_rel, self.s_emb_dim + self.t_emb_dim)

        self.create_time_embedds()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.time_nl = torch.sin
        self.lp_loss_fn = nn.CrossEntropyLoss()


        nn.init.xavier_uniform_(self.ent_embs_h.weight)
        nn.init.xavier_uniform_(self.ent_embs_t.weight)
        nn.init.xavier_uniform_(self.rel_embs_f.weight)
        nn.init.xavier_uniform_(self.rel_embs_i.weight)

    def create_time_embedds(self):

        # frequency embeddings for the entities
        self.m_freq_h = nn.Embedding(self.n_ent, self.t_emb_dim)
        self.m_freq_t = nn.Embedding(self.n_ent, self.t_emb_dim)
        self.d_freq_h = nn.Embedding(self.n_ent, self.t_emb_dim)
        self.d_freq_t = nn.Embedding(self.n_ent, self.t_emb_dim)
        self.y_freq_h = nn.Embedding(self.n_ent, self.t_emb_dim)
        self.y_freq_t = nn.Embedding(self.n_ent, self.t_emb_dim)

        # phi embeddings for the entities
        self.m_phi_h = nn.Embedding(self.n_ent, self.t_emb_dim)
        self.m_phi_t = nn.Embedding(self.n_ent, self.t_emb_dim)
        self.d_phi_h = nn.Embedding(self.n_ent, self.t_emb_dim)
        self.d_phi_t = nn.Embedding(self.n_ent, self.t_emb_dim)
        self.y_phi_h = nn.Embedding(self.n_ent, self.t_emb_dim)
        self.y_phi_t = nn.Embedding(self.n_ent, self.t_emb_dim)

        # frequency embeddings for the entities
        self.m_amps_h = nn.Embedding(self.n_ent, self.t_emb_dim)
        self.m_amps_t = nn.Embedding(self.n_ent, self.t_emb_dim)
        self.d_amps_h = nn.Embedding(self.n_ent, self.t_emb_dim)
        self.d_amps_t = nn.Embedding(self.n_ent, self.t_emb_dim)
        self.y_amps_h = nn.Embedding(self.n_ent, self.t_emb_dim)
        self.y_amps_t = nn.Embedding(self.n_ent, self.t_emb_dim)

        nn.init.xavier_uniform_(self.m_freq_h.weight)
        nn.init.xavier_uniform_(self.d_freq_h.weight)
        nn.init.xavier_uniform_(self.y_freq_h.weight)
        nn.init.xavier_uniform_(self.m_freq_t.weight)
        nn.init.xavier_uniform_(self.d_freq_t.weight)
        nn.init.xavier_uniform_(self.y_freq_t.weight)

        nn.init.xavier_uniform_(self.m_phi_h.weight)
        nn.init.xavier_uniform_(self.d_phi_h.weight)
        nn.init.xavier_uniform_(self.y_phi_h.weight)
        nn.init.xavier_uniform_(self.m_phi_t.weight)
        nn.init.xavier_uniform_(self.d_phi_t.weight)
        nn.init.xavier_uniform_(self.y_phi_t.weight)

        nn.init.xavier_uniform_(self.m_amps_h.weight)
        nn.init.xavier_uniform_(self.d_amps_h.weight)
        nn.init.xavier_uniform_(self.y_amps_h.weight)
        nn.init.xavier_uniform_(self.m_amps_t.weight)
        nn.init.xavier_uniform_(self.d_amps_t.weight)
        nn.init.xavier_uniform_(self.y_amps_t.weight)

    def get_time_embedd(self, entities, years, months, days, h_or_t):
        if h_or_t == "head":
            emb  = self.y_amps_h(entities) * self.time_nl(self.y_freq_h(entities) * years.unsqueeze(1)  + self.y_phi_h(entities))
            emb += self.m_amps_h(entities) * self.time_nl(self.m_freq_h(entities) * months.unsqueeze(1) + self.m_phi_h(entities))
            emb += self.d_amps_h(entities) * self.time_nl(self.d_freq_h(entities) * days.unsqueeze(1)   + self.d_phi_h(entities))
        else:
            emb  = self.y_amps_t(entities) * self.time_nl(self.y_freq_t(entities) * years.unsqueeze(1)  + self.y_phi_t(entities))
            emb += self.m_amps_t(entities) * self.time_nl(self.m_freq_t(entities) * months.unsqueeze(1) + self.m_phi_t(entities))
            emb += self.d_amps_t(entities) * self.time_nl(self.d_freq_t(entities) * days.unsqueeze(1)   + self.d_phi_t(entities))

        return emb

    def getEmbeddings(self, heads, rels, tails, years, months, days, intervals = None):
        #years = years.view(-1,1)
        #months = months.view(-1,1)
        #days = days.view(-1,1)
        h_embs1 = self.ent_embs_h(heads)
        r_embs1 = self.rel_embs_f(rels)
        t_embs1 = self.ent_embs_t(tails)
        h_embs2 = self.ent_embs_h(tails)
        r_embs2 = self.rel_embs_i(rels)
        t_embs2 = self.ent_embs_t(heads)

        h_embs1 = torch.cat([h_embs1, self.get_time_embedd(heads, years, months, days, "head")], -1)
        t_embs1 = torch.cat([t_embs1, self.get_time_embedd(tails, years, months, days, "tail")], -1)
        h_embs2 = torch.cat([h_embs2, self.get_time_embedd(tails, years, months, days, "head")], -1)
        t_embs2 = torch.cat([t_embs2, self.get_time_embedd(heads, years, months, days, "tail")], -1)

        return h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2

    def train_forward(self, sub, rel, obj, year, month, day):
        bs = sub.shape[0]
        neg = torch.randint(0, self.n_ent, (bs, 500)).to(sub.device)
        ent_type = torch.cat([obj.unsqueeze(1),neg],dim = 1)
        h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 = self.getEmbeddings(sub.unsqueeze(1), rel.unsqueeze(1), ent_type, year.unsqueeze(1), month.unsqueeze(1), day.unsqueeze(1))

        type_intes = self.dropout(((h_embs1 * r_embs1) * t_embs1 + (h_embs2 * r_embs2) * t_embs2) / 2.0).sum(dim= -1)

        loss_lp = self.link_prediction_loss(type_intes, ent_type, torch.zeros_like(obj))

        return loss_lp.mean()
    def test_forward(self, sub, rel, obj, year, month, day):
        bs = sub.shape[0]
        with torch.no_grad():
            ent_type =  torch.arange(self.n_ent, device=sub.device).unsqueeze(0).repeat(sub.size(0), 1)
            h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 = self.getEmbeddings(sub.unsqueeze(1), rel.unsqueeze(1), ent_type, year.unsqueeze(1), month.unsqueeze(1), day.unsqueeze(1))
            scores = ((h_embs1 * r_embs1) * t_embs1 + (h_embs2 * r_embs2) * t_embs2) / 2.0
            scores = self.dropout(scores).sum(dim= -1)

        return scores
    def link_prediction_loss(self, intens, type, answers):
        loss = self.lp_loss_fn(intens, answers)
        return loss
