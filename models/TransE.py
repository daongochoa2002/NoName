import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class TransE(nn.Module):
    def __init__(self, config, eps=0.2, noise_scale = 0.1, noise_min = 0.0001, noise_max = 0.02, steps = 5, beta_fixed=True):
        super(TransE, self).__init__()
        self.config = config
        self.n_ent = config.n_ent
        self.n_rel = config.n_rel
        self.d_model = config.d_model
        self.dropout_rate = config.dropout

        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-6)
        self.encoder = ConvTransE(self.n_ent, self.n_rel,self.d_model, self.dropout_rate, self.dropout_rate, self.dropout_rate)

        self.lp_loss_fn = nn.CrossEntropyLoss()

    def get_time_embedd(self, entities, year, month, day):
        y = self.y_amp(entities) * torch.sin(self.y_freq(entities)*year.unsqueeze(1) + self.y_phi(entities))
        m = self.m_amp(entities) * torch.sin(self.m_freq(entities)*month.unsqueeze(1) + self.m_phi(entities))
        d = self.d_amp(entities) * torch.sin(self.d_freq(entities)*day.unsqueeze(1) + self.d_phi(entities))
        return y + m + d

    def get_betas(self):
        start = self.noise_scale * self.noise_min
        end = self.noise_scale * self.noise_max
        variance = np.linspace(start, end, self.steps, dtype=np.float64)
        alpha_bar = 1 - variance
        betas = []
        betas.append(1 - alpha_bar[0])
        for i in range(1, self.steps):
          betas.append(min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999))
        return np.array(betas)
    def forward(self, query_entities, query_relations, obj_entities):
        bs = query_entities.size(0)
        query_rel_embeds = self.encoder.get_rel_embedding(query_relations)
        query_ent_embeds = self.encoder.get_ent_embedding(query_entities)
        obj_embeds = self.encoder.get_ent_embedding(obj_entities)
        return query_ent_embeds, query_rel_embeds, obj_embeds


    def train_forward(self, s_ent, relation, o_ent, time):
        query_ent_embeds, query_rel_embeds, obj_embeds = self.forward(s_ent, relation, o_ent)
        bs = query_ent_embeds.size(0)

        condition_emb = query_ent_embeds + query_rel_embeds
        neg = torch.randint(0, self.n_ent, (bs, 100)).to(condition_emb.device)
        ent_embeds = self.encoder.get_ent_embedding(neg)
        ent_embeds = torch.cat([obj_embeds.unsqueeze(1),ent_embeds],dim = 1)
        ent_type = torch.cat([o_ent.unsqueeze(1),neg],dim = 1)
        type_intes = -torch.norm(condition_emb.unsqueeze(1) - ent_embeds, dim = -1)
        loss_lp = self.link_prediction_loss(type_intes, ent_type, torch.zeros_like(o_ent))
        return loss_lp.mean()


    def test_forward(self, s_ent, relation, o_ent, time, local_weight=1.):
        query_ent_embeds, query_rel_embeds, obj_embeds = self.forward(s_ent, relation, o_ent)
        #condition_emb = query_ent_embeds + query_rel_embeds
        condition_emb = query_ent_embeds + query_rel_embeds
        bs = query_ent_embeds.size(0)

        #neg = torch.randint(0, self.n_ent, (bs, self.n_ent)).to(condition_emb.device)
        ent_embeds = self.encoder.get_all_ent_embedding().unsqueeze(0).repeat(bs,1,1)

        scores =  -torch.norm(condition_emb.unsqueeze(1) - ent_embeds, dim = -1)
        estimate_dt = 0
        dur_last = 0

        return scores, estimate_dt, dur_last
    def link_prediction_loss(self, intens, type, answers):
        loss = self.lp_loss_fn(intens[:, :-1], answers)
        return loss

