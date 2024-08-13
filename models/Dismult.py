import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class DisMult(nn.Module):
    def __init__(self, config):
        super(DisMult, self).__init__()
        self.config = config
        self.n_ent = config.n_ent
        self.n_rel = config.n_rel
        self.d_model = config.d_model
        self.dropout_rate = config.dropout


        self.dropout = nn.Dropout(self.dropout_rate)

        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-6)

        self.encoder = ConvTransE(self.n_ent, self.n_rel,self.d_model, self.dropout_rate, self.dropout_rate, self.dropout_rate)
        #self.tp_loss_fn = nn.MSELoss()
        self.lp_loss_fn = nn.CrossEntropyLoss()

    def forward(self, query_entities, query_relations, obj_entities):
        bs = query_entities.size(0)
        query_rel_embeds = self.encoder.get_rel_embedding(query_relations)
        query_ent_embeds = self.encoder.get_ent_embedding(query_entities)
        obj_embeds = self.encoder.get_ent_embedding(obj_entities)
        return query_ent_embeds, query_rel_embeds, obj_embeds


    def train_forward(self, s_ent, relation, o_ent, year, month, day):
        query_ent_embeds, query_rel_embeds, obj_embeds = self.forward(s_ent, relation, o_ent)
        bs = query_ent_embeds.size(0)

        condition_emb = query_ent_embeds * query_rel_embeds
        neg = torch.randint(0, self.n_ent, (bs, 500)).to(condition_emb.device)
        ent_embeds = self.encoder.get_ent_embedding(neg)
        ent_embeds = torch.cat([obj_embeds.unsqueeze(1),ent_embeds],dim = 1)
        ent_type = torch.cat([o_ent.unsqueeze(1),neg],dim = 1)

        type_intes = (condition_emb.unsqueeze(1) * ent_embeds).sum(dim=-1)
        loss_lp = self.link_prediction_loss(type_intes, ent_type, torch.zeros_like(o_ent))
        return loss_lp.mean()


    def test_forward(self, s_ent, relation, o_ent, year, month, day, local_weight=1.):
        query_ent_embeds, query_rel_embeds, obj_embeds = self.forward(s_ent, relation, o_ent)
        #condition_emb = query_ent_embeds + query_rel_embeds
        condition_emb = query_ent_embeds * query_rel_embeds
        bs = query_ent_embeds.size(0)

        neg = torch.randint(0, self.n_ent, (bs, self.n_ent)).to(condition_emb.device)
        #ent_embeds = self.encoder.get_all_ent_embedding().unsqueeze(0).repeat(bs,1,1)
        ent_embeds = self.encoder.get_ent_embedding(neg)
        ent_embeds = torch.cat([obj_embeds.unsqueeze(1),ent_embeds],dim = 1)

        scores =  (condition_emb.unsqueeze(1) * ent_embeds).sum(dim=-1)
        #scores = torch.linalg.norm(x_t.unsqueeze(1) - self.ent_embeds.weight.unsqueeze(0), dim=-1)
        #scores = torch.norm(query_ent_embeds + query_rel_embeds - x_t, dim=1).unsqueeze(1) - \
        #                torch.norm(query_ent_embeds.unsqueeze(1) + query_rel_embeds.unsqueeze(1) - self.ent_embeds.weight.unsqueeze(0), dim=-1)
        #scores = self.ent_decoder(query_ent_embeds,query_rel_embeds, x_t,self.ent_embeds.weight)

        return scores
    def link_prediction_loss(self, intens, type, answers):
        loss = self.lp_loss_fn(intens[:, :-1], answers)
        return loss

