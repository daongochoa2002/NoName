import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from regularizers import N3
from GraphEmbedding import GraphEmbedding
from Denoiser import Denoiser
class NoName(nn.Module):
    def __init__(self, config, eps=0.2, noise_scale = 0.1, noise_min = 0.001, noise_max = 0.02, steps = 4, beta_fixed=True):
        super(NoName, self).__init__()
        self.config = config
        self.n_ent = config.n_ent
        self.n_rel = config.n_rel
        self.d_model = config.d_model
        self.dropout_rate = config.dropout
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-6)
        self.emb_regularizer = N3(0.004)
        self.encoder1 = GraphEmbedding(self.n_ent, self.n_rel,self.d_model, self.dropout_rate, self.dropout_rate, self.dropout_rate)
        self.encoder2 = GraphEmbedding(self.n_ent, self.n_rel,self.d_model, self.dropout_rate, self.dropout_rate, self.dropout_rate)
        self.denoiser1 = Denoiser(self.d_model, self.d_model, self.d_model)
        self.denoiser2 = Denoiser(self.d_model,self.d_model, self.d_model)

        self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
        self.alphas = 1 - self.betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.time_mlp = nn.Linear(self.d_model, self.d_model)
        self.w = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.d_model))).float(), requires_grad=True)

        self.lp_loss_fn = nn.CrossEntropyLoss()
    def product(self,a_real, a_img, b_real, b_img):
        c_real = a_real * b_real - a_img * b_img
        c_img = a_real * b_img + a_img * b_real
        return c_real, c_img
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
    def forward(self, heads, rels, tails, day):
        bs = heads.size(0)
        d_img = torch.sin(self.w.view(1, -1) * (day).unsqueeze(1))
        d_real = torch.cos(self.w.view(1, -1) * (day).unsqueeze(1))
        rels_embeds_real = d_real * self.encoder1.get_rel_embedding(rels) - d_img * self.encoder2.get_rel_embedding(rels)
        heads_embeds_real = self.encoder1.get_ent_embedding(heads)
        tails_embeds_real = self.encoder1.get_ent_embedding(tails)
        rels_embeds_img = d_real * self.encoder2.get_rel_embedding(rels) + d_img * self.encoder1.get_rel_embedding(rels)
        heads_embeds_img = self.encoder2.get_ent_embedding(heads)
        tails_embeds_img = self.encoder2.get_ent_embedding(tails)
        return heads_embeds_real, rels_embeds_real, tails_embeds_real, heads_embeds_img, rels_embeds_img, tails_embeds_img

    def link_prediction_loss(self, intens, type, answers):
        loss = self.lp_loss_fn(intens, answers)
        return loss

    def q_sample(self, x_start, t, noise=None):
      if noise is None:
        noise = torch.randn_like(x_start)
      return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
      arr = arr.cuda()
      res = arr[timesteps].float()
      while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
      return res.expand(broadcast_shape)

    def train_forward(self, heads, rels, tails, year, month, day, neg):
        heads_embs1, rels_embs1, x_start1, \
        heads_embs2, rels_embs2, x_start2 = self.forward(heads, rels, tails,(month + day%month + year % month))
        bs = heads.size(0)
        ts = torch.randint(0, self.steps,(bs,))
        d_img = torch.cos(self.w.view(1, -1) * (month + day%month + year % month).unsqueeze(1))
        d_real = torch.sin(self.w.view(1, -1) *(month + day%month + year % month).unsqueeze(1))
        #x_t = torch.sqrt(alphas[t]) * obj_embeds + self.betas[t] / torch.sqrt(1 - torch.cumprod(alphas[ : t])) * noise
        noise = torch.randn_like(x_start1)
        pos_x_t1 = self.q_sample(x_start1, ts, noise)
        condition_emb1 = self.encoder1(heads_embs1 + rels_embs1, d_real)
        noise = torch.randn_like(x_start2)
        pos_x_t2 = self.q_sample(x_start2, ts, noise)
        condition_emb2 = self.encoder2(heads_embs2 - rels_embs2, d_img)
        #condition_emb = query_ent_embeds * query_rel_embeds
        #condition_emb = self.dropout(query_ent_embeds * time + query_rel_embeds * time)
        #condition_emb = self.ent_decoder(query_ent_embeds,query_rel_embeds,self.ent_embeds.weight)


        neg_x_start1 = self.encoder1.get_ent_embedding(neg)
        noise = torch.randn_like(neg_x_start1)
        neg_x_t1 = self.q_sample(neg_x_start1, ts, noise)

        neg_x_start2 = self.encoder2.get_ent_embedding(neg)
        noise = torch.randn_like(neg_x_start2)
        neg_x_t2 = self.q_sample(neg_x_start2, ts, noise)

        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.d_model // 2, dtype=torch.float32) / (self.d_model // 2)).to(x_start1.device)
        temp = ts[:, None].float().to(x_start1.device) * freqs[None]
        time_embs = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
        if self.d_model % 2:
            time_embs = torch.cat([time_embs, torch.zeros_like(time_embs[:, :1])], dim=-1)
        time_embs = self.time_mlp(time_embs)

        ent_embs1 = torch.cat([pos_x_t1.unsqueeze(1),neg_x_t1],dim = 1)
        ent_embs2 = torch.cat([pos_x_t2.unsqueeze(1),neg_x_t2],dim = 1)
        #pos_x_t = 1 / self.sqrt_alphas_cumprod[ts].unsqueeze(1)  * pos_x_t  - \
        #                torch.sqrt(1 / self.alphas_cumprod[ts] - 1).unsqueeze(1) * self.denoiser(pos_x_t.unsqueeze(1), time_emb.unsqueeze(1), condition_emb.unsqueeze(1)).squeeze(1)
        ent_embs1 = 1 / self.sqrt_alphas_cumprod[ts].unsqueeze(1).unsqueeze(1) * ent_embs1 - \
                        torch.sqrt(1 / self.alphas_cumprod[ts].unsqueeze(1).unsqueeze(1) - 0.1) * self.denoiser1(ent_embs1, time_embs, condition_emb1)

        ent_embs2 = 1 / self.sqrt_alphas_cumprod[ts].unsqueeze(1).unsqueeze(1) * ent_embs2 - \
                        torch.sqrt(1 / self.alphas_cumprod[ts].unsqueeze(1).unsqueeze(1) - 0.1) * self.denoiser2(ent_embs2, time_embs, condition_emb2)

        #condition_emb1 = (heads_embs1 + rels_embs1) * condition_emb1
        condition_emb1, condition_emb2 = self.product((heads_embs1 + rels_embs1), (heads_embs1 - rels_embs1), condition_emb1, condition_emb2) #condition_emb2 = (heads_embs2 - rels_embs2) * condition_emb2
        ent_type = torch.cat([tails.unsqueeze(1),neg],dim = 1)
        type_intes = self.dropout(condition_emb1.multiply(x_start1).unsqueeze(1) \
                                  - condition_emb1.unsqueeze(1).multiply(ent_embs1)).sum(dim=-1) \
                                  + self.dropout(condition_emb2.multiply(x_start2).unsqueeze(1) \
                                  - condition_emb2.unsqueeze(1).multiply(ent_embs2)).sum(dim=-1)

        labels = torch.cat([torch.ones_like(tails).unsqueeze(1), torch.zeros_like(neg)], dim=1)
        #type_intes = (torch.norm(condition_emb - x_start, dim=1).unsqueeze(1) - torch.norm(condition_emb.unsqueeze(1) - ent_embeds, dim=-1))
        factor = (torch.linalg.norm(heads_embs1 - x_start1, ord = 3, dim=1), torch.linalg.norm(heads_embs2 -x_start2, ord = 3, dim=1),\
                  torch.linalg.norm(rels_embs1, ord = 3, dim=1),torch.linalg.norm(rels_embs1, ord = 3, dim=1))

        loss_lp =  self.link_prediction_loss(type_intes, ent_type, torch.zeros_like(tails)) + 5 * contrastive_loss(type_intes, labels)

        return loss_lp.mean() + self.emb_regularizer.forward(factor)


    def test_forward(self, sub, rels, tails, year, month, day):
        heads_embs1, rels_embs1, _, \
        heads_embs2, rels_embs2, _  = self.forward(sub, rels, tails, (month + day%month + year % month))
        d_img = torch.cos(self.w.view(1, -1) * (month + day%month + year % month).unsqueeze(1))
        d_real = torch.sin(self.w.view(1, -1) * (month + day%month + year % month).unsqueeze(1))
        condition_emb1 = self.encoder1(heads_embs1 + rels_embs1, d_real)
        condition_emb2 = self.encoder2(heads_embs2 - rels_embs2, d_img)
        bs = heads_embs1.size(0)
        #type_intes, type = self.link_prediction(time, query_ent_embeds, query_rel_embeds)
        #alphas = 1 - self.betas
        #alphas_cumprod = torch.cumprod(alphas, axis=0)
        x_t1 = torch.randn_like(condition_emb1)
        x_t2 = torch.randn_like(condition_emb2)
        t = self.steps - 1
        while t >= 0:
          freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.d_model // 2, dtype=torch.float32) / (self.d_model // 2)).to(x_t1.device)
          temp = t * freqs[None].repeat(bs,1)
          time_embs = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)

          if self.d_model % 2:
              time_embs = torch.cat([time_embs, torch.zeros_like(time_embs[:, :1])], dim=-1)

          time_embs = self.time_mlp(time_embs)
          x_t1 = 1 / self.sqrt_alphas[t] * (x_t1 - (1 - self.alphas[t]) / self.sqrt_one_minus_alphas_cumprod[t] * \
                                              self.denoiser1(x_t1.unsqueeze(1), time_embs, condition_emb1).squeeze(1))
          x_t2 = 1 / self.sqrt_alphas[t] * (x_t2 - (1 - self.alphas[t]) / self.sqrt_one_minus_alphas_cumprod[t] * \
                                              self.denoiser2(x_t2.unsqueeze(1), time_embs, condition_emb2).squeeze(1))
          if t > 0:
            x_t1 += self.betas[t] * torch.randn_like(condition_emb1)
            x_t2 += self.betas[t] * torch.randn_like(condition_emb2)
          t -=  1
        #scores = self.ents_score(type_intes, type, local_weight)
        #
        #print(x_t)
        #neg = torch.randint(0, self.n_ent, (bs, 2500)).to(x_t.device)
        #ent_type =  torch.arange(self.n_ent, device=sub.device).unsqueeze(0).repeat(sub.size(0), 1)
        #ent_embeds = torch.cat([obj_embeds.unsqueeze(1),ent_embeds],dim = 1)
        ent_embeds_real = self.encoder1.get_all_ent_embedding()
        ent_embeds_img = self.encoder2.get_all_ent_embedding()
        #condition_emb1 = (heads_embs1 + rels_embs1) * condition_emb1
        condition_emb1, condition_emb2 = self.product((heads_embs1 + rels_embs1), (heads_embs1 - rels_embs1), condition_emb1, condition_emb2) #condition_emb2 = (heads_embs2 - rels_embs2) * condition_emb2
        scores = F.softplus(self.dropout(condition_emb1.multiply(x_t1).unsqueeze(1)).sum(dim=-1)\
                      - condition_emb1.mm(ent_embeds_real.transpose(0,1))\
                      + self.dropout(condition_emb2.multiply(x_t2).unsqueeze(1)).sum(dim=-1)\
                      - condition_emb2.mm(ent_embeds_img.transpose(0,1)))

        return scores

def contrastive_loss(distances, labels, pos_margin=20, neg_margin=-20.0):
    """
    Compute contrastive loss.
    distances: Pairwise distances between embeddings.
    labels: Binary labels (1 for positive pairs, 0 for negative pairs).
    pos_margin: Margin for positive pairs.
    neg_margin: Margin for negative pairs.
    """
    loss_pos = labels * torch.pow(F.relu(distances - pos_margin), 2)
    loss_neg = (1 - labels) * torch.pow(F.relu(neg_margin - distances), 2)
    return torch.mean(loss_pos + loss_neg)
