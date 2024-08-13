import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
#from models.GraphEncoder import RGTEncoder, RGCNEncoder
#from models.SequenceEncoder import TransformerEncoder
#import torch_scatter
from ConvTransE import ConvTransE
from Denoiser import Denoiser
class TemporalTransformerHawkesGraphModel(nn.Module):
    def __init__(self, config, eps=0.2, noise_scale = 0.1, noise_min = 0.0001, noise_max = 0.02, steps = 5, beta_fixed=True):
        super(TemporalTransformerHawkesGraphModel, self).__init__()
        self.config = config
        self.n_ent = config.n_ent
        self.n_rel = config.n_rel
        self.d_model = config.d_model
        self.dropout_rate = config.dropout
        self.PAD_TIME = -1
        self.PAD_ENTITY = self.n_ent - 1
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps

        self.dropout = nn.Dropout(self.dropout_rate)

        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-6)

        self.encoder_real = ConvTransE(self.n_ent, self.n_rel,self.d_model, self.dropout_rate, self.dropout_rate, self.dropout_rate)
        self.encoder_img = ConvTransE(self.n_ent, self.n_rel,self.d_model, self.dropout_rate, self.dropout_rate, self.dropout_rate)
        self.denoiser_real = Denoiser(self.d_model,self.d_model,self.d_model)
        self.denoiser_img = Denoiser(self.d_model,self.d_model,self.d_model)

        self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.gamma = 1e-6  # Replace with your desired margin value
        self.time_mlp = nn.Linear(self.d_model, self.d_model)
        self.w1 = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.d_model))).float(), requires_grad=True)
        self.w2 = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.d_model))).float(), requires_grad=True)
        #self.tp_loss_fn = nn.MSELoss()
        self.lp_loss_fn = nn.CrossEntropyLoss()


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
    def forward(self, heads, rels, tails, year, month, day):
        bs = heads.size(0)
        rels_embeds_real = self.encoder_real.get_rel_embedding(rels)
        heads_embeds_real = self.encoder_real.get_ent_embedding_s(heads)
        tails_embeds_real = self.encoder_real.get_ent_embedding_s(tails)
        rels_embeds_img = self.encoder_img.get_rel_embedding(rels)
        heads_embeds_img = self.encoder_img.get_ent_embedding_s(heads)
        tails_embeds_img = self.encoder_img.get_ent_embedding_s(tails)
        return heads_embeds_real, rels_embeds_real, tails_embeds_real, heads_embeds_img, rels_embeds_img, tails_embeds_img

    def link_prediction_loss(self, intens, type, answers):
        loss = self.lp_loss_fn(intens, answers)
        return loss


    def ents_score(self, intens, type, local_weight=1.):
        return intens[:, :-1]


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


    def train_forward(self, s_ent, relation, o_ent, year, month, day, neg):
        heads_embeds_real, rels_embeds_real, x_start_real, \
        heads_embeds_img, rels_embeds_img, x_start_img = self.forward(s_ent, relation, o_ent, year, month, day)
        bs = s_ent.size(0)
        ts = torch.randint(0, self.steps,(bs,))
        d_img = torch.sin(self.w1.view(1, -1) * day.unsqueeze(1))
        d_real = torch.cos(self.w2.view(1, -1) * day.unsqueeze(1))
        #x_t = torch.sqrt(alphas[t]) * obj_embeds + self.betas[t] / torch.sqrt(1 - torch.cumprod(alphas[ : t])) * noise
        noise = torch.randn_like(x_start_real)
        pos_x_t_real = self.q_sample(x_start_real, ts, noise)
        condition_emb_real = self.encoder_real(heads_embeds_real + rels_embeds_img, d_real)
        noise = torch.randn_like(x_start_img)
        pos_x_t_img = self.q_sample(x_start_img, ts, noise)
        condition_emb_img = self.encoder_img(heads_embeds_img - rels_embeds_real, d_img)
        #condition_emb = query_ent_embeds * query_rel_embeds
        #condition_emb = self.dropout(query_ent_embeds * time + query_rel_embeds * time)
        #condition_emb = self.ent_decoder(query_ent_embeds,query_rel_embeds,self.ent_embeds.weight)


        neg_x_start_real = self.encoder_real.get_ent_embedding_s(neg)
        noise = torch.randn_like(neg_x_start_real)
        neg_x_t_real = self.q_sample(neg_x_start_real, ts, noise)

        neg_x_start_img = self.encoder_img.get_ent_embedding_s(neg)
        noise = torch.randn_like(neg_x_start_img)
        neg_x_t_img = self.q_sample(neg_x_start_img, ts, noise)

        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.d_model // 2, dtype=torch.float32) / (self.d_model // 2)).to(x_start_real.device)
        temp = ts[:, None].float().to(x_start_real.device) * freqs[None]
        time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
        if self.d_model % 2:
            time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
        time_emb = self.time_mlp(time_emb)

        ent_embeds_real = torch.cat([pos_x_t_real.unsqueeze(1),neg_x_t_real],dim = 1)
        ent_embeds_img = torch.cat([pos_x_t_img.unsqueeze(1),neg_x_t_img],dim = 1)
        #pos_x_t = 1 / self.sqrt_alphas_cumprod[ts].unsqueeze(1)  * pos_x_t  - \
        #                torch.sqrt(1 / self.alphas_cumprod[ts] - 1).unsqueeze(1) * self.denoiser(pos_x_t.unsqueeze(1), time_emb.unsqueeze(1), condition_emb.unsqueeze(1)).squeeze(1)
        ent_embeds_real = 1 / self.sqrt_alphas_cumprod[ts].unsqueeze(1).unsqueeze(1) * ent_embeds_real - \
                        torch.sqrt(1 / self.alphas_cumprod[ts].unsqueeze(1).unsqueeze(1) - 1) * self.denoiser_real(ent_embeds_real, time_emb.unsqueeze(1), condition_emb_real.unsqueeze(1))
        ent_embeds_img = 1 / self.sqrt_alphas_cumprod[ts].unsqueeze(1).unsqueeze(1) * ent_embeds_img - \
                        torch.sqrt(1 / self.alphas_cumprod[ts].unsqueeze(1).unsqueeze(1) - 1) * self.denoiser_img(ent_embeds_img, time_emb.unsqueeze(1), condition_emb_img.unsqueeze(1))


        #ent_type = torch.arange(self.n_ent, device=query_ent_embeds.device).unsqueeze(0).repeat(o_ent.size(0), 1)
        #mask = (global_type != obj.unsqueeze(1).repeat(1, self.n_ent))
        #print(pos_x_t.shape)
        #pos_x_t = pos_x_t.float()
        #neg_x_t = neg_x_t.double()
        #pos_dist = (condition_emb.multiply(x_start) - condition_emb.multiply(pos_x_t)).sum(dim=-1)
        #neg_dist = (condition_emb.multiply(x_start).unsqueeze(1) - condition_emb.unsqueeze(1).multiply(neg_x_t)).sum(dim=-1)
        #pos_dist = torch.norm(condition_emb.multiply(x_start) - condition_emb.multiply(pos_x_t), dim=-1)
        #neg_dist = torch.norm(condition_emb.multiply(x_start).unsqueeze(1) - condition_emb.unsqueeze(1).multiply(neg_x_t), dim=-1)
        #type_intes = self.ent_decoder(query_ent_embeds,query_rel_embeds,self.ent_embeds.weight)
        #print(denoised_neg_x.shape)
        #a = - torch.norm(query_ent_embeds + query_rel_embeds - x_start, dim=1)
        #pos_dist = torch.linalg.norm(x_start - pos_x_t, dim=1)
        #neg_dist = torch.linalg.norm(x_start.unsqueeze(1) - neg_x_t , dim= -1)
        #print(neg_dist.shape)
        #pos_dist = torch.norm(query_ent_embeds + query_rel_embeds - x_start, dim=1)\
        #                             - torch.norm(query_ent_embeds + query_rel_embeds - pos_x_t, dim=1)
        #neg_dist = torch.norm(query_ent_embeds + query_rel_embeds - x_start, dim=1).unsqueeze(1)\
        #                             - torch.norm(query_ent_embeds.unsqueeze(1) + query_rel_embeds.unsqueeze(1) - neg_x_t, dim=-1)
        #pos_dist = type_intes.
        #loss_lp = - torch.log(torch.sigmoid(self.gamma - pos_dist)) - torch.sum(torch.log(torch.sigmoid(neg_dist - self.gamma)), dim=1)
        condition_emb_real = (heads_embeds_real + rels_embeds_real) * condition_emb_real
        condition_emb_img = (heads_embeds_img - rels_embeds_img) * condition_emb_img
        ent_type = torch.cat([o_ent.unsqueeze(1),neg],dim = 1)
        type_intes = (self.dropout(condition_emb_real.multiply(x_start_real).unsqueeze(1) \
                                  - condition_emb_real.unsqueeze(1).multiply(ent_embeds_real)).sum(dim=-1) \
                                  + self.dropout(condition_emb_img.multiply(x_start_img).unsqueeze(1) \
                                  - condition_emb_img.unsqueeze(1).multiply(ent_embeds_img)).sum(dim=-1)) /2
        #type_intes = (torch.norm(condition_emb - x_start, dim=1).unsqueeze(1) - torch.norm(condition_emb.unsqueeze(1) - ent_embeds, dim=-1))
        loss_lp = self.link_prediction_loss(type_intes, ent_type, torch.zeros_like(o_ent))

        return loss_lp.mean()


    def test_forward(self, s_ent, relation, o_ent, year, month, day, local_weight=1.):
        heads_embeds_real, rels_embeds_real, x_start_real, \
        heads_embeds_img, rels_embeds_img, x_start_img  = self.forward(s_ent, relation, o_ent, year, month, day)
          #condition_emb = self.encoder(query_ent_embeds,query_rel_embeds)
          #condition_emb = query_ent_embeds * query_rel_embeds
          #condition_emb = self.dropout(query_ent_embeds * time + query_rel_embeds * time)
        d_img = torch.sin(self.w1.view(1, -1) * day.unsqueeze(1))
        d_real = torch.cos(self.w2.view(1, -1) * day.unsqueeze(1))
        condition_emb_real = self.encoder_real(heads_embeds_real + rels_embeds_img, d_real)
        condition_emb_img = self.encoder_img(heads_embeds_img - rels_embeds_real, d_img)
        bs = heads_embeds_real.size(0)
        #type_intes, type = self.link_prediction(time, query_ent_embeds, query_rel_embeds)
        #alphas = 1 - self.betas
        #alphas_cumprod = torch.cumprod(alphas, axis=0)
        x_t_real = torch.randn_like(condition_emb_real)
        x_t_img = torch.randn_like(condition_emb_img)
        t = self.steps - 1
        while t >= 0:
          freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.d_model // 2, dtype=torch.float32) / (self.d_model // 2)).to(x_t_real.device)
          temp = t * freqs[None].repeat(bs,1)
          time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)

          if self.d_model % 2:
              time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)

          time_emb = self.time_mlp(time_emb)
          x_t_real = 1 / self.sqrt_alphas_cumprod[t] * (x_t_real - self.betas[t] / self.sqrt_one_minus_alphas_cumprod[t] * \
                                              self.denoiser_real(x_t_real.unsqueeze(1), time_emb.unsqueeze(1), condition_emb_real.unsqueeze(1)).squeeze(1))
          x_t_img = 1 / self.sqrt_alphas_cumprod[t] * (x_t_img - self.betas[t] / self.sqrt_one_minus_alphas_cumprod[t] * \
                                              self.denoiser_img(x_t_img.unsqueeze(1), time_emb.unsqueeze(1), condition_emb_img.unsqueeze(1)).squeeze(1))
          if t > 0:
            x_t_real += 0.1 * torch.randn_like(condition_emb_real)
            x_t_img += 0.1 * torch.randn_like(condition_emb_img)
          t -=  1
        #scores = self.ents_score(type_intes, type, local_weight)
        #
        #print(x_t)
        #neg = torch.randint(0, self.n_ent, (bs, 2500)).to(x_t.device)
        #ent_type =  torch.arange(self.n_ent, device=sub.device).unsqueeze(0).repeat(sub.size(0), 1)
        #ent_embeds = torch.cat([obj_embeds.unsqueeze(1),ent_embeds],dim = 1)
        ent_embeds_real = self.encoder_real.get_all_ent_embedding()
        ent_embeds_img = self.encoder_img.get_all_ent_embedding()
        condition_emb_real = (heads_embeds_real + rels_embeds_real) * condition_emb_real
        condition_emb_img = (heads_embeds_img - rels_embeds_img) * condition_emb_img
        scores = (self.dropout(condition_emb_real.multiply(x_t_real).unsqueeze(1)).sum(dim=-1)\
                      - condition_emb_real.mm(ent_embeds_real.transpose(0,1))\
                      + self.dropout(condition_emb_img.multiply(x_t_img).unsqueeze(1)).sum(dim=-1)\
                      - condition_emb_img.mm(ent_embeds_img.transpose(0,1))) /2
        #scores = (torch.norm(condition_emb - x_t, dim=1).unsqueeze(1) - torch.norm(condition_emb.unsqueeze(1) - ent_embeds, dim=-1))
        #print(x_t)
        #scores =  torch.norm((condition_emb * x_t).unsqueeze(1) - condition_emb.unsqueeze(1) * ent_embeds,dim=-1)
        #scores =  condition_emb.multiply(x_t).sum(dim=-1).unsqueeze(1) - condition_emb.unsqueeze(1).multiply(ent_embeds).sum(dim=-1)
        #scores = torch.linalg.norm(x_t.unsqueeze(1) - self.ent_embeds.weight.unsqueeze(0), dim=-1)
        #scores = torch.norm(query_ent_embeds + query_rel_embeds - x_t, dim=1).unsqueeze(1) - \
        #                torch.norm(query_ent_embeds.unsqueeze(1) + query_rel_embeds.unsqueeze(1) - self.ent_embeds.weight.unsqueeze(0), dim=-1)
        #scores = self.ent_decoder(query_ent_embeds,query_rel_embeds, x_t,self.ent_embeds.weight)

        return scores