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
    def __init__(self, config, eps=0.2, noise_scale=0.1, noise_min=0.0001, noise_max=0.02, steps=5, beta_fixed=True):
        super(NoName, self).__init__()
        # Assign configuration and hyperparameters
        self.config = config
        self.n_ent = config.n_ent  # Number of entities
        self.n_rel = config.n_rel  # Number of relations
        self.d_model = config.d_model  # Dimensionality of model embeddings
        self.dropout_rate = config.dropout  # Dropout rate
        self.noise_scale = noise_scale  # Scaling factor for noise
        self.noise_min = noise_min  # Minimum noise level
        self.noise_max = noise_max  # Maximum noise level
        self.steps = steps  # Number of diffusion steps
        
        # Define layers and operations
        self.dropout = nn.Dropout(self.dropout_rate)  # Dropout layer
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-6)  # Layer normalization
        self.emb_regularizer = N3(0.0025)  # Regularizer
        self.encoder1 = GraphEmbedding(self.n_ent, self.n_rel, self.d_model, self.dropout_rate, self.dropout_rate, self.dropout_rate)  # First encoder
        self.encoder2 = GraphEmbedding(self.n_ent, self.n_rel, self.d_model, self.dropout_rate, self.dropout_rate, self.dropout_rate)  # Second encoder
        self.denoiser1 = Denoiser(self.d_model, self.d_model, self.d_model)  # First denoiser
        self.denoiser2 = Denoiser(self.d_model, self.d_model, self.d_model)  # Second denoiser


        self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
        self.alphas = 1 - self.betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.time_mlp = nn.Linear(self.d_model, self.d_model)  # Linear layer for time-based embeddings
        self.w = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.d_model))).float(), requires_grad=True)  # Frequency scaling parameter
        self.lp_loss_fn = nn.CrossEntropyLoss()  # Loss function for link prediction


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

    def link_prediction_loss(self, intens, answers):
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

    def make_noise(self, x_start, t):
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        return x_t

    def sinusoidal_pos_embedding(self, t):
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.d_model // 2, dtype=torch.float32) / (self.d_model // 2))
        temp = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1).cuda()
        if self.d_model % 2:
            emb  = torch.cat([emb , torch.zeros_like(emb [:, :1])], dim=-1)
        emb  = self.time_mlp(emb )
        return emb 

    def train_forward(self, heads, rels, tails, year, month, day, neg):
        """
        Perform a forward pass during training to compute the loss for the link prediction task.

        Args:
            heads (Tensor): Batch of head entity indices.
            rels (Tensor): Batch of relation indices.
            tails (Tensor): Batch of tail entity indices.
            year (int): The year parameter, currently unused.
            month (int): The month parameter used for temporal encoding.
            day (int): The day parameter, currently unused.
            neg (Tensor): Batch of negative entity indices for contrastive learning.

        Returns:
            Tensor: The combined loss (link prediction loss + regularization loss).
        """
        # Obtain embeddings for heads, relations, and tails at the specified month
        heads_embs1, rels_embs1, x_start1, heads_embs2, rels_embs2, x_start2 = self.forward(heads, rels, tails, month)
        
        # Get the batch size
        bs = heads.size(0)
        
        # Generate random time steps for each item in the batch
        ts = torch.randint(0, self.steps, (bs,))
        
        # Compute temporal encodings using sine and cosine functions
        d_img = torch.sin(self.w.view(1, -1) * (month).unsqueeze(1))
        d_real = torch.cos(self.w.view(1, -1) * (month).unsqueeze(1))
        
        # Calculate conditioning embeddings using the encoders
        cond_embs1 = self.encoder1(heads_embs1 + rels_embs1, d_real)
        cond_emb2 = self.encoder2(heads_embs2 + rels_embs2, d_img)
        
        # Retrieve embeddings for the negative samples
        neg_x_start1 = self.encoder1.get_ent_embedding(neg)
        neg_x_start2 = self.encoder2.get_ent_embedding(neg)
        
        # Compute sinusoidal positional embeddings based on time steps
        time_embs = self.sinusoidal_pos_embedding(ts)
        
        # Concatenate starting embeddings with negative sample embeddings
        ent_embs1 = torch.cat([x_start1.unsqueeze(1), neg_x_start1], dim=1)
        ent_embs2 = torch.cat([x_start2.unsqueeze(1), neg_x_start2], dim=1)
        
        # Add noise to the concatenated embeddings
        ent_embs1 = self.make_noise(ent_embs1)
        ent_embs2 = self.make_noise(ent_embs2)
        
        # Apply denoising and normalization to the embeddings
        ent_embs1 = 1 / self.sqrt_alphas_cumprod[ts].unsqueeze(1).unsqueeze(1) * ent_embs1 - \
                    torch.sqrt(1 / self.alphas_cumprod[ts].unsqueeze(1).unsqueeze(1) - 0.1) * \
                    self.denoiser1(ent_embs1, time_embs.unsqueeze(1), cond_embs1.unsqueeze(1))

        ent_embs2 = 1 / self.sqrt_alphas_cumprod[ts].unsqueeze(1).unsqueeze(1) * ent_embs2 - \
                    torch.sqrt(1 / self.alphas_cumprod[ts].unsqueeze(1).unsqueeze(1) - 0.1) * \
                    self.denoiser2(ent_embs2, time_embs.unsqueeze(1), cond_emb2.unsqueeze(1))
        
        # Compute conditioning embeddings and intensities
        cond_embs1, cond_emb2 = self.product((heads_embs1 + rels_embs1), (heads_embs1 + rels_embs1), cond_embs1, cond_emb2)

        type_intes = (self.dropout(cond_embs1.multiply(x_start1).unsqueeze(1) \
                                  - cond_embs1.unsqueeze(1).multiply(ent_embs1)).sum(dim=-1) \
                      + self.dropout(cond_emb2.multiply(x_start2).unsqueeze(1) \
                                  - cond_emb2.unsqueeze(1).multiply(ent_embs2)).sum(dim=-1))
        
        # Compute link prediction loss
        loss_lp = self.link_prediction_loss(type_intes, torch.zeros_like(tails))

        # Compute regularization factors based on the norms of embeddings
        factor = (torch.linalg.norm(heads_embs1, ord=3, dim=1), torch.linalg.norm(heads_embs2, ord=3, dim=1),
                  torch.linalg.norm(rels_embs1, ord=3, dim=1), torch.linalg.norm(rels_embs1, ord=3, dim=1))

        # Return the combined loss (link prediction loss + regularization loss)
        return loss_lp.mean() + self.emb_regularizer.forward(factor)

    def test_forward(self, sub, rels, tails, year, month, day):
        """
        Perform a forward pass during testing to compute scores for entity links.

        Args:
            sub (Tensor): Batch of subject entity indices.
            rels (Tensor): Batch of relation indices.
            tails (Tensor): Batch of tail entity indices.
            year (int): The year parameter, currently unused.
            month (int): The month parameter used for temporal encoding.
            day (int): The day parameter, currently unused.

        Returns:
            Tensor: The computed scores for the link prediction task.
        """
        # Obtain embeddings for subjects, relations, and tails at the specified month
        h_embs1, r_embs1, _, h_embs2, r_embs2, _ = self.forward(sub, rels, tails, month)

        # Compute temporal encodings using sine and cosine functions
        d_img = torch.sin(self.w.view(1, -1) * (month).unsqueeze(1))
        d_real = torch.cos(self.w.view(1, -1) * (month).unsqueeze(1))

        # Calculate conditioning embeddings using the encoders
        cond_emb1 = self.encoder1(h_embs1 + r_embs1, d_real)
        cond_emb2 = self.encoder2(h_embs2 - r_embs2, d_img)
        
        # Initialize random noise for the forward pass
        x_t1 = torch.randn_like(cond_emb1)
        x_t2 = torch.randn_like(cond_emb2)
        t = self.steps - 1  # Start from the last time step

        # Iteratively refine the embeddings by simulating the denoising process
        while t >= 0:
            # Compute sinusoidal positional embeddings for the current time step
            time_embs = self.sinusoidal_pos_embedding(t)
            
            # Apply denoising and normalization to the embeddings
            x_t1 = 1 / self.sqrt_alphas[t] * (x_t1 - (1 - self.alphas[t]) / self.sqrt_one_minus_alphas_cumprod[t] * \
                                                self.denoiser1(x_t1.unsqueeze(1), time_embs.unsqueeze(1), cond_emb1.unsqueeze(1)).squeeze(1))
            x_t2 = 1 / self.sqrt_alphas[t] * (x_t2 - (1 - self.alphas[t]) / self.sqrt_one_minus_alphas_cumprod[t] * \
                                                self.denoiser2(x_t2.unsqueeze(1), time_embs.unsqueeze(1), cond_emb2.unsqueeze(1)).squeeze(1))
            # Add noise if there are more time steps remaining
            if t > 0:
                x_t1 += self.betas[t] * torch.randn_like(cond_emb1)
                x_t2 += self.betas[t] * torch.randn_like(cond_emb2)
            t -= 1

        # Retrieve all entity embeddings
        ent_embs_1 = self.encoder1.get_all_ent_embedding()
        ent_embs_2 = self.encoder2.get_all_ent_embedding()
        
        # Compute conditioning embeddings for the final prediction
        cond_emb1, cond_emb2 = self.product((h_embs1 + r_embs1), (h_embs1 + r_embs1), cond_emb1, cond_emb2)
        
        # Compute scores for the link prediction task
        scores = F.softplus(self.dropout(cond_emb1.multiply(x_t1).unsqueeze(1)).sum(dim=-1) \
                            - cond_emb1.mm(ent_embs_1.transpose(0, 1)) \
                            + self.dropout(cond_emb2.multiply(x_t2).unsqueeze(1)).sum(dim=-1) \
                            - cond_emb2.mm(ent_embs_2.transpose(0, 1)))

        return scores
