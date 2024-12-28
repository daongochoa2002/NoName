import torch
import torch.nn as nn
import torch.nn.functional as F

class CEDenoiserBlock(nn.Module):
    def __init__(self, d_model):
        super(CEDenoiserBlock, self).__init__()
        self.d_model = d_model
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)
        # Pointwise convolution is similar to a linear layer applied across channels
        self.pointwise_conv = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.mlp1 = nn.Linear(d_model, d_model // 2)
        self.mlp2 = nn.Linear(d_model, d_model // 2)
        self.fc = torch.nn.Linear(d_model * 16, d_model // 2)
        # Learnable scaling parameters alpha
        self.alpha1 = nn.Parameter(torch.ones(d_model))
        self.alpha2 = nn.Parameter(torch.ones(d_model))
        self.bn0 = torch.nn.BatchNorm1d(1)
        self.bn1 = torch.nn.BatchNorm1d(16)

    def forward(self, x, time_embedding,condition):
        # Apply LayerNorm before the Pointwise Conv
        bs = x.size(0)
        norm_x = self.layer_norm1(x)
        norm_x = F.relu(self.mlp1(norm_x))
        h = condition + time_embedding
        h = self.dropout(F.relu(self.mlp2(h)))
        out1 = torch.cat([norm_x, h.repeat(1,x.size(1),1)],dim = -1)
        out1 = out1 + x
        #norm_out1 = self.layer_norm2(out1)
        # Pointwise Convolution
        conv_out = self.bn1(self.dropout(self.pointwise_conv(self.bn0(out1.view(bs,1,-1))))).view(bs,-1, 16 * self.d_model)
        conv_out = self.dropout(F.relu(self.fc(conv_out)))
        conv_out = torch.cat([conv_out,h.repeat(1,x.size(1),1)],dim = -1)  # Residual connection with scaling
        # First MLP layer with scaling applied to residual connection
        out = out1 + conv_out  # Residual connection with scaling

        return out

class Denoiser(nn.Module):
    def __init__(self, d_model, time_emb_dim, condition_emb_dim):
        super(Denoiser, self).__init__()
        self.d_model = d_model
        self.time_emb_dim = time_emb_dim
        self.condition_emb_dim = condition_emb_dim

        # MLP for time and condition embedding
        #self.time_cond_mlp = nn.Sequential(
        #    nn.Linear(2 * self.d_model, self.d_model),
        #    nn.ReLU(),
            #nn.Linear(self.d_model, self.d_model)
        #)

        # Encoder and Decoder for noised embedding processing
        self.encoder = nn.Sequential(
            #nn.Conv1d(1, 16, 3, stride=2, padding=1),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            #nn.ConvTranspose1d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Linear(self.d_model // 2, self.d_model),
            nn.Sigmoid()
        )

        # Layer Norm layers
        self.layer_norm1 = nn.LayerNorm(d_model)

    def forward(self, x, time_emb, condition_emb):
        bs = x.size(0)

        # Concatenate and process time and condition embeddings
        h = (time_emb * condition_emb).unsqueeze(1).repeat(1, x.size(1), 1)
        #h = self.time_cond_mlp(h)

        # Normalize x and apply first layer norm
        #x = self.layer_norm1(x)
        h = 0.2 * h + 0.8 * x
        #h = h.view(-1, 100).unsqueeze(1)

        # Encode and decode the noised embedding
        h = self.encoder(h)
        h = self.decoder(h)

        # Reshape the output to the original format
        h = h.view(bs, -1, self.d_model)

        return h


class ConditionalEntityDenoiser(nn.Module):
    def __init__(self, d_model, num_blocks):
        super(ConditionalEntityDenoiser, self).__init__()
        self.d_model = d_model
        self.blocks = nn.ModuleList([Denoiser(self.d_model,self.d_model,self.d_model) for _ in range(num_blocks)])
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x, condition, time_embedding):
        for block in self.blocks:
            x = block(x, condition, time_embedding)
