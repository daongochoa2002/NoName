import torch
import torch.nn as nn

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
        h = (time_emb * condition_emb).repeat(1, x.size(1), 1)
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
