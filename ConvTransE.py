import torch
import math
import torch.nn as nn
import torch.nn.functional as F
class ConvTransE(torch.nn.Module):

    def __init__(self, num_entities, num_relations, embedding_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=50, kernel_size=3, use_bias=True):

        super(ConvTransE, self).__init__()
        self.n_ent = num_entities
        self.n_rel = num_relations
        self.d_model = embedding_dim
        self.ent_embeds = nn.Embedding(self.n_ent, self.d_model)
        self.rel_embeds = nn.Embedding(self.n_rel, self.d_model)
        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)

        self.conv1 = torch.nn.Conv1d(2, channels, kernel_size, stride=1,
                               padding=int(math.floor(kernel_size / 3)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', nn.Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(embedding_dim)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.ent_embeds.weight)
        nn.init.xavier_uniform_(self.rel_embeds.weight)

    def get_ent_embedding(self, ents):
        return self.ent_embeds(ents)
    def get_rel_embedding(self,rels):
        return self.rel_embeds(rels)

    def get_all_ent_embedding(self):
        return self.ent_embeds.weight

    def forward(self, query_ent_embeds, query_rel_embeds):
        batch_size = query_ent_embeds.size(0)
        stacked_inputs = torch.cat((query_ent_embeds.unsqueeze(1), query_rel_embeds.unsqueeze(1)), dim=1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)

        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)

        #x = torch.mm(x, all_embeds.transpose(1, 0))
        return x
