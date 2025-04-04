import torch
import torch.nn as nn
from models.conf_lass.conformer import Conformer


class PositionalEmbedding(nn.Module):
    def __init__(self, nb_in: int, dropout: float = 0.1, max_length: int = 3001):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe          = torch.zeros(max_length, nb_in)
        position    = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term    = torch.exp(torch.arange(0, nb_in, 2).float() * (-torch.log(torch.tensor(10000.0)) / nb_in))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe          = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.FloatTensor):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, nb_in: int, reduced_dim: int = 16, max_length: int = 3001):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_length, reduced_dim)) 
        self.proj = nn.Linear(reduced_dim, nb_in)  # 低維 -> 原維度投影

    def forward(self, x: torch.FloatTensor):
        pos_emb = self.proj(self.pe[:, :x.size(1), :])  # 投影回 nb_in 維度
        return x + pos_emb


class SeparationNet(nn.Module):
    def __init__(
            self,
            feature_size: int,
            n_layer: int,
            n_head: int
        ):
        super().__init__()

        self.cross_attn     = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.pos_enc        = PositionalEmbedding(
            feature_size,
            max_length=6001
        )
        self.separation_net = Conformer(
            input_dim       = feature_size,
            ffn_dim         = 256,
            num_heads       = n_head,
            num_layers      = n_layer,
            dropout         = 0.1,
            use_group_norm  = True,
            depthwise_conv_kernel_size=31,
        )
        self.output_prelu   = torch.nn.PReLU()
        self.output_conv    = torch.nn.Conv1d(
            in_channels     = feature_size,
            out_channels    = feature_size,
            kernel_size     = 1,
        )
        self.out_act        = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        B, F, L = x.shape[0], x.shape[1], x.shape[2]

        out     = x.permute(0, 2, 1)
        out     = self.pos_enc(out)
        out, _  = self.cross_attn(out, query, query)
        lengths = torch.full((B, ), fill_value=L, dtype=torch.long).to(device=x.device)
        out, _  = self.separation_net(out, lengths)
        out     = out.permute(0, 2, 1)

        out     = self.output_prelu(out)
        out     = self.output_conv(out)
        out     = self.out_act(out)
        
        return out
