import torch.nn as nn

class FiLM(nn.Module):
    def __init__(self, feature_dim, num_heads=1, num_layers=1, d_model=384):
        """
        Args:
            feature_dim (int): Dimension of the feature map to be modulated.
            num_heads (int): Number of attention heads in Transformer.
            num_layers (int): Number of Transformer encoder layers.
            d_model (int): Hidden dimension of Transformer.
        """
        super(FiLM, self).__init__()

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Gamma 和 Beta 生成層
        self.gamma_fc = nn.Linear(d_model, feature_dim)
        self.beta_fc = nn.Linear(d_model, feature_dim)
    
    def forward(self, x, conditioning_input):
        """
        Args:
            x (Tensor): Feature map of shape (batch, length, feature_dim)
            conditioning_input (Tensor): Conditioning input of shape (batch, conditioning_dim)
        
        Returns:
            Tensor: Modulated feature map of shape (batch, length, feature_dim)
        """
        B, L, _ = x.shape  # (batch, seq_len, feature_dim)
        
        conditioning_input = conditioning_input.unsqueeze(1).expand(-1, L, -1)  # (batch, seq_len, d_model)

        encoded = self.transformer(conditioning_input)  # (batch, seq_len, d_model)

        gamma   = self.gamma_fc(encoded)  # (batch, seq_len, d_model)
        beta    = self.beta_fc(encoded)   # (batch, seq_len, d_model)
        
        return gamma * x + beta  # 逐元素縮放與偏移