import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

class SentenceEmbbeding(nn.Module):
    def __init__(self, last_n_layers=2):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # 先凍結所有參數
        for param in self.model.parameters():
            param.requires_grad = False

        # 只解凍最後 last_n_layers 層
        encoder_layers = self.model[0].auto_model.encoder.layer
        for layer in encoder_layers[-last_n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
    
    def forward(self, x: list[str]) -> torch.Tensor:
        out = torch.from_numpy(self.model.encode(x)).to(self.device)
        return out
     
    def save_model(self, path="./temp/all_MiniLM_L6_v2_SBERT.pth"):
        """ 儲存 fine-tuned 的權重 """
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path="./temp/all_MiniLM_L6_v2_SBERT.pth"):
        """ 載入已訓練好的權重 """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")

if __name__ == "__main__":
    pass