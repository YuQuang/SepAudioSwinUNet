import torch
import torch.nn as nn
from transformers import AlbertTokenizer, AlbertModel

class WordEmbbeding(nn.Module):
    def __init__(self, model_name: str="albert-base-v2"):
        super().__init__()
        self.tokenizer = AlbertTokenizer.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16
                        )
        self.model     = AlbertModel.from_pretrained(
                            model_name
                        )
    
    def forward(self, query: list[str]):
        with torch.no_grad():
            tokens  = self.tokenizer(query, padding=True, truncation=True, return_tensors="pt")
            if torch.cuda.is_available(): tokens = tokens.to("cuda")
            outputs = self.model(**tokens)
        return outputs.last_hidden_state[:, :, :]
    
if __name__ == "__main__":
    model = WordEmbbeding()
    out = model([
        "Hello World. It's a beautiful day.",
        "Hello There"
    ])
    print(out.shape)