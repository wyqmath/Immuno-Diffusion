import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
from epigenetic_encoding import RealNVP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmbeddingProjector(nn.Module):
    def __init__(self, input_dim=512, output_dim=768):
        super().__init__()
        self.projector = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.projector(x)



def get_text_embeddings(prompts, tokenizer, text_encoder, device):
    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    with torch.no_grad():
        outputs = text_encoder(input_ids)
        embeddings = outputs.last_hidden_state[:, 0, :]  
    return embeddings

def load_prompts(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

def train():
    work_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file = os.path.join(work_dir, "generated_prompts.txt")
    prompts = load_prompts(prompt_file)
    batch_size = 4

    # 初始化CLIP文本编码器和tokenizer
    model_id = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    text_encoder = CLIPTextModel.from_pretrained(model_id).to(device)
    text_encoder.eval()  # 不训练文本编码器

    print(f"Text encoder hidden size: {text_encoder.config.hidden_size}")

    embedding_dim_in = text_encoder.config.hidden_size  # 512
    embedding_dim_out = 768
    hidden_dim = 256
    num_coupling_layers = 6

    projector = EmbeddingProjector(input_dim=embedding_dim_in, output_dim=embedding_dim_out).to(device)
    flow_model = RealNVP(dim=embedding_dim_out, hidden_dim=hidden_dim, num_coupling_layers=num_coupling_layers).to(device)
    optimizer = optim.Adam(list(projector.parameters()) + list(flow_model.parameters()), lr=1e-4)

    epochs = 20
    flow_model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            embeddings = get_text_embeddings(batch_prompts, tokenizer, text_encoder, device)  
            embeddings_proj = projector(embeddings)  # 映射到768维

            optimizer.zero_grad()
            
            z, log_det_jacobian = flow_model(embeddings_proj, reverse=False)
            # 计算标准正态分布的log概率
            log_prob = -0.5 * torch.sum(z ** 2, dim=1)
            loss = -torch.mean(log_prob + log_det_jacobian)  # 最大化似然 -> 最小化负对数似然
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_prompts)

        avg_loss = total_loss / len(prompts)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    # 保存权重到 checkpoints 文件夹
    save_dir = os.path.join(work_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    torch.save({
        'projector_state_dict': projector.state_dict(),
        'flow_model_state_dict': flow_model.state_dict()
    }, os.path.join(save_dir, "flow_model_weights.pth"))

    print(f"训练完成，权重已保存为 {save_dir}/flow_model_weights.pth")

if __name__ == "__main__":
    train()