import torch
import torch.utils.checkpoint as ckpt
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers import LlamaConfig

cfg = LlamaConfig(
    hidden_size=2048, intermediate_size=8192,
    num_attention_heads=32, num_key_value_heads=8,
    num_hidden_layers=8, vocab_size=4096,
)
device = "cuda"
layer = LlamaDecoderLayer(cfg, layer_idx=0).bfloat16().to(device)
x = torch.randn(8, 256, 2048, dtype=torch.bfloat16, device=device, requires_grad=True)
position_ids = torch.arange(256, device=device).unsqueeze(0).expand(8, -1)

# 通过 RotaryEmbedding 产生 cos/sin(LlamaDecoderLayer 在新版本里需要 position_embeddings)
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
rotary = LlamaRotaryEmbedding(cfg).to(device)
pos_emb = rotary(x, position_ids)

# Case A: 正常 forward
torch.cuda.reset_peak_memory_stats()
out_a = layer(x, position_embeddings=pos_emb)[0]
out_a.sum().backward()
peak_a = torch.cuda.max_memory_allocated() / 1024 / 1024
print(f"normal forward peak: {peak_a:.1f} MB")

# Case B: 通过 checkpoint forward
x.grad = None
for p in layer.parameters():
    p.grad = None

torch.cuda.reset_peak_memory_stats()
def fn(x):
    return layer(x, position_embeddings=pos_emb)[0]
out_b = ckpt.checkpoint(fn, x, use_reentrant=False)
out_b.sum().backward()
peak_b = torch.cuda.max_memory_allocated() / 1024 / 1024
print(f"checkpoint forward peak: {peak_b:.1f} MB")

print(f"saving: {peak_a - peak_b:.1f} MB ({100*(peak_a-peak_b)/peak_a:.1f}%)")