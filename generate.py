from osc_transformers import registry, Config, Tokenizer
from lightning import Fabric
import torch
import time
import sys

device = 'cuda:0'
fabric = Fabric(devices=[0], accelerator='gpu', precision='bf16-true')
ckpt = fabric.load('./chinese-alpaca-2-7B.ckpt')

with fabric.init_module(empty_init=True):
    config = Config(ckpt['config'])
    llm = registry.resolve(config)['model']
llm.load_state_dict(ckpt['model'])
llm.build_caches(batch_size=1, device=device)

tokenizer = Tokenizer('./checkpoints/chinese-alpaca-2-7b')

b_inst, e_inst = "[INST]", "[/INST]"
b_sys, e_sys = "<<SYS>>\n", "\n<</SYS>>\n\n"
system_prompt = (
    f"{b_inst} {b_sys}You are a helpful assistant, 你是一个乐于助人的助手.{e_sys} {{prompt}} {e_inst} "
)
prompt = system_prompt.format(prompt="帮我写首500字左右的作文")

max_length = 512
k: int = 1
temperature: float = 1.0
input_ids = tokenizer.encode(prompt).unsqueeze(0).to(device)
input_pos = torch.arange(0, input_ids.shape[1], device=device)
pred_tokens = []
with torch.inference_mode():
    start = time.perf_counter()
    for i in range(max_length):
        logits = llm(input_ids, input_pos)[:, -1, :]
        # 将topk之外的置为-inf
        top_k_logits, top_k_indices = torch.topk(logits, k=k)
        logits = torch.full_like(logits, -float("inf")).scatter_(-1, index=top_k_indices, src=top_k_logits)
        probs = torch.softmax(logits / temperature, dim=-1)
        # 在最大的k个概率中随机采样
        res = torch.multinomial(probs, num_samples=1)
        input_ids = res
        pred_token = res[0]
        pred_tokens.append(pred_token)
        fabric.print(tokenizer.decode(pred_token), flush=True, end='')
        if pred_token == tokenizer.eos_id:
            break
        input_pos = input_pos[-1:] + 1
    end = time.perf_counter()
spent = end - start
fabric.print(f'\n tokens per second: {len(pred_tokens) / spent}')
fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)