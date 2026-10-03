"""Smoke test: extract residual-stream activations from Llama-3.1-8B-Instruct with HF transformers."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
M = "meta-llama/Llama-3.1-8B-Instruct"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda")
msgs = [{"role": "user", "content": "Who had a 70s No 1 hit with Kiss You All Over?"}, {"role": "assistant", "content": "Exile."}]
ids = tok.apply_chat_template(msgs, return_tensors="pt", return_dict=True).to("cuda")
with torch.no_grad():
    out = model(**ids, output_hidden_states=True)
print("n_hidden_states", len(out.hidden_states), "shape", tuple(out.hidden_states[12].shape))
