from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "checkpoints/qwen2.5-1.5b-sft/final_sft_checkpoint",
    dtype=torch.bfloat16,
)

# 1. Confirm the tie actually holds in the loaded checkpoint
same_tensor = model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
print("Tied:", same_tensor)
print("Values match:", torch.equal(model.lm_head.weight, model.model.embed_tokens.weight))

# 2. Coherence sanity check -- garbage head would produce garbage text
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("checkpoints/qwen2.5-1.5b-sft/final_sft_checkpoint")
prompt = "### Instruction:\nWhat is the capital of France?\n\n### Response:\n"
ids = tok(prompt, return_tensors="pt").input_ids
out = model.generate(ids, max_new_tokens=30)
print(tok.decode(out[0], skip_special_tokens=True))
