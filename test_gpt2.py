from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "./gpt2_shoolini"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

prompt = input("Prompt: ")
inputs = tokenizer(prompt, return_tensors="pt").to(device)
out = model.generate(**inputs, max_new_tokens=80, do_sample=True, top_p=0.9, temperature=0.8)
print(tokenizer.decode(out[0], skip_special_tokens=True))
