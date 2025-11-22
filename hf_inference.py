from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "gpt2"  # or ANY small HF model you choose

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

ask = input("Enter a prompt: ")
out = pipe(ask, max_new_tokens=100)[0]["generated_text"]

print("\nModel Output:\n", out)
