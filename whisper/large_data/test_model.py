from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_name = "meta-llama/Llama-2-7b-hf"   # or whatever base you used
adapter_path = "./llama_eln_prefix_lora"       # your fine-tuned LoRA

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, adapter_path)

model.eval()

prompt = """You are a transcription error correction assistant.
Hypotheses:
1. رومیزی را پهد کردن
2. رومیزی را پهت کردن
3. رومیزی را پهد کردن
4. رومیزی را پهد کردن
5. رومیزی را پهد کردن
Correct transcription:"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        temperature=0.7,
        top_p=0.9,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
