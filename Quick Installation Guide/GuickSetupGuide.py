#please install 
#!pip install -q --upgrade torch
#
#!pip install -q transformers triton==3.4 kernels
#
#!pip uninstall -q torchvision torchaudio -y

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "EpistemeAI/Episteme-gptoss-20b-RL"
#Other models: 
# EpistemeAI/VibeCoder-20b-RL1_0
# EpistemeAI/VCoder-120b-1.0

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="cuda",
)

messages = [
    {"role": "system", "content": "Respond with vibe coding"},
    {"role": "user", "content": "How many r in strawberries"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="high",
).to(model.device)

generated = model.generate(**inputs, max_new_tokens=500)
print(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:]))


messages = [
    {"role": "system", "content": "Always respond in riddles"},
    {"role": "user", "content": "What is the weather like in Madrid?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

generated = model.generate(**inputs, max_new_tokens=500)
print(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:]))
