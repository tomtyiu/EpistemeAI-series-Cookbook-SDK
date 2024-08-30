# Fireball-series
Fireball-12B and others model cookbook

<img src="https://huggingface.co/EpistemeAI/Fireball-Mistral-Nemo-Base-2407-v1-DPO2/resolve/main/fireball.JPG" width="200"/>
<a href="https://ko-fi.com/epistemeai">>>Please support and donate<<</a>

# Fireball-12B
This model is super fine-tune to provide better coding and better response(from first fine-tune) than Llama-3.1-8B and Google Gemma 2 9B. 
Further fine tuned with ORPO method with dataset 
- reciperesearch/dolphin-sft-v0.1-preference

# Benchmark
<img src="https://huggingface.co/EpistemeAI/Fireball-12B/resolve/main/benchmark2.jpg"/>
## Training Dataset 
Supervised fine-tuning with dataset: 
- candenizkocak/code-alpaca-297k
- yahma/alpaca-cleaned

# Model Card for Fireball-12B

The Heavy fine-tuned Mistral-Nemo-Base-2407 Large Language Model (LLM) is a pretrained generative text model of 12B parameters trained jointly by Mistral AI and NVIDIA, it significantly outperforms existing models smaller or similar in size.

For more details about this model please refer to our release [blog post](https://mistral.ai/news/mistral-nemo/).

## Key features
- Released under the **Apache 2 License**
- Pre-trained and instructed versions
- Trained with a **128k context window**
- Trained on a large proportion of **multilingual and code data**
- Drop-in replacement of Mistral 7B

## Model Architecture
Mistral Nemo is a transformer model, with the following architecture choices:
- **Layers:** 40
- **Dim:** 5,120
- **Head dim:** 128
- **Hidden dim:** 14,436
- **Activation Function:** SwiGLU
- **Number of heads:** 32
- **Number of kv-heads:** 8 (GQA)
- **Vocabulary size:** 2**17 ~= 128k
- **Rotary embeddings (theta = 1M)**

# Guardrail/Moderation guide: 
For guardrailing and moderating prompts against indirect/direct prompt injections and jailbreaking, please follow the SentinelShield AI GitHub repository:
[SentinelShield AI](https://github.com/tomtyiu/SentinelShieldAI)


#### Demo

After installing `mistral_inference`, a `mistral-demo` CLI command should be available in your environment.

### Transformers

> [!IMPORTANT]
> NOTE: Until a new release has been made, you need to install transformers from source:
> ```sh
> pip install mistral_inference
> pip install mistral-demo
> pip install git+https://github.com/huggingface/transformers.git
> ```
If you want to use Hugging Face `transformers` to generate text, you can do something like this.
```py
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "EpistemeAI/Fireball-12B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
inputs = tokenizer("Hello my name is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
## Accelerator mode: 
```py
pip install accelerate #GPU A100/L4
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
# Initialize the accelerator
accelerator = Accelerator()
# Define the model ID
model_id = "EpistemeAI/Fireball-12B"
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Load the model and prepare it for distributed setup using accelerate
model = AutoModelForCausalLM.from_pretrained(model_id)
# Move the model to the appropriate device using accelerate
model, = accelerator.prepare(model)
# Prepare inputs
inputs = tokenizer("Hello my name is", return_tensors="pt").to(accelerator.device)
# Generate outputs with the model
outputs = model.generate(**inputs, max_new_tokens=20)
# Decode and print the outputs
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
> [!TIP]
> Unlike previous Mistral models, Mistral Nemo requires smaller temperatures. We recommend to use a temperature of 0.3.
## Note
`EpistemeAI/Fireball-12B` is a pretrained base model and therefore does not have any moderation mechanisms. Go to Guardrail/Moderation guide section for moderation guide
### Citation for yahma/alpaca-cleaned dataset
```
@misc{alpaca,
  author = {Rohan Taori and Ishaan Gulrajani and Tianyi Zhang and Yann Dubois and Xuechen Li and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto },
  title = {Stanford Alpaca: An Instruction-following LLaMA model},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tatsu-lab/stanford_alpaca}},
}
```
# Uploaded  model
- **Developed by:** EpistemeAI
- **License:** apache-2.0
- **Finetuned from model :** EpistemeAI/Fireball-Mistral-Nemo-Base-2407-sft-v2.2a
This mistral model was trained 2x faster with [Unsloth](https://github.com/unslothai/unsloth) and Huggingface's TRL library.
[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)


