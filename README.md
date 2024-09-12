# Fireball-series
Fireball-12B, Fireball-Llama3.1-8B series and others model cookbook

<img src="https://huggingface.co/EpistemeAI/Fireball-Mistral-Nemo-Base-2407-v1-DPO2/resolve/main/fireball.JPG" width="200"/>
<a href="https://ko-fi.com/epistemeai">>>Please support and donate<<</a>

# Fireball-12B
This model is super fine-tune to provide better coding and better response(from first fine-tune) than Llama-3.1-8B and Google Gemma 2 9B. 
Further fine tuned with ORPO method with dataset 
- reciperesearch/dolphin-sft-v0.1-preference

# Fireball-Llama3.1-8B
Model is fine tuned using sft technique.  

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

# programming languages

### Langchain: Please go to langchain to use our Fireball series in langchain

### llamaindex: please go to llamaindex to use our Fireball series in llamaindex

### vllm: please go to vllm to uses our Fireball series in vllm

```
> [!TIP]
> Unlike previous Mistral models, Mistral Nemo requires smaller temperatures. We recommend to use a temperature of 0.3.
## Note
`EpistemeAI/Fireball-12B` is a pretrained base model and therefore does not have any moderation mechanisms. Go to Guardrail/Moderation guide section for moderation guide

```


