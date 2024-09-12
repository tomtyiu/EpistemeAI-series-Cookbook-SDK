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


# Guardrail/Moderation guide: 
For guardrailing and moderating prompts against indirect/direct prompt injections and jailbreaking, please follow the SentinelShield AI GitHub repository:
[SentinelShield AI](https://github.com/tomtyiu/SentinelShieldAI)

# programming languages

### Langchain: Please go to langchain to use our Fireball series in langchain

### llamaindex: please go to llamaindex to use our Fireball series in llamaindex

### vllm: please go to vllm to uses our Fireball series in vllm

> [!TIP]
> Unlike previous Mistral models, Mistral Nemo requires smaller temperatures. We recommend to use a temperature of 0.3.
## Note
`EpistemeAI/Fireball-12B` is a pretrained base model and therefore does not have any moderation mechanisms. Go to Guardrail/Moderation guide section for moderation guide


