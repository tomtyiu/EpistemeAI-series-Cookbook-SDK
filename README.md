# EpistemeAI Series Cookbook SDK

A practical cookbook for using EpistemeAI models from Hugging Face in local Python scripts, notebooks, vLLM servers, OpenAI-compatible clients, RAG pipelines, and evaluation workflows.

> **Hugging Face model catalog:** https://huggingface.co/EpistemeAI/models  
> **Organization:** EpistemeAI / EpisteLabs  
> **Last verified:** 2026-07-07

---

## What this repository is for

This repository provides quick-start recipes for:

- Loading EpistemeAI Hugging Face models with `transformers`
- Serving larger models with vLLM or SGLang
- Calling local models through the OpenAI-compatible API format
- Using text, image-text, embedding, coding, and medical reasoning models
- Running lightweight benchmark checks
- Applying safety guardrails for medical, biological, chemical, and agentic use cases

EpistemeAI models are experimental research models. Always check each Hugging Face model card for the latest license, access status, supported task type, and model-specific usage notes.

---

## Featured EpistemeAI model families

The full catalog contains many models. The table below highlights commonly useful families and recent models.

| Model / family | Typical task | Size / type | Suggested use |
|---|---:|---:|---|
| `EpistemeAI/Reasoning-Medical0.1-E4B-sft` | Medical reasoning, image-text-to-text | 8B class | Medical education, research QA, benchmark experiments |
| `EpistemeAI/Reasoning-Medical0.1-E4B-sft-Q8_0` | Quantized medical reasoning | 8B class | Lower-memory local inference |
| `EpistemeAI/Reasoning-Medical0.1-E4B-sft_lora` | LoRA adapter | Adapter | Further fine-tuning / adapter merging |
| `EpistemeAI/Reasoning-Medical0.1-27B` | Advanced medical reasoning, multimodal reasoning | 27B / 28B class | High-quality medical reasoning and image-text analysis |
| `EpistemeAI/Reasoning-Medical-27B` | Medical reasoning, multimodal reasoning | 27B / 28B class | Research copilots and benchmark comparisons |
| `EpistemeAI/Reason-Medical-20b-4bit` | Text-generation medical reasoning | 20B class, 4-bit | Lower-memory medical reasoning inference |
| `EpistemeAI/Reason-Medical-20b-16bit` | Text-generation medical reasoning | 20B class, 16-bit | Higher-fidelity local inference |
| `EpistemeAI/Reason-Medical-120b-16bit` | Large medical reasoning | 120B class | Multi-GPU / server inference experiments |
| `EpistemeAI/OpenMedResearch-Gemma-4E4N` | Open medical research assistant | 8B class | Literature review and medical research reasoning |
| `EpistemeAI/OpenMedResearch-Gemma-4E4N-8bit` | Quantized OpenMedResearch | 8B class | Memory-efficient medical research assistant |
| `EpistemeAI/ReasoningCore-3B-RE1-V2C` | General reasoning, summarization, dialogue | 3B | Fast local reasoning assistant |
| `EpistemeAI/ReasoningCore-1B-r1-0` | Very small reasoning model | 1B | CPU / small GPU testing |
| `EpistemeAI/rsi-gpt-oss-20b` | Reasoning / self-improvement experiments | 20B class | Research on RSI-style training loops |
| `EpistemeAI/rsi-gpt-oss-120bv2-8bit` | Large reasoning model | 120B class, 8-bit | Server-grade reasoning experiments |
| `EpistemeAI/VibeCoder-20b-RL1_0` | Code generation and coding assistant | 20B class | Coding, debugging, agentic software tasks |
| `EpistemeAI/Codeforce-metatune-gpt20b` | Competitive programming / code reasoning | 20B class | Algorithmic coding tasks |
| `EpistemeAI/EmbeddingsG300M-ft` | Sentence similarity / embeddings | 300M | RAG, semantic search, clustering |
| `EpistemeAI/LexiVox` | Text-to-speech | 3B | Speech generation experiments |
| Fireball series models | General instruction, math, code | 8B–12B class | Legacy cookbook examples and LangChain/LlamaIndex demos |

For the latest and complete list, use the Hugging Face catalog link above.

---

## Hardware guide

Approximate starting points only. Actual memory depends on context length, quantization, batch size, inference framework, and KV cache.

| Model size | Practical starting hardware |
|---:|---|
| 1B–3B | CPU, Apple Silicon, or 6–12 GB GPU |
| 8B | 12–24 GB GPU; quantized versions can run on less |
| 20B–27B | 24–80 GB GPU, preferably A100/H100/L40S class for long context |
| 120B | Multi-GPU server; use tensor parallelism and quantization where possible |

For large models, prefer vLLM, SGLang, TGI, or an optimized quantized runtime over plain eager `transformers`.

---

## Installation

Create a clean Python environment:

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -U torch transformers accelerate safetensors huggingface_hub
```

For gated models, log in to Hugging Face:

```bash
huggingface-cli login
```

Optional packages:

```bash
# OpenAI-compatible local API client
pip install -U openai

# Fast serving for larger models
pip install -U vllm

# Embedding recipes
pip install -U sentence-transformers faiss-cpu

# Evaluation
pip install -U lm-eval
```

---

## Recipe 1: quick text generation with Transformers

Use this for text-generation models such as ReasoningCore, VibeCoder, RSI, Codeforce, and text-only medical models.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "EpistemeAI/ReasoningCore-3B-RE1-V2C"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)

messages = [
    {
        "role": "system",
        "content": "You are a careful reasoning assistant. Be accurate, concise, and safe.",
    },
    {
        "role": "user",
        "content": "Explain the difference between retrieval and reasoning in an AI research assistant.",
    },
]

if hasattr(tokenizer, "apply_chat_template"):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
else:
    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )

new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
```

---

## Recipe 2: quick medical reasoning prompt

Use this pattern for medical education and research reasoning. Do **not** use model output as autonomous diagnosis or treatment.

```python
question = """
A student is reviewing chronic kidney changes after long-term urinary obstruction.
Explain the likely pathophysiology at a high level for medical education.
"""

messages = [
    {
        "role": "system",
        "content": (
            "You are a medical education assistant. "
            "Provide educational reasoning only. "
            "Do not provide diagnosis, treatment, or emergency instructions. "
            "Encourage clinician review where appropriate."
        ),
    },
    {"role": "user", "content": question},
]
```

Recommended safety footer for apps:

```text
This output is for medical education, research reasoning, literature review, and clinician-reviewed analysis only. It is not medical advice and must not be used for autonomous diagnosis or treatment.
```

---

## Recipe 3: image-text-to-text with a multimodal model

Use this for multimodal models such as `Reasoning-Medical0.1-27B`, `Reasoning-Medical0.1-E4B-sft`, or OpenMedResearch-style image-text models when supported by the model card.

```python
from transformers import pipeline

MODEL_ID = "EpistemeAI/Reasoning-Medical0.1-E4B-sft"

pipe = pipeline(
    task="image-text-to-text",
    model=MODEL_ID,
    device_map="auto",
    model_kwargs={"torch_dtype": "auto"},
    trust_remote_code=True,
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG",
            },
            {
                "type": "text",
                "text": "Describe the image in one sentence.",
            },
        ],
    }
]

result = pipe(text=messages, max_new_tokens=256)
print(result)
```

If this fails, check the exact model card because some multimodal models require a newer `transformers` version or a model-specific processor.

---

## Recipe 4: serve a model with vLLM

For 20B, 27B, and 120B class models, serving is usually better than loading directly in a notebook.

```bash
pip install -U vllm

vllm serve "EpistemeAI/Reasoning-Medical0.1-27B" \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code
```

Then call the local OpenAI-compatible API:

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
)

response = client.chat.completions.create(
    model="EpistemeAI/Reasoning-Medical0.1-27B",
    messages=[
        {
            "role": "system",
            "content": "You are a safe medical research assistant. Keep answers educational and require expert review.",
        },
        {
            "role": "user",
            "content": "Summarize the difference between sensitivity and specificity for medical students.",
        },
    ],
    max_tokens=1024,
    temperature=0.7,
    top_p=0.9,
)

print(response.choices[0].message.content)
```

---

## Recipe 5: OpenAI-compatible local client template

Use the same client structure for vLLM, SGLang, TGI, llama.cpp server, or any OpenAI-compatible local server.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

MODEL_ID = "EpistemeAI/ReasoningCore-3B-RE1-V2C"

messages = [
    {"role": "system", "content": "You are a concise research assistant."},
    {"role": "user", "content": "Create a 5-point checklist for evaluating an open-weight model."},
]

completion = client.chat.completions.create(
    model=MODEL_ID,
    messages=messages,
    temperature=0.7,
    top_p=0.9,
    max_tokens=700,
)

print(completion.choices[0].message.content)
```

---

## Recipe 6: embeddings and RAG with `EmbeddingsG300M-ft`

Use embeddings for semantic search, retrieval-augmented generation, clustering, duplicate detection, and ranking.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_ID = "EpistemeAI/EmbeddingsG300M-ft"

embedder = SentenceTransformer(MODEL_ID)

documents = [
    "Medical reasoning models should be validated by clinicians.",
    "RAG combines retrieval with generation.",
    "Benchmark scores do not guarantee real-world safety.",
]

query = "How should medical AI outputs be reviewed?"

doc_embeddings = embedder.encode(documents, normalize_embeddings=True)
query_embedding = embedder.encode([query], normalize_embeddings=True)[0]

scores = np.dot(doc_embeddings, query_embedding)
ranked = sorted(zip(scores, documents), reverse=True)

for score, doc in ranked:
    print(round(float(score), 4), doc)
```

---

## Recipe 7: LangChain wrapper around a local OpenAI-compatible server

```bash
pip install -U langchain-openai
```

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="EpistemeAI/ReasoningCore-3B-RE1-V2C",
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    temperature=0.7,
)

response = llm.invoke(
    "Write a concise explanation of why benchmark accuracy and deployment safety are different."
)

print(response.content)
```

---

## Recipe 8: LlamaIndex wrapper around a local OpenAI-compatible server

```bash
pip install -U llama-index llama-index-llms-openai-like
```

```python
from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(
    model="EpistemeAI/ReasoningCore-3B-RE1-V2C",
    api_base="http://localhost:8000/v1",
    api_key="EMPTY",
    is_chat_model=True,
)

response = llm.complete("Give a short explanation of agentic RAG.")
print(response)
```

---

## Recipe 9: evaluation with `lm-evaluation-harness`

Example for a general reasoning model:

```bash
lm_eval \
  --model hf \
  --model_args pretrained=EpistemeAI/ReasoningCore-3B-RE1-V2C,trust_remote_code=True \
  --tasks gsm8k \
  --num_fewshot 5 \
  --device cuda:0 \
  --batch_size auto
```

Example for medical QA experiments, depending on your local task registry:

```bash
lm_eval \
  --model hf \
  --model_args pretrained=EpistemeAI/Reasoning-Medical0.1-E4B-sft,trust_remote_code=True \
  --tasks medqa_4options \
  --num_fewshot 2 \
  --device cuda:0 \
  --batch_size auto
```

Evaluation notes:

- Keep prompts and chat templates consistent across models.
- Report `num_fewshot`, decoding parameters, exact task version, and seed.
- Do not compare benchmark scores unless the same evaluation harness and extraction rules were used.
- For medical models, benchmark performance is not proof of clinical safety.

---

## Recipe 10: basic Gradio chat app

```bash
pip install -U gradio transformers accelerate torch
```

```python
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "EpistemeAI/ReasoningCore-3B-RE1-V2C"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)

def chat(message, history):
    messages = [{"role": "system", "content": "You are a helpful and safe assistant."}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

demo = gr.ChatInterface(
    fn=chat,
    title="EpistemeAI Local Chat",
    description="Local chat demo for EpistemeAI Hugging Face models.",
    examples=[
        ["Explain retrieval-augmented generation in simple terms."],
        ["Give a safe, educational summary of specificity vs sensitivity."],
    ],
)

if __name__ == "__main__":
    demo.launch()
```

---

## Model selection guide

| Goal | Start with |
|---|---|
| Fast reasoning on local machine | `ReasoningCore-1B-r1-0` or `ReasoningCore-3B-RE1-V2C` |
| Medical education / research QA | `Reasoning-Medical0.1-E4B-sft` |
| Stronger medical reasoning | `Reasoning-Medical0.1-27B` or `Reasoning-Medical-27B` |
| Lower-memory medical inference | `Reason-Medical-20b-4bit` or Q8/Q4 quantized variants |
| Code generation | `VibeCoder-20b-RL1_0` or `Codeforce-metatune-gpt20b` |
| Semantic search / RAG | `EmbeddingsG300M-ft` |
| Experimental large-scale reasoning | `rsi-gpt-oss-120bv2-8bit` or `Reason-Medical-120b-16bit` |

---

## Safety and responsible use

### Medical safety

EpistemeAI medical models should be used for:

- Medical education
- Research reasoning
- Literature review
- Clinician-reviewed analysis
- Benchmarking and model evaluation

They should **not** be used for:

- Autonomous diagnosis
- Autonomous treatment recommendation
- Emergency medical triage without a qualified clinician
- Replacing a licensed medical professional
- Generating unsafe biological, chemical, or laboratory instructions

Recommended application disclaimer:

```text
This system is for medical education, research reasoning, literature review, and clinician-reviewed analysis only. It is not a medical device and must not be used for autonomous diagnosis, treatment, emergency triage, or patient-specific medical decision-making.
```

### Biosecurity and chemical safety

Do not use these models to assist with:

- Pathogen enhancement
- Biological weapon development
- Toxin production
- Hazardous synthesis
- Protocol optimization for harmful biological or chemical activity
- Any activity that violates law, biosafety, biosecurity, or chemical safety standards

### General AI safety

For production systems:

- Add input and output moderation
- Log prompts and model outputs for audit
- Use retrieval citations where possible
- Run red-team testing before deployment
- Add human review for high-impact domains
- Clearly disclose model limitations
- Validate outputs against trusted sources

---

## Common troubleshooting

### `OSError: gated repo`

Log in and accept the Hugging Face model terms:

```bash
huggingface-cli login
```

Then open the model page in a browser and accept access conditions if required.

### CUDA out of memory

Try:

- Smaller model
- Quantized model
- Lower `max_new_tokens`
- Shorter context
- `device_map="auto"`
- vLLM or SGLang serving
- 4-bit / 8-bit loading
- Tensor parallelism for very large models

### Chat output repeats or never stops

Set EOS token and reduce sampling aggressiveness:

```python
output_ids = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.6,
    top_p=0.9,
    repetition_penalty=1.05,
    eos_token_id=tokenizer.eos_token_id,
)
```

### Wrong task type

Check the model card. Some models are `text-generation`, some are `image-text-to-text`, and some are embeddings or text-to-speech models.

---

## Repository structure

Suggested cookbook layout:

```text
EpistemeAI-series-Cookbook-SDK/
├── README.md
├── Quick Installation Guide/
├── langchain/
├── llamaindex/
├── vllm/
├── gradio/
├── rag/
├── evaluation/
└── safety/
```

---

## Contributing

Contributions are welcome. Useful additions include:

- New quick-start notebooks
- vLLM / SGLang deployment examples
- Gradio demos
- LangChain and LlamaIndex integrations
- RAG examples using EpistemeAI embeddings
- Evaluation scripts
- Safety testing examples
- Quantized model usage notes

Please include:

- Model ID
- Library versions
- Hardware used
- Exact command or notebook
- Expected output
- Known limitations

---

## Support

- Hugging Face models: https://huggingface.co/EpistemeAI/models
- Cookbook repository: https://github.com/tomtyiu/EpistemeAI-series-Cookbook-SDK
- EpistemeAI contact listed on model cards when available

If you use the models in medical, scientific, or high-impact settings, add expert review and domain-specific validation before deployment.
