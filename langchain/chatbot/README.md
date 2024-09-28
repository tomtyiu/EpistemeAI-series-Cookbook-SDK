# Fireball-series
Fireball search and tooling chatbot

# How to install: 

Install langchain-huggingface, google search and bitsandbytes packages
```shell
!pip install --upgrade --quiet  langchain-huggingface text-generation transformers google-search-results numexpr langchainhub sentencepiece jinja2 bitsandbytes accelerate langchain_community
```

please go to https://huggingface.co/ to get HF Token

```python
HF_TOKEN = os.environ.get('HF_TOKEN')  # Ensure token is set
```

Please use model to unlock build-in tooling: 

```python
model_id="EpistemeAI/Fireball-Meta-Llama-3.1-8B-Instruct-Agent-0.003"
```

to run charbot:
```python
python Search_chatbot.py
```


