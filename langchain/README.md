# For LangServe - serve.py, please follow: 

For both client and server:

```shell
pip install "langserve[all]"
```

or pip install "langserve[client]" for client code, and pip install "langserve[server]" for server code.

for more information of [LangServe](https://python.langchain.com/docs/langserve/)

## For Agent python file:

```shell
!pip install --upgrade --quiet  langchain-huggingface text-generation transformers google-search-results numexpr langchainhub sentencepiece jinja2 bitsandbytes accelerate

!pip install langchain_community
```

## Agent-llama.py (Ollama)
### please download [Huggingface gguf Q8_0](https://huggingface.co/legolasyiu/Fireball-Meta-Llama-3.1-8B-Instruct-Agent-0.003-128K-code-Q8_0-GGUF/blob/main/fireball-meta-llama-3.1-8b-instruct-agent-0.003-128k-code-q8_0.gguf)
```shell
ollama run Fireball-Meta-Llama-3.1-8B-Instruct-Agent-0.003-128K-code:latest 
pip install -U langchain-ollama
```


```shell
pip install langchain-ollama
```
