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



```shell
pip install langchain-ollama
```
 
## Latest models:
**Episteme-gptoss-20b-RL** model
```shell
ollama run hf.co/mradermacher/Episteme-gptoss-20b-RL-GGUF:MXFP4_MOE
```

**VibeCoder-20b-RL1_0** model
```shell
ollama run hf.co/mradermacher/VibeCoder-20b-RL1_0-GGUF:Q8_0
```
