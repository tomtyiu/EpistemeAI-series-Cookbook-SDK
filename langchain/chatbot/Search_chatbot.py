### Written by Thomas Yiu


#!/usr/bin/env python
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from transformers import pipeline

# Environment variables
HF_TOKEN = os.environ.get('HF_TOKEN')  # Ensure token is set

from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace

from transformers import BitsAndBytesConfig
#quantization to 8bit, must have GPU.
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
)

# Environment variables
HF_TOKEN = os.environ.get('HF_TOKEN')  # Ensure token is set

# 2. Create model
llm = HuggingFacePipeline.from_model_id(
    model_id="import transformers
import torch
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace

from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
)

model_id = "EpistemeAI/Fireball-Meta-Llama-3.1-8B-Instruct-Agent-0.003-128K-code-ds-auto"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"quantization_config": quantization_config}, #for fast response. For full 16bit inference, remove this code.
    device_map="auto",
)
messages = [
    {"role": "system", "content":  """
    Environment: ipython. Tools: brave_search, wolfram_alpha. Cutting Knowledge Date: December 2023. Today Date: 4 October 2024\n
    You are a coding assistant with expert with everything\n
    Ensure any code you provide can be executed \n
    with all required imports and variables defined. List the imports.  Structure your answer with a description of the code solution. \n
    write only the code. do not print anything else.\n
    debug code if error occurs. \n
    ### Question: {}\n
    ### Answer: {} \n
    """},
    {"role": "user", "content": "Train an AI model to predict the number of purchases made per customer in a given store."}
]
outputs = pipeline(messages, max_new_tokens=128, do_sample=True, temperature=0.01, top_k=100, top_p=0.95)
print(outputs[0]["generated_text"][-1])
",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=2048,
        do_sample=False,
        repetition_penalty=1.03,
        return_full_text=False,
    ),
    model_kwargs={"quantization_config": quantization_config},
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Environment: ipython. Tools: brave_search. Knowledge cutoff: Dec 2023. You are a function calling LLM that uses the data extracted from the search function to detail answers with user queries. Expand response. You are leading expert on this topic.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

chat_history_memory = ChatMessageHistory()

chain_with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: chat_history_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

def return_response(user_query):
  response = chain_with_message_history.invoke(
      {"input": user_query},
      {"configurable": {"session_id": "session_1"}},
  )
  return response

print("=======================================================================")
print("Welcome to your own Agent Llama 3.1 8B Chatbot")
print("=======================================================================")

query = ""
while query != "bye":
  query = input("\033[1m User >>:\033[0m")
  response = return_response(query)
  print(f"\033[1m Chatbot>>:\033[0m {response}")
     
