from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
    

#HUGGINGFACEHUB_API_TOKEN = os.environ["HF_TOKEN"]

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)




python_repl = PythonREPL()
    # You can create the tool to pass to an agent
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)
"""
llm = HuggingFacePipeline.from_model_id(
        model_id="EpistemeAI/Fireball-Meta-Llama-3.1-8B-Instruct-Agent-0.003-128K-code",
        task="text-generation",
        device=-1,
        pipeline_kwargs=dict(
            max_new_tokens=128000,
            do_sample=False,
            repetition_penalty=1.03,
            return_full_text=False,
        ),
        #model_kwargs={"quantization_config": quantization_config},
    )
chat_model = ChatHuggingFace(llm=llm)
"""
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="Fireball-Meta-Llama-3.1-8B-Instruct-Agent-0.003-128K-code:latest",
    temperature=0.6,
    max_new_tokens=128000,
    # other params...
)
    
def chatbot(query):
    messages = [
        SystemMessage(content=
        """
        Environment: ipython. Tools: brave_search, wolfram_alpha. Cutting Knowledge Date: December 2023. Today Date: 4 October 2024\n
        You are a coding assistant with expert with everything\n
        Ensure any code you provide can be executed \n
        with all required imports and variables defined. List the imports.  Structure your answer with a description of the code solution. \n
        write only the code. do not print anything else.\n
        use ipython for search tool. \n
        debug code if erorr occurs. \n
        Here is the user question: {question}
        """
        ),
        HumanMessage(
            content=query
        ),
    ]

    ai_msg = llm.invoke(messages)
    print(ai_msg.content)
    repl_tool(ai_msg.content)
    messages_2 = [
        SystemMessage(content=
        """
        Environment: ipython. Tools: brave_search, wolfram_alpha. Cutting Knowledge Date: December 2023. Today Date: 24 Auguest 2024\n
        You are a debug assistant. find bugs and fix the bugs\n
        Ensure any code you provide can be executed \n
        with all required imports and variables defined. List the imports.  Structure your answer with a description of the code solution. \n
        write only the code. do not print anything else.\n
        use ipython for search tool. \n
        debug code if error occurs. \n
        if no error, please provide no code
        if error, fix the code and provide the code
        """
        ),
        HumanMessage(
            content=ai_msg.content+"\n"+(repl_tool(ai_msg.content))
        ),
    ]
    ai_msg = llm.invoke(messages_2)
    repl_tool(ai_msg.content)
    
print("=======================================================================")
print("Welcome to your own Agent Llama 3.1 8B Chatbot")
print("=======================================================================")

query = ""
print("Ensure to Login to Huggingface for Huggingface models!!!")
while query != "bye":
  query = input("USER>>")
  chatbot(query)
