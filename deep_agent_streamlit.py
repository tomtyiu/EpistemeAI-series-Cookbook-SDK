import os
import streamlit as st
from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langgraph.store.memory import InMemoryStore

# Page config
st.set_page_config(
    page_title="Deep Agent Researcher",
    page_icon="🔍",
    layout="wide"
)

# Title
st.title("🔍 Deep Agent Researcher")
st.markdown("An AI-powered research assistant using deep agents and web search")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    tavily_api_key = st.text_input(
        "Tavily API Key",
        type="password",
        value=os.environ.get("TAVILY_API_KEY", ""),
        help="Enter your Tavily API key for web search"
    )
    
    # Search parameters
    st.subheader("Search Parameters")
    max_results = st.slider("Max Results", 1, 10, 5)
    topic = st.selectbox("Topic", ["general", "news", "finance"])
    include_raw_content = st.checkbox("Include Raw Content", value=False)
    
    st.divider()
    st.markdown("### About")
    st.markdown("This app uses a deep agent with web search capabilities to conduct thorough research.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "store" not in st.session_state:
    st.session_state.store = InMemoryStore()

# Initialize agent
@st.cache_resource
def initialize_agent(_store, api_key):
    """Initialize the deep agent with all required components"""
    
    # Set API key
    os.environ["TAVILY_API_KEY"] = api_key
    tavily_client = TavilyClient(api_key=api_key)
    
    # Web search tool
    def internet_search(
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = False,
    ):
        """Run a web search"""
        return tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
    
    # System prompt
    research_instructions = """You are an expert code researcher. Your job is to conduct thorough research, and then write a polished report.

You have access to an internet search tool as your primary means of gathering information.

## `internet_search`

Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included."""
    
    # Initialize LLM
    llm = HuggingFacePipeline.from_model_id(
        model_id="EpistemeAI/Episteme-gptoss-20b-RL",
        task="text-generation",
        device_map="cuda",
        pipeline_kwargs=dict(
            max_new_tokens=4048,
            do_sample=False,
            repetition_penalty=1.03,
        ),
    )
    
    model = ChatHuggingFace(llm=llm)
    
    # Create agent
    agent = create_deep_agent(
        store=_store,
        use_longterm_memory=True,
        model=model,
        tools=[internet_search],
        system_prompt=research_instructions,
    )
    
    return agent

# Check if API key is provided
if not tavily_api_key:
    st.warning("⚠️ Please enter your Tavily API key in the sidebar to continue.")
    st.stop()

# Initialize agent if not already done
try:
    if st.session_state.agent is None:
        with st.spinner("Initializing agent... This may take a few minutes."):
            st.session_state.agent = initialize_agent(
                st.session_state.store,
                tavily_api_key
            )
        st.success("✅ Agent initialized successfully!")
except Exception as e:
    st.error(f"❌ Error initializing agent: {str(e)}")
    st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Enter your research query..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Researching..."):
            try:
                result = st.session_state.agent.invoke({
                    "messages": [{"role": "user", "content": prompt}]
                })
                response = result["messages"][-1].content
                st.markdown(response)
                
                # Add assistant response to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
