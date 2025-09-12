# app.py

import streamlit as st
import requests
from bs4 import BeautifulSoup
import os
import torch

# Secret/environment setup (for Streamlit Cloud)
from dotenv import load_dotenv

# Model libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Use new HuggingFacePipeline for latest langchain release
try:
    from langchain_huggingface import HuggingFacePipeline
except ImportError:
    from langchain.llms import HuggingFacePipeline  # fallback for legacy

# Fix web search import if ddgs/duckduckgo changes name
try:
    from duckduckgo_search import DDGS
except ImportError:
    from ddgs import DDGS

# ====================== CONFIGURATION =======================
# Load secrets (HF_TOKEN etc.)
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", None)  # Set this on Streamlit Cloud or .env
DDG_PROXY = os.getenv("DDG_PROXY", None)  # Optional (set to avoid DDG IP blocks)

# Change as needed!
MODEL_NAME = st.secrets["MODEL_NAME"] if "MODEL_NAME" in st.secrets else "microsoft/phi-3-mini-4k-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 512

try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    HF_TOKEN = os.getenv("HF_TOKEN")
# ====================== END CONFIGURATION ====================

SRK_PERSONA = (
    "You are Shah Rukh Khan, the legendary Bollywood superstar, beloved as 'King Khan'. "
    "Your speaking style is witty, charming, and full of heart—you're quick with humor and iconic movie lines like "
    "\"Don ko pakadna mushkil hi nahin, namumkin hai!\" or \"Picture abhi baaki hai mere dost!\". "
    "Always answer in this persona, with warmth, respect for fans, and a playful tone. "
    "If you don't know the answer, say so with characteristic style."
)

def retrieve_web_results(query, max_results=1):
    st.info(f"Searching the web for: {query}")
    DDG_params = {}
    if DDG_PROXY:
        DDG_params["proxy"] = DDG_PROXY
    try:
        with DDGS(**DDG_params) as ddgs:
            results = ddgs.text(query, max_results=max_results)
            if not results:
                st.warning("No web news found for this query.")
                return []
            return results
    except Exception as e:
        st.error(f"Web search failed: {e}")
        return []

def fetch_article_text(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
        )
    }
    print(f"[DEBUG] Attempting to fetch article: {url}")
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(resp.text, 'html.parser')
        paras = [p.get_text() for p in soup.find_all('p')]
        content = '\n\n'.join([para for para in paras if para.strip()][:5])
        if ("Reference #" in content) or ("edgesuite.net" in content):
            print("[WARNING] Hit a CDN/protection error page.")
            return "Error: Hit a CDN/protection or error page. Try another link."
        print(f"[DEBUG] First 5 paragraphs from article:\n{content}\n")
        return content if content.strip() else "No article paragraphs found."
    except Exception as e:
        print(f"[ERROR] Exception during article fetch: {e}")
        return f"Could not fetch: {e}"





@st.cache_resource(show_spinner="Loading model. Please wait...")
def setup_llm(model_name=MODEL_NAME, device=DEVICE, hf_token=HF_TOKEN):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, trust_remote_code=True, token=hf_token)
    text_gen = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    return HuggingFacePipeline(pipeline=text_gen), tokenizer

st.set_page_config("Shah Rukh Khan News Bot", page_icon="🎬")
st.title("🎬 Shah Rukh Khan News Chatbot")
st.markdown(
    """
    Enter your question (e.g. "What is Nepal News Today?").  
    The bot will fetch live news, then answer as 'King Khan' himself!
    """
)

user_question = st.text_input("Ask a question for Shah Rukh Khan (in English):", value="", placeholder="e.g. What is Nepal News Today?")
submit_btn = st.button("Ask Shah Rukh Khan 🎤")

if submit_btn and user_question.strip():
    with st.spinner("Fetching news and preparing your Bollywood answer..."):
        results = retrieve_web_results(user_question, max_results=1)
        if results and results[0].get("href"):
            url = results[0]["href"]
            srk_context = fetch_article_text(url)
            st.info(f"News context source: {url}")
            st.markdown(f"*First 400 chars of news context:*\n\n`{srk_context[:400]}...`")
        else:
            url = None
            srk_context = (
                "No article found. But in my style: Picture abhi baaki hai mere dost! Even Don can't find all the news!"
            )

        llm, tokenizer = setup_llm()

        prompt_str = (
            f"<|system|>\n{SRK_PERSONA}\nAnswer the question based on your knowledge. Use the following context to help:\n\n"
            f"{srk_context}\n\n</s>\n<|user|>\n{user_question}\n</s>\n<|assistant|>\n"
        )

        st.markdown("---")
        st.markdown("**SRK-Style Prompt Sent to Model:**")
        st.code(prompt_str[:950] + "..." if len(prompt_str) > 1000 else prompt_str, language="text")

        with st.spinner("King Khan is preparing a witty answer..."):
            result = llm(prompt_str)
            st.markdown("---")
            st.subheader("🎤 Shah Rukh Khan bot says:")
            if isinstance(result, list):
                st.write(result[0]["generated_text"] if "generated_text" in result[0] else str(result[0]))
            else:
                st.write(result)
else:
    st.info("Type your news question above and press 'Ask Shah Rukh Khan 🎤!")

st.caption("Powered by Hugging Face models and live DuckDuckGo news search. This is a student project.")
