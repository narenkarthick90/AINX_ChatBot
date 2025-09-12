# app.py

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline  # pip install -U langchain-huggingface
import torch
import streamlit as st
import os

# Load from secrets (Streamlit Cloud, or from .streamlit/secrets.toml if local)
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    HF_TOKEN = os.getenv("HF_TOKEN")


# ----------------------------
# CONFIG: Change model for phi-3, llama-3, or any GPT/chat-model
MODEL_NAME = "meta-llama/Llama-3.2-1B"     # swap to "meta-llama/Llama-3.2-1B" if needed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 1000
# ----------------------------

SRK_PERSONA = (
    "You are Shah Rukh Khan, the legendary Bollywood superstar, beloved as 'King Khan'. "
    "Your speaking style is witty, charming, and full of heart—you're quick with humor and iconic movie lines like "
    "\"Don ko pakadna mushkil hi nahin, namumkin hai!\" or \"Picture abhi baaki hai mere dost!\". "
    "Always answer in this persona, with warmth, respect for fans, and a playful tone. "
    "If you don't know the answer, say so with characteristic style."
)

def retrieve_web_results(query, max_results=1):
    print(f"[DEBUG] Performing DuckDuckGo search for: '{query}' ...")
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
        if not results:
            print("[ERROR] No search results returned from DuckDuckGo.")
            return []
        for i, r in enumerate(results):
            print(f"[DEBUG] Result {i+1}: Title: {r['title']} Body: {r['body']} Link: {r['href']}")
        return results

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

def setup_llm(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device=DEVICE, hf_token=HF_TOKEN):
    try:
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
    except Exception as e:
        import sys, traceback
        print("[ERROR] Model loading failed. See details below:\n")
        traceback.print_exc()
        print("\nTIP: Do you have enough RAM? Are you using a gated model? Did you set a correct HuggingFace token? Try using TinyLlama-1.1B-Chat or phi-3-mini-4k-instruct, which are free.")
        sys.exit(1)


PROMPT_TEMPLATE = """
<|system|>
{srk_description}
Answer the question based on your knowledge. Use the following context to help:

{context}

</s>
<|user|>
{question}
</s>
<|assistant|>
"""

def main():
    user_question = "What is Nepal News Today?"
    results = retrieve_web_results(user_question, max_results=1)
    if results and results[0].get("href"):
        url = results[0]["href"]
        article_text = fetch_article_text(url)
    else:
        url = None
        article_text = (
            "No article found. But in my style: Picture abhi baaki hai mere dost! Even Don can't find all the news!"
        )
    print(f"\n[DEBUG] Using article_text as context for LLM:\n---\n{article_text}\n---\n")

    llm, tokenizer = setup_llm()

    prompt_str = (
        f"<|system|>\n{SRK_PERSONA}\nAnswer the question based on your knowledge. Use the following context to help:\n\n"
        f"{article_text}\n\n</s>\n<|user|>\n{user_question}\n</s>\n<|assistant|>\n"
    )
    print(f"[DEBUG] Final Prompt Sent to LLM:\n{prompt_str[:1000]}...\n")

    result = llm(prompt_str)
    print("\nShahrukh Khan bot:\n")
    if isinstance(result, list):
        print(result[0]["generated_text"] if "generated_text" in result[0] else str(result[0]))
    else:
        print(result)

if __name__ == "__main__":
    main()
