from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import pickle
import os
from dotenv import load_dotenv
import json
import pandas as pd

load_dotenv()

csv = "Data-Doc Vidya - Sheet1.csv"

df = pd.read_csv(csv)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=500)
docs, metadatas = [], []
for index, row in df.iterrows():
    with open(row['text_path'], "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    splits = text_splitter.split_text(text)
    docs.extend(splits)
    metadatas.extend([{"article_url": row['url']}] * len(splits))
    print(f"Split {row['url']} into {len(splits)} chunks")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


store_new = FAISS.from_texts(docs, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY), metadatas=metadatas)

store_new.save_local('assets/faiss_index')