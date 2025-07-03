from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import pickle
import os
from dotenv import load_dotenv
import json

load_dotenv()

pages = [
    {"text_path": "assets/data/article_1.txt", "url": "https://www.docvidya.com/ketorol-dt-ache-relief-understanding-pain-levels-during-and-after-root-canal-treatment"},
    {"text_path": "assets/data/article_2.txt", "url": "https://www.docvidya.com/Ketorol-dt-pre-emptive-oral-analgesia-nsaids"},
    {"text_path": "assets/data/article_3.txt", "url": "https://www.docvidya.com/ketorol-dt-smiles-safe-hands-evaluating-dental-treatments-under-general-anesthesia-special-needs-and"},
    {"text_path": "assets/data/article_4.txt", "url": "https://www.docvidya.com/ketorol-dt-unlocking-pain-puzzle-key-predictors-discomfort-after-wisdom-tooth-extraction"},
    {"text_path": "assets/data/article_5.txt", "url": "https://www.docvidya.com/ketorol-dt-laser-your-pain-away-exploring-benefits-low-level-laser-therapy-root-canal-treatment"},
    {"text_path": "assets/data/article_6.txt", "url": "https://www.docvidya.com/ketorol-dt-ache-relief-understanding-pain-levels-during-and-after-root-canal-treatment"},
    {"text_path": "assets/data/article_7.txt", "url": "https://www.docvidya.com/Bro-zedex-LS-parent-child-unraveling-genetic-puzzle-chronic-cough?specialities=ent"},
    {"text_path": "assets/data/article_8.txt", "url": "https://www.docvidya.com/Bro-zedex-LS-safe-use-otc-cold-medications-children-risks-and-guidelines?specialities=ent"},
    {"text_path": "assets/data/article_9.txt", "url": "https://www.docvidya.com/Bro-zedex-LS-science-behind-levosalbutamol-precision-approach-respiratory-care?specialities=ent"},
    {"text_path": "assets/data/article_10.txt", "url": "https://www.docvidya.com/node/5606?specialities=ent"},
    {"text_path": "assets/data/article_11.txt", "url": "https://www.docvidya.com/node/5581?specialities=ent"},
    {"text_path": "assets/data/article_12.txt", "url": "https://www.docvidya.com/medshorts/oral-quadruple-therapy-offers-effective-safe-alternative-injectables-type-2-diabetes?specialities=ent"}
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=500)
docs, metadatas = [], []
for page in pages:
    with open(page['text_path'], "r") as f:
        text = f.read()
    splits = text_splitter.split_text(text)
    docs.extend(splits)
    metadatas.extend([{"article_url": page['url']}] * len(splits))
    print(f"Split {page['url']} into {len(splits)} chunks")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


store_new = FAISS.from_texts(docs, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY), metadatas=metadatas)

store_new.save_local('assets/faiss_index')