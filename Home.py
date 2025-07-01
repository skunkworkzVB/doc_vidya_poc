import os
import utils
import requests
import traceback
import validators
import streamlit as st
from streaming import StreamHandler

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.embeddings import OpenAIEmbeddings

from langchain.prompts import (
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from utils import context_prompt

from langchain_community.vectorstores import FAISS

import faiss

from dotenv import load_dotenv

# Use load_dotenv only if running locally (not on Streamlit Cloud)
if not st.secrets.get("OPENAI_API_KEY"):
    load_dotenv()

st.set_page_config(
    page_title="Chat with Docvidya"
)
st.header("Docvidya")
st.text("Demo by Valuebound")


class ChatbotWeb:

    def __init__(self):
        utils.sync_st_session()
        self.llm = utils.configure_llm()
        self.embedding_model = utils.configure_embedding_model()

    def scrape_website(self, url):
        content = ""
        try:
            base_url = "https://r.jina.ai/"
            final_url = base_url + url
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:88.0) Gecko/20100101 Firefox/88.0"
            }
            response = requests.get(final_url, headers=headers)
            content = response.text
        except Exception as e:
            traceback.print_exc()
        return content

    @st.cache_resource(show_spinner="Loading VectorDB", ttl=3600)
    def setup_vectordb(_self):
        # Scrape and load documents
        # Use st.secrets for API key if available, else fallback to env
        openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        embed = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectordb = FAISS.load_local(
            "assets/faiss_index",
            embed,
            allow_dangerous_deserialization=True,
        )
        return vectordb

    def setup_qa_chain(self, vectordb):

        # Define retriever
        retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 10},
        )

        # Setup memory for contextual conversation
        memory = ConversationBufferMemory(
            memory_key="chat_history", output_key="answer", return_messages=True
        )

        # Added context
        messages = [
            SystemMessagePromptTemplate.from_template(context_prompt + " {context}"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
        prompt = ChatPromptTemplate.from_messages(messages=messages)

        # Setup QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": prompt},
        )
        return qa_chain

    @utils.enable_chat_history
    def main(self):
        # User Inputs

        vectordb = self.setup_vectordb()
        qa_chain = self.setup_qa_chain(vectordb)

        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:

            utils.display_msg(user_query, "user")

            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                result = qa_chain.invoke(
                    {
                        "question": user_query,
                        "chat_history": st.session_state.messages,
                    },
                    {"callbacks": [st_cb]},
                )
                response = result["answer"]
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                utils.print_qa(ChatbotWeb, user_query, response)

                # to show references
                for idx, doc in enumerate(result["source_documents"], 1):
                    url = doc.metadata["article_url"]
                    ref_title = f":blue[Reference {idx}: *{url}*]"
                    with st.popover(ref_title):
                        st.caption(doc.page_content)


if __name__ == "__main__":
    obj = ChatbotWeb()
    obj.main()
