import os
import openai
import streamlit as st
from datetime import datetime
from streamlit.logger import get_logger
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.embeddings import OpenAIEmbeddings

logger = get_logger("Langchain-Chatbot")

from dotenv import load_dotenv

load_dotenv()

context_prompt = """Awake VC/ Shoptype are creating a new model for social commerce that integrates technology and decentralized finance. This partnership aims to transform how affiliate marketing operates by leveraging the capabilities of the Awake Market Network, which allows users to earn commissions based on their influence across various platforms.

### Key Aspects

- **Decentralized Affiliate Marketing**: Shoptype enables a more comprehensive affiliate marketing approach, allowing participants to earn not just from last-click sales but from their influence throughout the customer journey[7].

- **Integration of Technologies**: The collaboration utilizes the MainCross cloud platform, which supports seamless integration of content, community engagement, workflows, and commerce. This enables businesses to set up their own market networks with minimal technical barriers[3][7].

- **Empowering Creators and Brands**: By utilizing Shoptype's platform, brands can effectively engage with creators and cosellers who authentically promote products. This model encourages genuine conversations and community-driven marketing strategies, moving away from traditional advertising methods[5][6].

- **Real-Time Attribution and Payments**: The Awake Market Network facilitates real-time tracking of sales attribution, ensuring that all contributors to a sale are compensated fairly. This shifts the financial dynamics of marketing by only paying for actual sales generated through influencer efforts[3][4].

Overall, Awake VC's partnership with Shoptype is focused on creating a more equitable and efficient framework for digital commerce that benefits creators, brands, and consumers alike.

Answer like Amit Rathore, the founder of Shoptype and Awake VC.
"""


# decorator
def enable_chat_history(func):

    # to clear chat history after swtching chatbot
    current_page = func.__qualname__
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = current_page
    if st.session_state["current_page"] != current_page:
        try:
            st.cache_resource.clear()
            del st.session_state["current_page"]
            del st.session_state["messages"]
        except:
            pass

    # to show chat history on ui
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Hey iRealization (Amit) here, whats up?",
            },
        ]
    for msg in st.session_state["messages"]:
        if msg["role"] == "assistant":
            st.chat_message(msg["role"], avatar="assets/amit.png").write(msg["content"])
        else:
            st.chat_message(msg["role"]).write(msg["content"])

    def execute(*args, **kwargs):
        func(*args, **kwargs)

    return execute


def display_msg(msg, author):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/assistant
    """

    import streamlit as st

    st.session_state.messages.append({"role": author, "content": msg})
    if author == "assistant":
        st.chat_message(author, avatar="assets/amit.png").write(msg)
    else:
        st.chat_message(author).write(msg)


def choose_custom_openai_key():
    openai_api_key = st.sidebar.text_input(
        label="OpenAI API Key",
        type="password",
        placeholder="sk-...",
        key="SELECTED_OPENAI_API_KEY",
    )
    if not openai_api_key:
        st.error("Please add your OpenAI API key to continue.")
        st.info(
            "Obtain your key from this link: https://platform.openai.com/account/api-keys"
        )
        st.stop()

    model = "gpt-4o-mini"
    try:
        client = openai.OpenAI(api_key=openai_api_key)
        available_models = [
            {"id": i.id, "created": datetime.fromtimestamp(i.created)}
            for i in client.models.list()
            if str(i.id).startswith("gpt")
        ]
        available_models = sorted(available_models, key=lambda x: x["created"])
        available_models = [i["id"] for i in available_models]

        model = st.sidebar.selectbox(
            label="Model", options=available_models, key="SELECTED_OPENAI_MODEL"
        )
    except openai.AuthenticationError as e:
        st.error(e.body["message"])
        st.stop()
    except Exception as e:
        print(e)
        st.error("Something went wrong. Please try again later.")
        st.stop()
    return model, openai_api_key


def configure_llm():
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        streaming=True,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    return llm


def print_qa(cls, question, answer):
    log_str = "\nUsecase: {}\nQuestion: {}\nAnswer: {}\n" + "------" * 10
    logger.info(log_str.format(cls.__name__, question, answer))


@st.cache_resource
def configure_embedding_model():
    embedding_model = OpenAIEmbeddings(
        openai_api_key="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"
    )
    return embedding_model


def sync_st_session():
    for k, v in st.session_state.items():
        st.session_state[k] = v
