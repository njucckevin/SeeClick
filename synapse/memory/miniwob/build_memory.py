import logging
import json
import openai
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from collections import Counter

logger = logging.getLogger(__name__)

EXEMPLAR_LIST = [
    "book-flight",
    "choose-date",
    "click-button-sequence",
    "click-button",
    "click-checkboxes-large",
    "click-checkboxes-soft",
    "click-collapsible-2",
    "click-collapsible",
    "click-color",
    "click-dialog-2",
    "click-dialog",
    "click-link",
    "click-menu",
    "click-pie",
    "click-scroll-list",
    "click-shades",
    "click-shape",
    "click-tab-2",
    "click-tab",
    "click-widget",
    "copy-paste-2",
    "count-shape",
    "email-inbox-nl-turk",
    "enter-date",
    "enter-password",
    "enter-text-dynamic",
    "enter-time",
    "find-word",
    "focus-text-2",
    "focus-text",
    "grid-coordinate",
    "guess-number",
    "identify-shape",
    "login-user-popup",
    "multi-layouts",
    "navigate-tree",
    "read-table",
    "search-engine",
    "simple-algebra",
    "social-media-all",
    "social-media-some",
    "social-media",
    "terminal",
    "text-transform",
    "tic-tac-toe",
    "use-autocomplete",
    "use-slider",
    "use-spinner",
]


def build_memory(memory_path: str):
    with open(os.path.join(memory_path, "specifiers.json"), "r") as rf:
        specifier_dict = json.load(rf)
        exemplar_names = []
        specifiers = []
        for k, v in specifier_dict.items():
            assert k in EXEMPLAR_LIST
            for query in v:
                exemplar_names.append(k)
                specifiers.append(query)
    assert sorted(list(set(exemplar_names))) == sorted(EXEMPLAR_LIST)

    # embed memory_keys into VectorDB
    logger.info("Initilizing memory")
    openai.api_key = os.environ["OPENAI_API_KEY"]
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    metadatas = [{"name": name} for name in exemplar_names]
    memory = FAISS.from_texts(
        texts=specifiers,
        embedding=embedding,
        metadatas=metadatas,
    )
    memory.save_local(memory_path)


def retrieve_exemplar_name(memory, query: str, top_k) -> str:
    retriever = memory.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(query)
    retrieved_exemplar_names = [doc.metadata["name"] for doc in docs]
    logger.info(f"Retrieved exemplars: {retrieved_exemplar_names}")
    data = Counter(retrieved_exemplar_names)
    retrieved_exemplar_name = data.most_common(1)[0][0]

    return retrieved_exemplar_name


def load_memory(memory_path):
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    memory = FAISS.load_local(memory_path, embedding)

    return memory
