"""This module contains utility functions for the project"""

import mmh3
from haystack import Document


def get_unique_docs(dataset, unique_docs: set):
    """Get unique documents from dataset
    Args:
    dataset: list of dictionaries
    Returns:
    docs: list of haystack.Document
    """
    docs = list()
    for doc in dataset:
        if doc["context"] is not None and doc["context_id"] not in unique_docs:
            unique_docs.add(doc["context_id"])
            document = Document(
                content=doc["context"],
                meta={
                    "title": doc["context_title"],
                    "context_id": doc["context_id"],
                    "url": doc["url"],
                    "source": "QASports",
                },
            )
            docs.append(document)
    return docs


import streamlit as st
from datasets import load_dataset
from haystack import Pipeline
from haystack.components.readers import ExtractiveReader
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from utils import get_unique_docs


# Load the dataset
@st.cache_data(show_spinner=False)
def load_documents():
    """
    Load the documents from the dataset considering only unique documents.
    Returns:
    - documents: list of dictionaries with the documents.
    """
    unique_docs = set()
    dataset_name = "PedroCJardim/QASports"
    dataset_split = "basketball"
    st.caption(f'Fetching "{dataset_name}" dataset')
    # build the dataset
    dataset = load_dataset(dataset_name, dataset_split)
    docs_validation = get_unique_docs(dataset["validation"], unique_docs)
    docs_train = get_unique_docs(dataset["train"], unique_docs)
    docs_test = get_unique_docs(dataset["test"], unique_docs)
    documents = docs_validation + docs_train + docs_test
    return documents


@st.cache_resource(show_spinner=False)
def get_document_store(documents):
    """
    Index the files in the document store.
    Args:
    - files: list of dictionaries with the documents.
    """
    st.caption(f"Building the Document Store")
    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents=documents)
    return document_store


@st.cache_resource(show_spinner=False)
def get_question_pipeline(_doc_store):
    """
    Create the pipeline with the retriever and reader components.
    Args:
    - doc_store: instance of the document store.
    Returns:
    - pipe: instance of the pipeline.
    """
    st.caption(f"Building the Question Answering pipeline")
    retriever = InMemoryBM25Retriever(document_store=_doc_store)
    reader = ExtractiveReader(model="deepset/roberta-base-squad2")
    reader.warm_up()
    pipe = Pipeline()
    pipe.add_component(instance=retriever, name="retriever")
    pipe.add_component(instance=reader, name="reader")
    pipe.connect("retriever.documents", "reader.documents")
    return pipe


def search(pipeline, question: str):
    """
    Search for the answer to a question in the documents.
    Args:
    - pipeline: instance of the pipeline.
    - question: string with the question.
    Returns:
    - answer: dictionary with the answer.
    """
    top_k = 3
    answer = pipeline.run(
        data={
            "retriever": {"query": question, "top_k": 10},
            "reader": {"query": question, "top_k": top_k},
        }
    )
    max_k = min(top_k, len(answer["reader"]["answers"]))
    return answer["reader"]["answers"][0:max_k]


# Streamlit interface
_, centering_column, _ = st.columns(3)
with centering_column:
    st.image("assets/qasports-logo.png", use_column_width=True)

# Loading status
with st.status(
    "Downloading dataset...", expanded=st.session_state.get("expanded", True)
) as status:
    documents = load_documents()
    status.update(label="Indexing documents...")
    doc_store = get_document_store(documents)
    status.update(label="Creating pipeline...")
    pipe = get_question_pipeline(doc_store)
    status.update(
        label="Download and indexing complete!", state="complete", expanded=False
    )
    st.session_state["expanded"] = False

st.subheader("🔎 Basketball", divider="rainbow")
st.caption(
    """This website presents a collection of documents from the dataset named "QASports", the first large sports question answering dataset for open questions. QASports contains real data of players, teams and matches from the sports soccer, basketball and American football. It counts over 1.5 million questions and answers about 54k preprocessed, cleaned and organized documents from Wikipedia-like sources."""
)

if user_query := st.text_input(
    label="Ask a question about Basketball! 🏀",
    placeholder="How many field goals did Kobe Bryant score?",
):
    with st.spinner("Waiting"):
        try:
            answer = search(pipe, user_query)
            for idx, ans in enumerate(answer):
                st.info(
                    f"""
                    Answer {idx+1}: "{ans.data}" | Score: {ans.score:0.4f}  
                    Document: "{ans.document.meta["title"]}"  
                    URL: {ans.document.meta["url"]}
                """
                )
                with st.expander("See details", expanded=False):
                    st.write(ans)
                st.divider()
        except Exception as e:
            st.error("We do not have an answer for your question")
