"""This module contains utility functions for the project"""

import mmh3
from haystack.schema import Document


def get_unique_docs(dataset, unique_docs: set):
    """Get unique documents from dataset
    Args:
    dataset: list of dictionaries
    Returns:
    docs: list of haystack.schema.Document
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
from haystack.nodes import FARMReader, BM25Retriever
from haystack.document_stores import InMemoryDocumentStore

# Utility function to get unique documents
def load_documents():
    unique_docs = set()
    dataset_name = "PedroCJardim/QASports"
    dataset_split = "basketball"
    st.caption(f'Fetching "{dataset_name}" dataset')
    dataset = load_dataset(dataset_name, dataset_split)
    docs_validation = get_unique_docs(dataset["validation"], unique_docs)
    docs_train = get_unique_docs(dataset["train"], unique_docs)
    docs_test = get_unique_docs(dataset["test"], unique_docs)
    documents = docs_validation + docs_train + docs_test
    return documents

# Cache the document store
@st.cache_resource(show_spinner=False)
def get_document_store(documents):
    st.caption(f"Building the Document Store")
    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents=documents)
    return document_store

# Cache the question answering pipeline
@st.cache_resource(show_spinner=False)
def get_question_pipeline(doc_store):
    st.caption(f"Building the Question Answering pipeline")
    retriever = BM25Retriever(document_store=doc_store)
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
    pipe = Pipeline()
    pipe.add_node(component=retriever, name="Retriever", inputs=["Query"])
    pipe.add_node(component=reader, name="Reader", inputs=["Retriever"])
    return pipe

# Function to search for answers
def search(pipeline, question: str):
    top_k = 3
    result = pipeline.run(query=question, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": top_k}})
    return result["answers"]

# Streamlit interface
_, centering_column, _ = st.columns(3)
with centering_column:
    st.image("assets/qasports-logo.png", use_column_width=True)

# Loading status
with st.status("Downloading dataset...", expanded=st.session_state.get("expanded", True)) as status:
    documents = load_documents()
    status.update(label="Indexing documents...")
    doc_store = get_document_store(documents)
    status.update(label="Creating pipeline...")
    pipe = get_question_pipeline(doc_store)
    status.update(label="Download and indexing complete!", state="complete", expanded=False)
    st.session_state["expanded"] = False

st.subheader("🔎 Basketball", divider="rainbow")
st.caption(
    """This website presents a collection of documents from the dataset named "QASports", the first large sports question answering dataset for open questions. QASports contains real data of players, teams and matches from the sports soccer, basketball and American football. It counts over 1.5 million questions and answers about 54k preprocessed, cleaned and organized documents from Wikipedia-like sources."""
)

if user_query := st.text_input(label="Ask a question about Basketball! 🏀", placeholder="How many field goals did Kobe Bryant score?"):
    with st.spinner("Waiting"):
        try:
            answers = search(pipe, user_query)
            for idx, answer in enumerate(answers):
                st.info(
                    f"""
                    Answer {idx+1}: "{answer.answer}" | Score: {answer.score:0.4f}  
                    Document: "{answer.meta["title"]}"  
                    URL: {answer.meta["url"]}
                """
                )
                with st.expander("See details", expanded=False):
                    st.write(answer)
                st.divider()
        except Exception as e:
            st.error("We do not have an answer for your question")
