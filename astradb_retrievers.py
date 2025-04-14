import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.schema.document import Document
from langchain_astradb import AstraDBVectorStore

import pandas as pd

from utils import embeddings

llm = ChatOllama(model='mistral')

load_dotenv()

def load_prods_data():
    df = pd.read_csv('data/products.csv')
    docs = []
    ids = []

    for index, row in df.iterrows():
        doc = Document(
            page_content= 'name:' + row['name']
                          + ' description:' + row['description']
                          + ' age_gender:' + row['age_gender']
                          + ' size:' + row['size']
                          + ' price:' + str(row['price'])
                          + ' quantity_in_stock:' + str(row['quantity_in_stock']),
            metadata={
                'id': row['id'],
                'category_id': row['category_id'],
            },
            id=str(index)
        )
        ids.append(str(index))
        docs.append(doc)
    return docs, ids

def load_faqs_data():
    df = pd.read_csv('data/question_answer.csv')
    docs = []
    ids = []

    for index, row in df.iterrows():
        doc = Document(
            page_content= 'question:' + row['question']
                          + ' answer:' + row['answer'],
            metadata={
                'id': row['id'],
            },
            id=str(index)
        )
        ids.append(str(index))
        docs.append(doc)
    return docs, ids

def connect_to_vstore(coll_name: str):
    ASTRA_DB_API_ENDPOINT = os.getenv('ASTRA_DB_API_ENDPOINT')
    ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
    desired_namespace = os.getenv('ASTRA_DB_KEYSPACE')

    if desired_namespace:
        ASTRA_DB_KEYSPACE = desired_namespace
    else:
        ASTRA_DB_KEYSPACE = None

    vstore = AstraDBVectorStore(
        embedding=embeddings,
        collection_name=coll_name,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
        namespace=ASTRA_DB_KEYSPACE,
    )
    return vstore

prods_vstore = connect_to_vstore('products')
faqs_vstore = connect_to_vstore('faqs')

# Only uncomment this line when you need to delete all data
# prods_vstore.delete(ids=[str(i) for i in range(1355)])

# Only uncomment this line when you need to load/reload products data
# docs, ids = load_prods_data()
# prods_vstore.add_documents(documents=docs, ids=ids)

# Only uncomment this line when you need to load/reload faqs data
# docs, ids = load_faqs_data()
# faqs_vstore.add_documents(documents=docs, ids=ids)

prods_retriever = prods_vstore.as_retriever(search_kwargs={'k': 5})
# faqs_retriever = faqs_vstore.as_retriever(search_kwargs={'k': 5})
