from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
import os

from utils import embeddings

persist_directory = './chroma_db'

vector_store = Chroma(
    collection_name="osconcept",
    persist_directory=persist_directory,
    embedding_function=embeddings
)

# if True:
#     df = pd.read_csv('data/products.csv')
#     docs = []
#     ids = []
#     for index, row in df.iterrows():
#         doc = Document(
#             page_content= 'name:' + row['name']
#                           + ' description:' + row['description']
#                           + ' age_gender:' + row['age_gender']
#                           + ' size:' + row['size']
#                           + ' price:' + str(row['price'])
#                           + ' quantity_in_stock:' + str(row['quantity_in_stock']),
#             metadata={
#                 'id': row['id'],
#                 'category_id': row['category_id'],
#             },
#             id=str(index)
#         )
#         ids.append(str(index))
#         docs.append(doc)
#     vector_store.add_documents(documents=docs, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={'k': 5})

