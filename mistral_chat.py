import time
from langchain_core.prompts import ChatPromptTemplate

from utils import llm
from astradb_retrievers import prods_retriever

template = """
You are a helpful customer support assistant for a fashion e-commerce website. Answer the user's question based on the provided product information. If you don't know the answer, just say something like you don't know. Do not try to make up an answer.

Relevant product details:
{products}

User question: {question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    if question == "q":
        break

    start_time = time.time()

    print("Processing your question...")

    # Time the retrieval step
    retrieval_start = time.time()
    products = prods_retriever.invoke(question)
    retrieval_time = time.time() - retrieval_start

    # Time the LLM response step
    llm_start = time.time()
    result = chain.invoke({"products": products, "question": question})
    llm_time = time.time() - llm_start

    # Print timing information
    total_time = time.time() - start_time
    print('\nAnswer: ' + result.content)
    print(f"\nPerformance metrics:")
    print(f"  - Vector retrieval: {retrieval_time:.2f} seconds")
    print(f"  - LLM response generation: {llm_time:.2f} seconds")
    print(f"  - Total response time: {total_time:.2f} seconds")


#remember to check chat
