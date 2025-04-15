import time
from langchain_core.prompts import ChatPromptTemplate

from utils import llm
from astradb_retrievers import prods_retriever, faqs_retriever
from question_classification import classify_question

template = """
You are a helpful customer support assistant for a fashion e-commerce website. Answer the user's question based on the provided information. If you don't know the answer, just say that you don't have enough information.

This question is about: {category_description}

{context}

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
    print("Processing your question...\n")
    
    question_category = classify_question(question)
    
    # Time the retrieval step
    retrieval_start = time.time()
    
    context = ""
    category_description = ""
    
    if question_category == "1":
        products = prods_retriever.invoke(question)
        context = f"Relevant product details:\n{products}"
        category_description = "Product information and details"

        print("Category: Product information and details")
    elif question_category == "2":
        faqs = faqs_retriever.invoke(question)
        context = f"Relevant FAQ details:\n{faqs}"
        category_description = "Website usage and features"

        print("Category: Website usage and features")
    elif question_category == "3":
        products = prods_retriever.invoke(question)
        faqs = faqs_retriever.invoke(question)
        context = f"Relevant product details:\n{products}\n\nRelevant FAQ details:\n{faqs}"
        category_description = "Both product information and website usage"

        print("Category: Both product information and website usage")
    else:
        context = "No relevant information found, answer in a general way"
        category_description = "Irrelevant question"
        
    retrieval_time = time.time() - retrieval_start
    
    # Time the LLM response step
    llm_start = time.time()
    result = chain.invoke({
        "category_description": category_description,
        "context": context,
        "question": question
    })
    llm_time = time.time() - llm_start
    
    # Print timing information
    total_time = time.time() - start_time
    print('\nAnswer: ' + result.content)
    print(f"\nPerformance metrics:")
    print(f"  - Vector retrieval: {retrieval_time:.2f} seconds")
    print(f"  - LLM response generation: {llm_time:.2f} seconds")
    print(f"  - Total response time: {total_time:.2f} seconds")
