"""
This file is the RAG_QA_model for the assignment 2.
You can provide the question and the context and the model.

The RAG_QA_model will answer the question based on the context.

"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Dict, Any
from search import *
from tqdm import tqdm
import json
from param import *
import torch
import gc
from sentence_transformers import CrossEncoder
import re


def setup_llama_model() -> AutoModelForCausalLM:
    """
    Initialize the TinyLlama model
    
    Returns:
        AutoModelForCausalLM: Initialized model instance
    """
    tokenizer = AutoTokenizer.from_pretrained(LLM2['model_id'])
    model = AutoModelForCausalLM.from_pretrained(LLM2['model_id'])
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        print("Moving model to GPU")
        model = model.to("cuda")
    else:
        print("GPU not available. Using CPU.")
    
    return model, tokenizer

def get_model_response(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer,
    query: str, 
    context: str, 
    temperature: float = LLM["temperature"], 
    max_tokens: int = LLM["max_tokens"],
    top_p: float = LLM["top_p"]
) -> str:
    """
    Get response from the model.
    """
    # Build the prompt from the query and context using the provided template.
    prompt = RAG["prompt_template"].format(query=query, context=context)
    
    # Do NOT reinitialize the tokenizer and model here.
    # They are already loaded and moved to the proper device in setup_llama_model().
    
    # Ensure input tensors are on the same device as the model.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    
    outputs = model.generate(
        **input_ids,
        max_length=max_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,  # Set do_sample to True to use temperature and top_p
        num_return_sequences=1
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the answer part from the response without truncating
    answer = response
    marker = "**Your Answer:**"
    answer_start = answer.find(marker)
    if answer_start != -1:
        answer_start += len(marker)
        answer = answer[answer_start:].strip()
    elif answer.find("**Answer:**"):
        marker = "**Answer:**"
        answer_start += len(marker)
        answer = answer[answer_start:].strip()
    answer_end = answer.find("\n")
    if answer_end:
        answer = answer[:answer_end]

    return answer


def qa_with_context(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, query: str, context: str) -> str:
    """
    Perform question answering using search results
    
    Args:
        query: User question
        search_results: Results from vector database search
        
    Returns:
        str: Model's answer
    """
    # Prepare context from search results
    # context = "\n".join([result['content'] for result in search_results])
    
    try:
        answer = get_model_response(model, tokenizer, query, context)
        return answer
    except Exception as e:
        print(f"Error during QA: {str(e)}")
        raise RuntimeError
        return "Sorry, I encountered an error while processing your question."

# Integration with your search function
def main():
    """
    Main function demonstrating the QA pipeline
    """
    # Initialize vector database
    db = initialize_vector_db()
    model, tokenizer = setup_llama_model()
    
    # Read queries from file
    with open(PATHS["question"], "r") as f:
        querys = f.readlines()

    querys = [query.strip() for query in querys if query.strip()]
    
    # answers = {}

    for index, query in enumerate(tqdm(querys), start=1):
        gc.collect()
        torch.cuda.empty_cache()

        # Get search results
        search_results = search_documents(query=query, db=db, top_k=10, score_threshold=0.5)

        if not isinstance(search_results, list):
            raise TypeError(f"Expected a list from search_documents(), but got {type(search_results)}")

        # Debugging: Verify search_results
        print(f"DEBUG: search_results type={type(search_results)}")
        
        # if not search_results:
        #     print(f"Warning: No valid documents found for query: {query}")
        #     answers[str(index)] = "No relevant answer found."
        #     continue

        # Reranker processing
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, doc["content"]) for doc in search_results if isinstance(doc, dict) and "content" in doc]

        # if not pairs:
        #     print(f"Warning: No valid documents found for query: {query}")
        #     answers[str(index)] = "No relevant answer found."
        #     continue

        # Rerank results
        scores = reranker.predict(pairs)
        reranked_results = [
            doc for _, doc in sorted(zip(scores, search_results), key=lambda x: x[0], reverse=True)
            if isinstance(doc, dict)
        ]

        # Fix top_docs to ensure it is a list of dictionaries
        top_docs = reranked_results[:3] if isinstance(reranked_results[0], dict) else [{"content": doc} for doc in reranked_results[:3]]
        context = "\n".join([doc["content"] for doc in top_docs if isinstance(doc, dict) and "content" in doc])

        # Get answer
        answer = qa_with_context(model=model, tokenizer=tokenizer, query=query, context=context)
        # print(f"Query: {query}")
        # print(f"Context: {context}")
        # print(f"Answer: {answer}\n")

        # Write answers to JSON
        with open(PATHS["generated_answer"], "a") as f:  
            f.write(f"{index}::{answer}\n")
        print(f"processed: {index}")


if __name__ == "__main__":
    main()