from llama_cpp import Llama
from typing import List, Dict, Any
from search import *
from tqdm import tqdm
import json
from param import *

def setup_llama_model() -> Llama:
    """
    Initialize the quantized TinyLlama model
    
    Returns:
        Llama: Initialized model instance
    """
    model = Llama(
        model_path=LLM["model_path"],  # Download this file
        n_ctx=LLM["n_ctx"],        # Context window size
        n_threads=4,       # Adjust based on CPU
        n_gpu_layers=0,    # Set to use GPU if available
        verbose=False     
    )
    return model

def get_model_response(
    model: Llama, 
    query: str, 
    context: str, 
    temperature: float = 0.7, 
    max_tokens: int = 512
) -> str:
    """
    Get response from the model
    
    Args:
        model: Initialized Llama model
        query: User question
        context: Retrieved context from vector database
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        
    Returns:
        str: Model's response
    """
    prompt = f"""Based on the following context, please answer the question.
    
Context: {context}

Question: {query}

Answer: """
    
    response = model.create_completion(
        prompt,
        max_tokens=LLM["max_tokens"],
        temperature=LLM["temperature"],
        stop=["Question:", "\n\n"],
        echo=False
    )
    
    return response['choices'][0]['text'].strip()

def qa_with_context(model: Llama, query: str, search_results: List[Dict[str, Any]]) -> str:
    """
    Perform question answering using search results
    
    Args:
        query: User question
        search_results: Results from vector database search
        
    Returns:
        str: Model's answer
    """
    # Initialize model
    
    # Prepare context from search results
    context = "\n".join([result['content'] for result in search_results])
    
    try:
        answer = get_model_response(model, query, context)
        return answer
    except Exception as e:
        print(f"Error during QA: {str(e)}")
        return "Sorry, I encountered an error while processing your question."

# Integration with your search function
def main():
    """
    Main function demonstrating the QA pipeline
    """
    # Initialize vector database
    db = initialize_vector_db()
    model = setup_llama_model()
    
    # Example query
    # read question from file
    with open("test_data/question_Zijin.txt", "r") as f:
        querys = f.readlines()

    # remove change line character
    querys = [query.strip() for query in querys]

    answers = {}

    # remove empty lines
    
    index = 0
    for query in tqdm(querys):
        index += 1
    
        # Get search results
        search_results = search_documents(
            query=query,
            db=db,
            top_k=3,  # Adjust based on needs
            score_threshold=0.5
        )
    
        # Get answer
        answer = qa_with_context(model, query, search_results)
        answers[str(index)] = answer
        # Print results
        # print(f"Question: {query}")
        # print(f"Answer: {answer}")
        # print("\nSearch Results Used:")
        # print(format_search_results(search_results))
    
    # write answers to json format with question index and answer
    with open("test_data/generated_answers_Zijin.json", "w") as f:  
        json.dump(answers, f)


if __name__ == "__main__":
    main()