from RAG_QA_model import *
from process_text import *
from  evaluate import *

import RAG_QA_model
import process_text
import evaluate
import gc

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

gc.collect()
torch.cuda.empty_cache()
# Closed book
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    device_map="auto",
)

# Read queries from file
with open(PATHS["question"], "r") as f:
    querys = f.readlines()

querys = [query.strip() for query in querys if query.strip()]
    
answers = {}

for index, query in enumerate(tqdm(querys), start=1):

    input_ids = tokenizer(query, return_tensors="pt").to("cuda")

    outputs = model.generate(**input_ids, max_new_tokens=32)
    answer = tokenizer.decode(outputs[0])

    answers[str(index)] = answer
        # Print results
        # print(f"Question: {query}")
        # print(f"Answer: {answer}")
        # print("\nSearch Results Used:")
        # print(format_search_results(search_results))
    
# write answers to json format with question index and answer
with open("closed_book_generated_answers.json", "w") as f:  
    json.dump(answers, f, indent=2, separators=(',\n', ': '), )


# Evaluation
# Load data
with open(PATHS["answer"], "r") as f:
    ground_truth = json.load(f)
with open("closed_book_generated_answers.json", "r") as f:
    generated_answer = json.load(f)

# Evaluate
result = evaluate_qa(ground_truth, generated_answer)

# Print overall results
print("\nOverall Results:")
print(f"Precision: {result['precision']:.4f}")
print(f"Recall: {result['recall']:.4f}")
print(f"Accuracy: {result['accuracy']:.4f}")
print(f"Exact Match: {result['exact_match']:.4f}")
print(f"F1: {result['f1']:.4f}")

# Save results with model information
save_evaluation_results(result)