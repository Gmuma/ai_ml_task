from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

model_name = 'bert-base-uncased'
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

lecture_notes = ["Lecture 1 text...", "Lecture 2 text..."]
llm_table = ["BERT details...", "GPT-3 details..."]

lecture_embeddings = np.array([get_embeddings(note) for note in lecture_notes])
table_embeddings = np.array([get_embeddings(entry) for entry in llm_table])
