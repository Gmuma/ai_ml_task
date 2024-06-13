def find_relevant_texts(query):
    query_embedding = get_embeddings(query)
    D, I = index.search(np.array([query_embedding]), k=5)
    relevant_texts = [lecture_notes[i] for i in I[0] if i < len(lecture_notes)] + \
                     [llm_table[i - len(lecture_notes)] for i in I[0] if i >= len(lecture_notes)]
    return relevant_texts

user_query = "Explain the architecture of BERT."
relevant_texts = find_relevant_texts(user_query)
response = generate_response(relevant_texts)
print(response)
