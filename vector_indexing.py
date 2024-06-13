import faiss

d = 768  # Dimension of the BERT embeddings
index = faiss.IndexFlatL2(d)
index.add(np.vstack((lecture_embeddings, table_embeddings)))
