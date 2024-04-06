from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


model = SentenceTransformer('paraphrase-distilroberta-base-v1')

data = [
    'What is your name?',
    'What is your age?',
]
encoded_data = model.encode(data)


# IndexIDMap: store document ids in the index as well
index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
index.add_with_ids(encoded_data, np.arange(len(data)))



def search(query, k=1):
    query_vector = model.encode([query])
    top_k = index.search(query_vector, k)
    print(top_k)
    return [
        data[_id] for _id in top_k[1][0]
    ]
    
result=search("How old are you?")
print(result)