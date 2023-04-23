import faiss
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

path="data/faiss-index.bin"
index = faiss.read_index(path)


    
fp = open("data/chap1.txt")
corpus = fp.readlines()
#print(corpus)


from time_it import *

@timeit
def search(query, k=5):
    query_vector = model.encode([query])
    top_k = index.search(query_vector, k)
    #print(top_k)
    idx=(top_k[1][0][0])
    print(corpus[idx])
    
