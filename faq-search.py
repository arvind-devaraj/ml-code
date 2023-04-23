from sentence_transformers import SentenceTransformer, util
import torch

embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Corpus with example sentences
#with open('queries.txt') as f:
#    lines = [line.rstrip('\n') for line in f]
import pandas as pd
df = pd.read_csv('faq.csv')
lines=df['Query'].to_list()
solution=df['Solution'].to_list()

corpus = lines
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
 
def generate_response(query):

    top_k = min(1, len(corpus))
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    for score, idx in zip(top_results[0], top_results[1]):
        #print(corpus[idx], "(Score: {:.4f})".format(score))
        return corpus[idx] + "\n" + solution[idx]
