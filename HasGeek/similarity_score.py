from sentence_transformers import SentenceTransformer, util

# Load a pre-trained sentence embedding model
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

# Encode the two sentences into their vector representations
sentence1 = "How old are you?"
sentence2 = "What is your age?"
embeddings1 = model.encode(sentence1, convert_to_tensor=True)
embeddings2 = model.encode(sentence2, convert_to_tensor=True)
#print(embeddings1)
# Compute cosine similarity between the two vectors
cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)

print("Cosine Similarity Score:", cosine_similarity.item())