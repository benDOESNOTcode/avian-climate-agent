from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# 1. Initialize the client pointing to your local data folder
client = QdrantClient(path="./qdrant_data")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Define a natural language query
query = "Show me birds that travel in massive flocks"
print(f"Querying vector database for: '{query}'\n")

# 3. Embed the user query using the same local model
query_vector = model.encode(query).tolist()

# 4. Search the vector database using the updated parameter name 'query'
results = client.query_points(
    collection_name="avian_species",
    query=query_vector,
    limit=3
).points

# 5. Print out the semantic matches
for rank, hit in enumerate(results, start=1):
    print(f"Rank {rank} (Score: {hit.score:.4f}):")
    print(f"📄 {hit.payload['text']}\n")