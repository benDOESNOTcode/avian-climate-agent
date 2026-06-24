import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import duckdb

print("Loading Sentence Transformer (Local/Free Embeddings)...")
# This downloads a tiny, highly efficient embedding model to your laptop
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Initializing Local Qdrant Database...")
# This creates a local database folder right in your VS Code directory
client = QdrantClient(path="./qdrant_data")

collection_name = "avian_species"
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

print("Fetching Platinum Species Data from Hugging Face via DuckDB...")

# 1. Initialize connection and load httpfs for internet access
conn = duckdb.connect()
conn.execute("INSTALL httpfs;")
conn.execute("LOAD httpfs;")

# 2. Query using the correct single wildcard
query = """
    SELECT species, total_historical_sightings, avg_flock_size
    FROM read_parquet('hf://datasets/notBEn/avian-climate-intelligence/platinum_species_profiles.parquet/*.parquet')
"""
df = conn.execute(query).df()

print(f"Generating Embeddings for {len(df)} species (This takes about 30-60 seconds)...")
points = []
for i, row in df.iterrows():
    # Create a rich text document for the AI to read
    text_doc = f"Species: {row['species']}. Total historical sightings: {row['total_historical_sightings']:,}. Average observed flock size: {row['avg_flock_size']:.2f} birds."
    
    # Convert text to a vector
    vector = model.encode(text_doc).tolist()
    
    # Package it for Qdrant
    points.append(PointStruct(
        id=i,
        vector=vector,
        payload={
            "species": row['species'],
            "text": text_doc
        }
    ))
    
    # Batch upload to save memory
    if len(points) >= 500:
        client.upsert(collection_name=collection_name, points=points)
        points = []

# Upload any remaining points
if points:
    client.upsert(collection_name=collection_name, points=points)

print(f"\n✅ Successfully embedded and stored {len(df)} species profiles in local Qdrant!")