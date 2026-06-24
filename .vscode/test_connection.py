import duckdb

print("Connecting to Hugging Face via DuckDB...")
conn = duckdb.connect()

# Force DuckDB to load its internet capabilities
conn.execute("INSTALL httpfs;")
conn.execute("LOAD httpfs;")

# Notice the path now ends in just /*.parquet
query = """
    SELECT species, total_historical_sightings, avg_flock_size
    FROM read_parquet('hf://datasets/notBEn/avian-climate-intelligence/platinum_species_profiles.parquet/*.parquet')
    ORDER BY total_historical_sightings DESC
    LIMIT 5
"""

print("\nExecuting remote query across the network...\n")
result_df = conn.execute(query).df()
print(result_df)


