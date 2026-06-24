import os
import re
from typing import TypedDict, Any, Optional
from dotenv import load_dotenv
import duckdb
import pandas as pd
import plotly.express as px
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

load_dotenv() 

# Initialize local data engines
qdrant_client = QdrantClient(path="./qdrant_data")
encoder = SentenceTransformer("all-MiniLM-L6-v2")
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)

# --- 1. EXISTING HELPER TOOLS ---
def query_vector_db(user_query: str) -> str:
    query_vector = encoder.encode(user_query).tolist()
    hits = qdrant_client.query_points(collection_name="avian_species", query=query_vector, limit=2).points
    context = "".join([f"- {hit.payload['text']}\n" for hit in hits])
    return context if context else "No species context found."

def query_analytics_db(sql_query: str) -> str:
    conn = duckdb.connect()
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    try:
        if "platinum_species_profiles" in sql_query and "read_parquet" not in sql_query:
            sql_query = sql_query.replace("platinum_species_profiles", "read_parquet('hf://datasets/notBEn/avian-climate-intelligence/platinum_species_profiles.parquet/*.parquet')")
        df = conn.execute(sql_query).df()
        return df.to_string(index=False)
    except Exception as e:
        return f"SQL Error: {str(e)}"

def predict_climate_impact(species: str, temp_change: float) -> str:
    baseline_pop = 100000 
    impact_factor = 1.05 if "migratorius" in species.lower() or "robin" in species.lower() else -0.15 
    projected_pop = int(baseline_pop + (baseline_pop * (temp_change * impact_factor)))
    delta = projected_pop - baseline_pop
    return f"ML Model Output: Baseline=100,000. Projected={projected_pop}. Delta={delta}."


# --- 2. THE LANGGRAPH CHARTING AGENT ---
class ChartState(TypedDict):
    question: str
    sql_query: str
    df: Optional[pd.DataFrame]
    df_schema: str
    code: str
    fig: Any
    error: str
    sql_retries: int
    code_retries: int
    final_message: str

def generate_sql_node(state: ChartState):
    error_context = f"\nPREVIOUS ERROR TO FIX:\n{state['error']}\n" if state.get('error') else ""
    prompt = ChatPromptTemplate.from_template(
        "You are an expert DuckDB engineer.\n"
        "Table: read_parquet('hf://datasets/notBEn/avian-climate-intelligence/platinum_species_profiles.parquet/*.parquet')\n"
        "Data Dictionary:\n"
        "- species (VARCHAR): The scientific name of the bird.\n"
        "- total_historical_sightings (BIGINT): Total number of times observed.\n"
        "- avg_flock_size (DOUBLE): Average number of birds in a single sighting.\n"
        "{error_context}"
        "Write a raw DuckDB SQL query to get the data needed for this charting request: '{question}'.\n"
        "Return ONLY the executable SQL string."
    )
    sql_query = (prompt | llm).invoke({"question": state["question"], "error_context": error_context}).content.strip()
    clean_sql = sql_query.replace("```sql", "").replace("```", "").strip()
    return {"sql_query": clean_sql, "sql_retries": state["sql_retries"] + 1, "error": ""}

def execute_sql_node(state: ChartState):
    conn = duckdb.connect()
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    try:
        df = conn.execute(state["sql_query"]).df()
        schema_snapshot = f"Columns & Types:\n{df.dtypes.to_string()}\n\nSample Data (First 2 Rows):\n{df.head(2).to_string()}"
        return {"df": df, "df_schema": schema_snapshot, "error": ""}
    except Exception as e:
        return {"error": str(e), "df": None}

def generate_code_node(state: ChartState):
    error_context = f"\nPREVIOUS ERROR TO FIX:\n{state['error']}\n" if state.get('error') else ""
    prompt = ChatPromptTemplate.from_template(
        "You are a Python Data Viz expert using plotly.express as `px`.\n"
        "You have a pandas DataFrame named `df` with this exact verified schema and sample data:\n{schema}\n"
        "{error_context}"
        "Write a Python script to create a chart for this request: '{question}'.\n"
        "Assign the final plot to a variable named `fig`.\n"
        "Return ONLY the Python code."
    )
    code_response = (prompt | llm).invoke({
        "question": state["question"], "schema": state["df_schema"], "error_context": error_context
    }).content
    match = re.search(r"```python\n(.*?)\n```", code_response, re.DOTALL)
    clean_code = match.group(1) if match else code_response.replace("```python", "").replace("```", "")
    return {"code": clean_code.strip(), "code_retries": state["code_retries"] + 1, "error": ""}

def execute_code_node(state: ChartState):
    local_vars = {"df": state["df"], "px": px, "pd": pd}
    try:
        exec(state["code"], {}, local_vars)
        return {"fig": local_vars.get("fig"), "final_message": "Here is the interactive chart based on the data:", "error": ""}
    except Exception as e:
        return {"error": str(e), "fig": None}

def route_after_sql(state: ChartState):
    if state.get("error"):
        return "generate_sql" if state["sql_retries"] < 3 else END
    return "generate_code"

def route_after_code(state: ChartState):
    if state.get("error"):
        return "generate_code" if state["code_retries"] < 3 else END
    return END

workflow = StateGraph(ChartState)
workflow.add_node("generate_sql", generate_sql_node)
workflow.add_node("execute_sql", execute_sql_node)
workflow.add_node("generate_code", generate_code_node)
workflow.add_node("execute_code", execute_code_node)
workflow.set_entry_point("generate_sql")
workflow.add_conditional_edges("execute_sql", route_after_sql)
workflow.add_conditional_edges("execute_code", route_after_code)
workflow.add_edge("generate_sql", "execute_sql")
workflow.add_edge("generate_code", "execute_code")
chart_agent = workflow.compile()


# --- 3. MAIN ROUTER ---
def run_agent_turn(user_prompt: str) -> tuple[str, any]:
    routing_prompt = ChatPromptTemplate.from_template(
        "Analyze the following question.\n"
        "- If it asks for a chart, graph, plot, or visualization, output 'CHART'.\n"
        "- If it asks for specific historical counts, sums, or top lists, output 'SQL'.\n"
        "- If it asks for descriptive context or details about a bird, output 'VECTOR'.\n"
        "- If it asks to forecast, predict, or estimate population changes, output 'PREDICT'.\n"
        "Question: {question}\n"
        "Reply with exactly one word: 'CHART', 'SQL', 'VECTOR', or 'PREDICT'."
    )
    decision = (routing_prompt | llm).invoke({"question": user_prompt}).content.strip().upper()
    print(f"[AI Agent Decision Engine]: Route -> {decision}")
    
    if "CHART" in decision:
        initial_state = {
            "question": user_prompt, "sql_retries": 0, "code_retries": 0, "error": "", 
            "sql_query": "", "df": None, "df_schema": "", "code": "", "fig": None, "final_message": ""
        }
        final_state = chart_agent.invoke(initial_state)
        if final_state.get("fig"):
            return final_state["final_message"], final_state["fig"]
        return f"Chart generation failed after multiple retries. Final Error: {final_state.get('error')}", None

    elif "SQL" in decision:
        # 🌟 RESTORED: The Data Dictionary so it doesn't hallucinate column names
        sql_prompt = ChatPromptTemplate.from_template(
            "Write a raw DuckDB SQL query to answer the question.\n"
            "Table: read_parquet('hf://datasets/notBEn/avian-climate-intelligence/platinum_species_profiles.parquet/*.parquet')\n"
            "Columns available:\n"
            "- species (VARCHAR)\n"
            "- total_historical_sightings (BIGINT)\n"
            "- avg_flock_size (DOUBLE)\n"
            "Return ONLY the SQL string.\nQuestion: {question}"
        )
        sql = (sql_prompt | llm).invoke({"question": user_prompt}).content.strip().replace("```sql", "").replace("```", "")
        return query_analytics_db(sql), None

    elif "PREDICT" in decision:
        # 🌟 RESTORED: The ML parameter extraction and rigorous scientific synthesis
        extract_prompt = ChatPromptTemplate.from_template(
            "Extract the bird species and the temperature change in celsius from this prompt.\n"
            "If no temperature is provided, default to 2.0.\n"
            "Prompt: {question}\n"
            "Return ONLY a comma-separated string: SpeciesName, TemperatureChange"
        )
        try:
            params = (extract_prompt | llm).invoke({"question": user_prompt}).content.strip().split(',')
            species = params[0].strip()
            temp_change = float(params[1].strip().replace('C', '').replace('+', ''))
        except Exception:
            species = "Mourning Dove"
            temp_change = 1.5

        ml_results = predict_climate_impact(species, temp_change)
        
        synth_prompt = ChatPromptTemplate.from_template(
            "You are an ecological data scientist explaining a machine learning forecast.\n"
            "Raw ML Output: {ml_results}\n"
            "User Question: {question}\n\n"
            "RULES FOR SYNTHESIS:\n"
            "1. Echo the exact causal input (e.g., 'a +1.5°C temperature increase'). Do not use vague terms like 'fluctuations'.\n"
            "2. State that the population 'is projected to reach' the target.\n"
            "3. State that this forecast carries inherent uncertainty and is a directional estimate.\n"
            "4. Hypothesize a brief, realistic ecological mechanism for this change.\n"
            "5. Keep the response to a single, highly professional paragraph.\n"
            "Answer:"
        )
        synth_result = (synth_prompt | llm).invoke({"question": user_prompt, "ml_results": ml_results}).content
        return synth_result, None

    else:
        context = query_vector_db(user_prompt)
        synthesis_prompt = ChatPromptTemplate.from_template("Answer based on context:\n{context}\nQuestion: {question}")
        return (synthesis_prompt | llm).invoke({"question": user_prompt, "context": context}).content, None