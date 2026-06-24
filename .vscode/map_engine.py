import streamlit as st
import pydeck as pdk
import duckdb
import pandas as pd

@st.cache_data
def load_trajectory_data(species_keyword: str):
    """Fetches the 20-year centroid trajectory for a specific species"""
    conn = duckdb.connect()
    conn.execute("INSTALL httpfs;")
    conn.execute("LOAD httpfs;")
    
    # Target the yearly trends file which contains 'year', 'centroid_lon', and 'centroid_lat'
    query = f"""
        SELECT 
            year, 
            centroid_lon, 
            centroid_lat, 
            total_population
        FROM read_parquet('hf://datasets/notBEn/avian-climate-intelligence/platinum_yearly_trends.parquet/*.parquet')
        WHERE species ILIKE '%{species_keyword}%'
        ORDER BY year
    """
    df = conn.execute(query).df()

    # Convert year safely to an integer type
    if not df.empty and 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df = df.dropna(subset=['year'])
        df['year'] = df['year'].astype(int)

    return df

def render_interactive_map(species_name: str):
    """Builds the Streamlit UI for the Pydeck map"""
    st.subheader(f"🌐 Spatial Trajectory: {species_name}")
    
    with st.spinner("Fetching spatial data from Hugging Face..."):
        df = load_trajectory_data(species_name)
        
    if df.empty:
        st.warning(f"No spatial data found for {species_name}.")
        return

    min_year = int(df['year'].min())
    max_year = int(df['year'].max())
    
    selected_year = st.slider("Select Year", min_year, max_year, min_year, step=1)

    # Filter data safely
    current_data = df[df['year'] == selected_year]
    trail_data   = df[df['year'] <= selected_year]

    current_layer = pdk.Layer(
        "ScatterplotLayer",
        data=current_data,
        get_position="[centroid_lon, centroid_lat]",
        get_radius="total_population / 100",
        radius_min_pixels=10,
        radius_max_pixels=50,
        # 🌟 Lowered the 4th number (Alpha) from 200 to 100 to make the red dot more transparent
        get_fill_color=[255, 75, 75, 100], 
        pickable=True
    )
    
    path_layer = pdk.Layer(
        "ScatterplotLayer",
        data=trail_data,
        get_position="[centroid_lon, centroid_lat]",
        get_radius=5000,
        # 🌟 Changed RGB to [255, 255, 0] for bright yellow, with a 150 Alpha for a slight glow effect
        get_fill_color=[255, 255, 0, 150], 
    )

    view_state = pdk.ViewState(
        latitude=df['centroid_lat'].mean(),
        longitude=df['centroid_lon'].mean(),
        zoom=3.5,
        pitch=0
    )

    # 6. Render to Streamlit
    st.pydeck_chart(pdk.Deck(
        map_style="dark", # 🌟 CHANGE THIS LINE: Uses the free, built-in dark map
        initial_view_state=view_state,
        layers=[path_layer, current_layer],
        tooltip={"text": "Year: {year}\nPopulation: {total_population}"}
    ))