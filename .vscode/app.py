import streamlit as st
from agent_brain import run_agent_turn
from map_engine import render_interactive_map 

# Configure the web page
st.set_page_config(
    page_title="Avian Climate Intelligence", 
    page_icon="🦅", 
    layout="wide" 
)

st.title("🦅 Avian Climate Intelligence")

# Create two interactive tabs
tab_chat, tab_map = st.tabs(["💬 AI Agent Chat", "🗺️ Spatial Explorer"])

# --- TAB 1: The AI Agent ---
with tab_chat:
    st.markdown("Ask the AI about historical bird sightings, flock dynamics, or climate correlations.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # If a previous message has a chart stored, render it
            if "chart" in message and message["chart"] is not None:
                st.plotly_chart(message["chart"], use_container_width=True)

    if prompt := st.chat_input("E.g., Create a bar chart showing the top 5 most sighted birds."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Agent is analyzing data engines..."):
                try:
                    # Unpack the tuple returned by the agent
                    response_text, chart_figure = run_agent_turn(prompt)
                    
                    st.markdown(response_text)
                    
                    # Store the response AND the chart in session state history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_text,
                        "chart": chart_figure
                    })
                    
                    # Render the chart dynamically if the agent built one
                    if chart_figure:
                        st.plotly_chart(chart_figure, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"The agent encountered an error: {e}")

# --- TAB 2: The Map Engine ---
with tab_map:
    st.markdown("Visualize the shifting population centroids over the last 20 years.")
    
    species_to_map = st.selectbox(
        "Select a Species to Map:",
        ["Turdus migratorius", "Corvus brachyrhynchos", "Cardinalis cardinalis", "Zenaida macroura"]
    )
    
    if species_to_map:
        render_interactive_map(species_to_map)