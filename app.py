import streamlit as st
from PIL import Image
import os
from search_engine import SemanticSearchEngine

st.set_page_config(page_title="Semantic Fashion Search", layout="wide", page_icon="🔍")

# Cache the model to prevent reloading on every interaction
@st.cache_resource
def load_engine():
    try:
        return SemanticSearchEngine()
    except FileNotFoundError:
        return None

st.title("🔍 Semantic Fashion Search Engine")
st.markdown("Search for products using natural language. Built with OpenAI CLIP.")

engine = load_engine()

if engine is None:
    st.warning("⚠️ No embeddings found. Please add images to `data/images` and run `python encode_data.py` first.")
else:
    # Search Bar
    query = st.text_input("Describe the product (e.g., 'black leather shoes', 'red summer dress')...", 
                          placeholder="Type here and press Enter...")

    if query:
        with st.spinner("Searching for matches..."):
            results = engine.search(query, top_k=8)
            
            if not results:
                st.info("No matches found.")
            else:
                st.success(f"Top matches for: '{query}'")
                
                # Create a grid layout (4 columns)
                cols = st.columns(4)
                for i, res in enumerate(results):
                    with cols[i % 4]:
                        try:
                            img = Image.open(res["path"])
                            st.image(img, use_container_width=True)
                            st.caption(f"Score: {res['score']:.3f}")
                        except Exception as e:
                            st.error("Image not found.")

st.markdown("---")
st.caption("🚀 AI Prototype Search System")
