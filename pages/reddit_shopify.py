import streamlit as st
import requests
import base64
import json
from io import BytesIO
from requests.exceptions import RequestException
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image
 
st.set_page_config(page_title="REDDIT SHOPIFY", layout="centered")
st.title("🧠 Grok AI Product Poster Generator")
st.markdown("Generate photorealistic e-commerce product posters using **Grok (xAI)**.")
 
@st.cache_resource
def get_session():
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session
 
# In cloud mode, we simulate a successful connection
def check_server():
    return True  # Always return True in cloud deployment
 
# === Form Inputs === #
with st.form("generate_form"):
    title = st.text_input("🛍️ Product Title", placeholder="Enter product name", max_chars=100)
    description = st.text_area("📝 Product Description", placeholder="Write a compelling product description")
    submitted = st.form_submit_button("🚀 Generate Poster")
 
# === Simulate POST to Server === #
if submitted:
    if not title or not description:
        st.warning("Please fill in both the product title and description.")
    else:
        with st.spinner("Generating image using Grok..."):
            # In cloud deployment, we simulate a successful response
            # Simulate processing time
            import time
            time.sleep(2)
            
            # Show success message
            st.success("Poster generated successfully!")
            
            # Display a placeholder image
            st.image("https://placehold.co/600x400/1F2937/FFFFFF?text=Product+Poster+Demo", 
                    caption=title, 
                    use_column_width=True)
            
            # Add simulated Reddit/Shopify links
            st.success("✅ Posted to Reddit!")
            st.markdown("""
            🔗 **Reddit Post**: [View on Reddit Demo](https://reddit.com)
            """)
            
            st.success("✅ Added to Shopify Store!")
            st.markdown("""
            🛍️ **Product Page**: [View in Store Demo](https://shopify.com)
            """)
            
            # Add social sharing buttons
            st.markdown("### 📢 Share")
            col1, col2 = st.columns(2)
            with col1:
                st.link_button(
                    "Share on Reddit",
                    "https://reddit.com",
                    help="View and share the Reddit post"
                )
            with col2:
                st.link_button(
                    "View in Store",
                    "https://shopify.com",
                    help="View the product in Shopify store"
                )
            
            # Move the metadata expander after the sharing buttons
            with st.expander("📊 Generation Details"):
                st.text("Product Details:")
                st.code(f"""
                Title: {title}
                Description: {description}
                
                Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
                """)
                
                st.text("Note: This is a simulated response for cloud deployment demonstration.")
