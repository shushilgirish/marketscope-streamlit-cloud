import streamlit as st
import sys
import os
from typing import Dict, Any
import requests
import json
 
# Add root directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
 
# Import from config package
from config import Config
from frontend.utils import process_query, sidebar, create_visualization_from_mcp, render_chart, get_mcp_server_url
 
# Replace:
# from mcp.client import MCPClient
 
# With:
from frontend.custom_mcp_client import MCPClient
 
# Create a simple MCPClient alternative that uses requests
class SimpleMCPClient:
    def __init__(self, base_url):
        self.base_url = base_url
        print(f"Initialized SimpleMCPClient with base_url: {base_url}")
       
    def invoke(self, tool_name, params=None):
        """Invoke an MCP tool"""
        try:
            # The correct endpoint structure for FastMCP
            url = f"{self.base_url}/invoke/{tool_name}"
            print(f"Invoking tool: {tool_name} at URL: {url}")
           
            response = requests.post(
                url,
                json=params or {},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
           
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"Request failed with status {response.status_code}: {response.text}"
                print(f"Error: {error_msg}")
                return {
                    "status": "error",
                    "message": error_msg
                }
        except Exception as e:
            print(f"Exception calling MCP: {str(e)}")
            return {
                "status": "error",
                "message": f"Error invoking MCP tool: {str(e)}"
            }
   
    # Add the missing invoke_sync method as an alias to invoke
    invoke_sync = invoke
 
# Use SimpleMCPClient instead of MCPClient
MCPClient = SimpleMCPClient
 
# Check your get_mcp_server_url function to ensure it's looking up the right port
def get_mcp_server_url(segment_name):
    """Get the URL for a specific segment MCP server"""
    from frontend.config import Config
   
    # Default port if segment not found
    default_port = 8014
   
    # Get port from Config
    port = default_port
    if hasattr(Config, 'SEGMENT_CONFIG') and segment_name in Config.SEGMENT_CONFIG:
        port = Config.SEGMENT_CONFIG[segment_name].get('port', default_port)
    else:
        print(f"Warning: Segment {segment_name} not found in Config.SEGMENT_CONFIG")
        print(f"Available segments: {Config.SEGMENT_CONFIG.keys() if hasattr(Config, 'SEGMENT_CONFIG') else 'None'}")
   
    # Map segment names to server URLs
    return f"http://34.172.249.210:{port}"
 
st.set_page_config(layout="wide")
 
 
def show():
    """
    Display the Market Segmentation page content
    """
    sidebar()
 
    # Get model information from session state with fallback to config
    selected_model = "gpt-4o"
    temperature = 0.3
   
    # Get model config for display
    model_name = "gpt-4o"
   
    st.title("Market Segmentation Analysis")
    st.write(f"Using model: {model_name} with temperature: {temperature}")
   
    # Add a description
    st.markdown("""
    This tool helps analyze market segments based on various criteria:
    - Demographics (age, gender, income)
    - Geographic location
    - Customer needs and preferences
    - Industry-specific factors
    """)
   
    # Create tabs for different functionalities - Added new "Market Size Analysis" tab
    tab2 = st.tabs(["Market Size Analysis"])
 
    # Fix: Access the first (and only) tab in the tuple using index [0]
    with tab2[0]:
        st.subheader("Market Size Analysis (TAM, SAM, SOM)")
        st.markdown("""
        Analyze the Total Addressable Market (TAM), Serviceable Available Market (SAM),
        and Serviceable Obtainable Market (SOM) for selected industry segments based on Form 10Q reports.
        """)
       
        # Get the selected segment from session state or let user select one
        selected_segment = st.session_state.get("selected_segment")
       
        # Allow user to select a different segment for analysis
        segments = ["Skin Care Segment", "Healthcare - Diagnostic", "Pharmaceutical"]
        try:
            # Try to get segments from the API
            segment_url = get_mcp_server_url("segment")
            response = requests.get(f"{segment_url}/segments", timeout=2)
            if response.status_code == 200:
                segments = response.json().get("segments", segments)
        except Exception as e:
            st.warning(f"Could not retrieve segments from API: {str(e)}")
       
        selected_segment = st.selectbox(
            "Select industry segment for analysis:",
            options=segments,
            index=segments.index(selected_segment) if selected_segment in segments else 0
        )
       
        # Button to perform analysis
        col1, col2 = st.columns([1, 3])
        with col1:
            perform_analysis = st.button("Analyze Market Size")
        with col2:
            refresh_analysis = st.button("Clear & Refresh")
           
        if refresh_analysis:
            # Clear any cached results
            if "market_size_result" in st.session_state:
                del st.session_state.market_size_result
               
        if perform_analysis or "market_size_result" in st.session_state:
            with st.spinner(f"Analyzing market size for {selected_segment}..."):
                try:
                    # Use cached result if available, otherwise fetch from server
                    if "market_size_result" in st.session_state and not refresh_analysis:
                        result = st.session_state.market_size_result
                        st.info("Using cached analysis results. Click 'Clear & Refresh' for fresh data.")
                    else:
                        # Direct API call to get market size data
                        segment_url = get_mcp_server_url(selected_segment)
                       
                        response = requests.post(
                            f"{segment_url}/direct/analyze_market_size",
                            params={"segment": selected_segment},
                            timeout=60
                        )
                       
                        if response.status_code == 200:
                            result = response.json()
                            # Store in session state
                            st.session_state.market_size_result = result
                        else:
                            st.error(f"Error: Server returned status {response.status_code}")
                            st.code(response.text)
                            return
                           
                    # Replace the display section with this improved version:
                    if result.get("status") == "success":
                        st.success("Analysis complete!")
                       
                        # Display market summary with better formatting
                        if result.get("market_summary"):
                            st.markdown(result["market_summary"])
                       
                        # Display metrics in three columns with better error handling and formatting
                        st.subheader("Market Size Metrics")
                        col1, col2, col3 = st.columns(3)
                       
                        with col1:
                            tam_value = result["market_size"]["TAM"] if result["market_size"]["TAM"] else "Not available"
                            st.metric("Total Addressable Market (TAM)", tam_value)
                            st.caption("The total market demand for a product or service")
                       
                        with col2:
                            sam_value = result["market_size"]["SAM"] if result["market_size"]["SAM"] else "Not available"
                            st.metric("Serviceable Available Market (SAM)", sam_value)
                            st.caption("The segment of TAM targeted by your products/services")
                       
                        with col3:
                            som_value = result["market_size"]["SOM"] if result["market_size"]["SOM"] else "Not available"
                            st.metric("Serviceable Obtainable Market (SOM)", som_value)
                            st.caption("The portion of SAM that can be captured")
                       
                        # Display Data Sources section
                        st.subheader("Data Sources")
                        with st.expander("View Source Information", expanded=True):
                            st.markdown(f"**Documents analyzed:** {result.get('documents_analyzed', 15)}")
                           
                            # Show companies analyzed
                            if result.get("companies_analyzed") and len(result["companies_analyzed"]) > 0:
                                companies = ", ".join(result["companies_analyzed"])
                                st.markdown(f"**Companies analyzed:** {companies}")
                           
                            # Show sources in a list format
                            if result.get("sources") and len(result["sources"]) > 0:
                                st.markdown("**Source documents:**")
                                for source in result["sources"]:
                                    st.markdown(f"- {source}")
                    else:
                        st.error(f"Error: {result.get('message', 'Unknown error')}")
                       
                        # Show fallback data if available
                        if result.get("market_size"):
                            st.warning("Showing available data despite errors:")
                           
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                tam_value = result["market_size"].get("TAM") or "Not available"
                                st.metric("Total Addressable Market (TAM)", tam_value)
                           
                            with col2:
                                sam_value = result["market_size"].get("SAM") or "Not available"
                                st.metric("Serviceable Available Market (SAM)", sam_value)
                           
                            with col3:
                                som_value = result["market_size"].get("SOM") or "Not available"
                                st.metric("Serviceable Obtainable Market (SOM)", som_value)
                               
                except Exception as e:
                    st.error(f"Error analyzing market size: {str(e)}")
                    st.code(f"Exception details: {type(e).__name__}: {str(e)}")
                    st.info("Try refreshing the page or selecting a different segment.")
       
        # Replace the Segment Data Search section with this improved version
        st.subheader("Segment Data Search")
        st.markdown("Search for specific information within segment Form 10Q reports")
 
        search_query = st.text_input("Enter search query:", placeholder="market growth rate in diagnostic segment")
        search_button = st.button("Search")
 
        if search_button and search_query:
            with st.spinner("Searching relevant documents and generating insights..."):
                try:
                    # Get the server URL for the selected segment
                    segment_url = get_mcp_server_url(selected_segment)
                   
                    # Call the vector search endpoint
                    response = requests.post(
                        f"{segment_url}/direct/vector_search_and_summarize",
                        params={"query": search_query, "segment": selected_segment, "top_k": 5},
                        timeout=90  # Increased timeout for LLM processing
                    )
                   
                    if response.status_code == 200:
                        result = response.json()
                       
                        if result.get("status") == "success":
                            st.success("Analysis complete!")
                           
                            # Display the answer in a highlighted box
                            st.markdown("### Answer")
                            st.info(result["answer"])
                           
                            # Display source information
                            st.markdown("### Sources")
                            if "chunk_count" in result:
                                st.caption(f"Analysis based on {result['chunk_count']} relevant document excerpts")
                           
                            # Display detailed chunks in an expander
                            with st.expander("View Source Excerpts"):
                                for i, chunk in enumerate(result.get("chunks", [])):
                                    st.markdown(f"**Excerpt {i+1}:**")
                                   
                                    # Try to display source information
                                    if i < len(result.get("sources", [])) and result["sources"][i]:
                                        source = result["sources"][i]
                                        source_info = []
                                        if "company" in source:
                                            source_info.append(f"Company: {source['company']}")
                                        if "date" in source:
                                            source_info.append(f"Date: {source['date']}")
                                        if "file" in source:
                                            source_info.append(f"File: {source['file']}")
                                        if source_info:
                                            st.caption(" | ".join(source_info))
                                   
                                    # Display chunk text in a code block for better readability
                                    st.text_area("", chunk, height=100, key=f"chunk_{i}")
                                    st.markdown("---")
                        else:
                            st.error(f"Search failed: {result.get('message', 'Unknown error')}")
                    else:
                        st.error(f"API request failed with status code: {response.status_code}")
                        st.code(response.text)
               
                except Exception as e:
                    st.error(f"Search failed: {str(e)}")
                   
show()