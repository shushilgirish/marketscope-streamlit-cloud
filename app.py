import streamlit as st
import pandas as pd
import os

# Set page config to full width
st.set_page_config(layout="wide", page_title="MarketScope AI", page_icon="📊")

# Add custom CSS
try:
    with open('styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    # Basic styles if file not found
    st.markdown("""
    <style>
        .main { background-color: #0e1117; }
        h1, h2, h3 { color: #ffffff; }
    </style>
    """, unsafe_allow_html=True)

# Define mock segments
MOCK_SEGMENTS = [
    "Skin Care Segment", 
    "Healthcare - Diagnostic", 
    "Pharmaceutical", 
    "Supplements", 
    "Wearables"
]

# Initialize session state variables that need to be shared across pages
if "selected_segment" not in st.session_state:
    st.session_state.selected_segment = MOCK_SEGMENTS[0]
if "sales_data" not in st.session_state:
    st.session_state.sales_data = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "trends_result" not in st.session_state:
    st.session_state.trends_result = None
if "market_size_result" not in st.session_state:
    st.session_state.market_size_result = None

def sidebar():
    """Create sidebar with configuration options"""
    with st.sidebar:
        st.title("MarketScope AI")
        
        # Initialize session state for segment
        if "selected_segment" not in st.session_state:
            st.session_state.selected_segment = MOCK_SEGMENTS[0]
        
        # Use mock segments for cloud deployment
        segments = MOCK_SEGMENTS
        
        st.session_state.selected_segment = st.selectbox(
            "Select Healthcare Segment",
            options=segments,
            index=segments.index(st.session_state.selected_segment) if st.session_state.selected_segment in segments else 0
        )
        
        # Display connection status
        st.markdown("### Connection Status")
        st.success("Cloud Deployment - Connected")
        st.caption("This is a cloud deployment with simulated data")

def get_server_status():
    """Mock server status check"""
    return {
        "unified": "healthy",
        "market_analysis": "healthy", 
        "sales_analytics": "healthy",
        "segment": "healthy",
        "snowflake": "healthy"
    }

def show():
    """Show main app interface with market size analysis"""
    sidebar()
    
    st.title("MarketScope AI - Market Size Analysis")
    
    # Check if a segment is selected
    if st.session_state.get("selected_segment"):
        segment = st.session_state.selected_segment
        st.info(f"Currently analyzing: **{segment}**")
    
    # Add Server Status component
    with st.expander("MCP Server Status", expanded=False):
        st.markdown("### MarketScope MCP Server Status")
        st.markdown("Check the status of all backend MCP servers that power MarketScope AI:")
        
        # Get status of all MCP servers
        server_status = get_server_status()
        
        # Display server status in a clean format
        for server_name, status in server_status.items():
            col1, col2 = st.columns([1, 3])
            with col1:
                # All servers are "healthy" in the cloud demo
                st.markdown("✅")
            
            with col2:
                # Format server name for display
                display_name = server_name.replace("_", " ").title()
                if server_name == "unified":
                    display_name += " (Main)"
                
                st.markdown(f"**{display_name}**: Connected")
        
        # Add a button to refresh server status
        if st.button("Refresh Status"):
            st.experimental_rerun()
    
    # Market Size Analysis Section
    st.header("Market Size Analysis")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Market Size Metrics")
        
        # Create tabs for different market views
        tab1, tab2, tab3 = st.tabs(["Overview", "Segmentation", "Forecast"])
        
        with tab1:
            # Show mock market size data
            if st.session_state.selected_segment:
                # Create three columns for TAM, SAM, SOM
                tam_col, sam_col, som_col = st.columns(3)
                
                with tam_col:
                    st.metric(
                        "Total Addressable Market (TAM)", 
                        "$189.3B", 
                        "+5.2%"
                    )
                    
                with sam_col:
                    st.metric(
                        "Serviceable Available Market (SAM)", 
                        "$76.5B", 
                        "+7.8%"
                    )
                    
                with som_col:
                    st.metric(
                        "Serviceable Obtainable Market (SOM)", 
                        "$15.2B", 
                        "+12.4%"
                    )
                
                st.caption("Market data based on industry analysis and Form 10Q reports")
            else:
                st.info("Please select a segment to view market size analysis.")
        
        with tab2:
            st.write("## Market Segmentation")
            
            # Create a mock pie chart for market segmentation
            data = pd.DataFrame({
                'Segment': ['Premium', 'Mid-tier', 'Budget', 'Professional'],
                'Market Share': [35, 30, 20, 15]
            })
            
            st.write("### Market Share by Segment")
            st.bar_chart(data.set_index('Segment'))
            
        with tab3:
            st.write("## Market Growth Forecast")
            
            # Create a mock line chart for growth forecast
            years = ['2022', '2023', '2024', '2025', '2026']
            forecast = [150, 175, 210, 255, 315]
            
            forecast_df = pd.DataFrame({
                'Year': years,
                'Market Size ($B)': forecast
            })
            
            st.line_chart(forecast_df.set_index('Year'))
            
            st.caption("Forecast based on historical growth rates and industry trends")
    
    with col2:
        st.subheader("Industry Insights")
        
        st.markdown("""
        ### Key Growth Drivers
        
        - Increasing health consciousness among consumers
        - Growing aging population
        - Rising disposable income in emerging markets
        - Technological advancements in product formulations
        - E-commerce expansion enabling global reach
        
        ### Market Challenges
        
        - Regulatory compliance requirements
        - Intense competition from established brands
        - Raw material cost fluctuations
        - Consumer preference shifts
        - Sustainability demands
        """)
        
        with st.expander("Competitive Landscape"):
            st.markdown("""
            The market is dominated by key players including:
            
            - Johnson & Johnson
            - L'Oreal
            - Unilever
            - Procter & Gamble
            - Estée Lauder
            
            Emerging players are gaining market share through digital-first strategies and innovative product formulations.
            """)
    
    # Add a section for custom analysis
    st.header("Custom Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        analysis_query = st.text_area(
            "Enter your market analysis question",
            placeholder="e.g., What are the key growth trends in the skin care market?",
            height=100
        )
    
    with col2:
        st.write("### Analysis Options")
        include_forecast = st.checkbox("Include growth forecast", value=True)
        include_competitors = st.checkbox("Include competitor data", value=True)
        st.caption("Cloud deployment demo options")
        
        if st.button("Generate Analysis", type="primary"):
            if analysis_query:
                with st.spinner("Generating market analysis..."):
                    # Simulate processing time
                    import time
                    time.sleep(2)
                    
                    st.subheader("Analysis Results")
                    st.markdown(f"""
                    # Market Analysis: {st.session_state.selected_segment}
                    
                    ## Key Insights
                    
                    Based on your query: "{analysis_query}"
                    
                    The {st.session_state.selected_segment} market is projected to grow at a CAGR of 8.7% 
                    over the next five years. Key drivers include technological innovation, 
                    increasing health awareness, and expanding distribution channels in emerging markets.
                    
                    ## Strategic Recommendations
                    
                    1. **Product Innovation Focus**: Invest in R&D for natural and sustainable formulations
                    2. **Target Demographics**: Expand marketing efforts to reach younger consumers
                    3. **Digital Transformation**: Enhance direct-to-consumer e-commerce capabilities
                    4. **Strategic Partnerships**: Explore collaborations with complementary brands
                    
                    *This is a demonstration of the cloud deployment capabilities.*
                    """)
                    
                    # Create a sample visualization if forecast is included
                    if include_forecast:
                        data_forecast = pd.DataFrame({
                            'Year': [2022, 2023, 2024, 2025, 2026, 2027],
                            'Revenue ($M)': [320, 358, 412, 486, 572, 673]
                        })
                        
                        st.subheader("Revenue Growth Projection")
                        st.line_chart(data_forecast.set_index('Year'))
                    
                    # Add competitor comparison if requested
                    if include_competitors:
                        st.subheader("Competitive Landscape")
                        
                        comp_data = pd.DataFrame({
                            'Competitor': ['Company A', 'Company B', 'Company C', 'Company D', 'Others'],
                            'Market Share (%)': [28, 22, 18, 12, 20]
                        })
                        
                        st.bar_chart(comp_data.set_index('Competitor'))
            else:
                st.warning("Please enter an analysis question.")

# Call the main function
show()
