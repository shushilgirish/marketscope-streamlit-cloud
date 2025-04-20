import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# Define functions that would usually import from other parts of the project
def sidebar():
    with st.sidebar:
        st.title("MarketScope AI")
        
        # Initialize session state for segment
        if "selected_segment" not in st.session_state:
            st.session_state.selected_segment = "Skin Care Segment"
        
        # Define mock segments
        segments = [
            "Skin Care Segment", 
            "Healthcare - Diagnostic", 
            "Pharmaceutical", 
            "Supplements", 
            "Wearables"
        ]
        
        st.session_state.selected_segment = st.selectbox(
            "Select Healthcare Segment",
            options=segments,
            index=segments.index(st.session_state.selected_segment) if st.session_state.selected_segment in segments else 0
        )
        
        # Display connection status
        st.markdown("### Connection Status")
        st.success("Cloud Deployment - Connected")
        st.caption("This is a cloud deployment with simulated data")

# Set page title and layout
st.set_page_config(
    page_title="Market Segmentation",
    page_icon="📊",
    layout="wide"
)

# Show the page content
def show():
    sidebar()
    
    # Title and introduction
    st.title("📊 Market Segmentation Analysis")
    
    st.markdown("""
    Analyze market segmentation data for various healthcare segments to identify
    target markets, competitive positioning, and growth opportunities.
    """)
    
    # Get selected segment
    segment = st.session_state.get("selected_segment")
    if not segment:
        st.warning("Please select a segment in the sidebar first.")
        return
    
    st.info(f"Analyzing segmentation for: **{segment}**")
    
    # Segment analysis tabs
    tab1, tab2, tab3 = st.tabs(["Demographics", "Geographic Distribution", "Behavioral Patterns"])
    
    with tab1:
        st.header("Demographic Segmentation")
        
        # Create demo data
        age_groups = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        
        # Different values based on segment
        if "Skin Care" in segment:
            values = [15, 28, 22, 18, 12, 5]
            top_income = "Upper-middle income"
            top_education = "College degree"
        elif "Diagnostic" in segment:
            values = [5, 12, 18, 25, 22, 18]
            top_income = "High income"
            top_education = "Graduate degree"
        elif "Pharmaceutical" in segment:
            values = [8, 15, 20, 22, 20, 15]
            top_income = "Middle income"
            top_education = "College degree"
        else:
            values = [12, 20, 25, 20, 15, 8]
            top_income = "Middle income"
            top_education = "College degree"
            
        # Create dataframe and chart
        df_age = pd.DataFrame({
            "Age Group": age_groups,
            "Percentage": values
        })
        
        st.subheader("Age Distribution")
        st.bar_chart(df_age.set_index("Age Group"))
        
        # Income distribution
        st.subheader("Income Level Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            income_data = pd.DataFrame({
                "Income Level": ["Low", "Lower-middle", "Middle", "Upper-middle", "High"],
                "Percentage": [10, 15, 25, 35, 15]
            })
            
            st.bar_chart(income_data.set_index("Income Level"))
            st.markdown(f"**Top Consumer Group:** {top_income}")
            
        with col2:
            st.subheader("Education Level")
            education_data = pd.DataFrame({
                "Education": ["High School", "Some College", "College Degree", "Graduate Degree"],
                "Percentage": [20, 25, 35, 20]
            })
            
            st.bar_chart(education_data.set_index("Education"))
            st.markdown(f"**Most Common Education Level:** {top_education}")
    
    with tab2:
        st.header("Geographic Distribution")
        
        # Simulate map data
        st.subheader("Regional Market Penetration")
        
        # Create columns for regional data
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Simulated map - in a real app, we would use a real map visualization
            st.image("https://placehold.co/800x400/1F2937/FFFFFF?text=Geographic+Distribution+Map", 
                     caption="Market penetration by region", use_column_width=True)
        
        with col2:
            region_data = pd.DataFrame({
                "Region": ["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East & Africa"],
                "Market Share (%)": [45, 30, 15, 7, 3]
            })
            
            st.dataframe(region_data)
            
            st.markdown("""
            ### Key Geographic Insights:
            
            - Highest penetration in urban areas
            - Growing adoption in emerging markets
            - Significant opportunities in Asia-Pacific
            """)
            
        # Urban vs. Rural distribution
        st.subheader("Urban vs. Rural Distribution")
        urban_rural = pd.DataFrame({
            "Setting": ["Urban", "Suburban", "Rural"],
            "Percentage": [65, 25, 10]
        })
        
        st.bar_chart(urban_rural.set_index("Setting"))
    
    with tab3:
        st.header("Behavioral Patterns")
        
        # Create purchase frequency data
        purchase_freq = pd.DataFrame({
            "Frequency": ["Weekly", "Monthly", "Quarterly", "Annually", "One-time"],
            "Percentage": [10, 35, 25, 20, 10]
        })
        
        st.subheader("Purchase Frequency")
        st.bar_chart(purchase_freq.set_index("Frequency"))
        
        # Customer loyalty data
        st.subheader("Customer Loyalty")
        
        loyalty_data = pd.DataFrame({
            "Type": ["Loyal", "Occasional", "New Customers"],
            "Percentage": [40, 35, 25]
        })
        
        # Create a pie chart
        fig, ax = plt.subplots()
        ax.pie(loyalty_data["Percentage"], labels=loyalty_data["Type"], autopct='%1.1f%%')
        ax.axis('equal')
        st.pyplot(fig)
        
        # Usage patterns
        st.subheader("Usage Patterns")
        st.markdown("""
        ### Key Behavioral Insights:
        
        - Most customers purchase products on a monthly basis
        - Loyal customers account for 40% of sales volume
        - Brand switching is common among 35% of customers
        - Digital channels are preferred by 72% of customers
        - Product reviews heavily influence purchase decisions
        """)
        
        # Price sensitivity analysis
        st.subheader("Price Sensitivity Analysis")
        
        price_data = pd.DataFrame({
            "Price Range": ["Premium", "Mid-range", "Budget", "Economy"],
            "Market Share (%)": [25, 40, 25, 10]
        })
        
        st.bar_chart(price_data.set_index("Price Range"))

# Run the application
show()
