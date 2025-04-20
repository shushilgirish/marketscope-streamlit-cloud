import streamlit as st
import pandas as pd
import numpy as np
import time

# Define sidebar function for cloud deployment
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

# Set page config to full width
st.set_page_config(
    page_title="Sales & Marketing Analysis",
    page_icon="📈",
    layout="wide"
)

# Generate mock sales data based on segment
def generate_mock_sales_data(segment):
    # Create date range for past 12 months
    dates = pd.date_range(end=pd.Timestamp.now(), periods=12, freq='M')
    
    # Different product names based on segment
    if "Skin Care" in segment:
        products = ["Facial Cleanser", "Anti-Aging Serum", "Moisturizer", "Sunscreen", "Eye Cream"]
    elif "Diagnostic" in segment:
        products = ["Glucose Monitor", "Blood Pressure Meter", "ECG Device", "Thermometer", "Oximeter"]
    elif "Pharmaceutical" in segment:
        products = ["Pain Relief", "Allergy Medicine", "Antacid", "Cold & Flu", "Vitamins"]
    else:
        products = ["Product A", "Product B", "Product C", "Product D", "Product E"]
    
    # Generate random data
    data = []
    for date in dates:
        for product in products:
            # More realistic values based on segment
            if "Skin Care" in segment:
                units_sold = np.random.randint(5000, 15000)
                price = np.random.uniform(15, 75)
            elif "Diagnostic" in segment:
                units_sold = np.random.randint(1000, 8000)
                price = np.random.uniform(50, 250)
            else:
                units_sold = np.random.randint(2000, 10000)
                price = np.random.uniform(20, 120)
                
            revenue = units_sold * price
            cost = revenue * np.random.uniform(0.4, 0.6)
            profit = revenue - cost
            margin = profit / revenue
            
            data.append({
                "date": date,
                "product": product,
                "units_sold": units_sold,
                "price": price,
                "revenue": revenue,
                "cost": cost,
                "profit": profit,
                "margin": margin
            })
    
    return pd.DataFrame(data)

# Generate mock marketing campaign data
def generate_mock_campaign_data(segment):
    # Create different campaign types and metrics based on segment
    if "Skin Care" in segment:
        campaigns = [
            "Social Media Campaign", 
            "Influencer Partnership", 
            "Email Newsletter", 
            "Seasonal Promotion", 
            "Product Launch"
        ]
        channels = ["Instagram", "Facebook", "Email", "YouTube", "TikTok"]
    else:
        campaigns = [
            "Professional Conference", 
            "Email Marketing", 
            "LinkedIn Campaign", 
            "Educational Webinar", 
            "Product Demo"
        ]
        channels = ["LinkedIn", "Email", "Professional Journals", "Conferences", "Website"]
    
    # Generate random data
    data = []
    for campaign in campaigns:
        for channel in channels:
            impressions = np.random.randint(10000, 100000)
            clicks = impressions * np.random.uniform(0.01, 0.05)
            conversions = clicks * np.random.uniform(0.05, 0.15)
            cost = np.random.uniform(2000, 10000)
            revenue = conversions * np.random.uniform(50, 200)
            roi = (revenue - cost) / cost
            
            data.append({
                "campaign": campaign,
                "channel": channel,
                "impressions": int(impressions),
                "clicks": int(clicks),
                "conversions": int(conversions),
                "cost": cost,
                "revenue": revenue,
                "roi": roi
            })
    
    return pd.DataFrame(data)

def show():
    """Show the sales and marketing analysis page"""
    sidebar()
    
    st.title("📈 Sales & Marketing Analysis")
    
    # Get segment selection from session state
    segment = st.session_state.get("selected_segment", None)
    
    if not segment:
        st.warning("Please select a segment on the Home page first.")
        return
        
    st.markdown(f"""
    ## Sales & Marketing Analysis for {segment}
    
    This tool provides comprehensive analytics for sales performance and marketing campaign effectiveness
    to help you optimize your go-to-market strategy for {segment}.
    """)
    
    # Create tabs for different analysis types
    sales_tab, marketing_tab, roi_tab = st.tabs(["Sales Performance", "Marketing Campaigns", "ROI Analysis"])
    
    # ---- SALES PERFORMANCE TAB ----
    with sales_tab:
        st.header("Sales Performance Analysis")
        
        # Filter options
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            time_period = st.selectbox(
                "Time Period",
                ["Last 3 Months", "Last 6 Months", "Last 12 Months", "Year to Date"],
                index=2
            )
        with col2:
            metrics = st.multiselect(
                "Metrics",
                ["Revenue", "Units Sold", "Profit", "Margin"],
                default=["Revenue"]
            )
        with col3:
            if st.button("Refresh Data", key="refresh_sales"):
                # Simulate data refresh
                with st.spinner("Refreshing data..."):
                    time.sleep(1.5)
                st.success("Data refreshed!")
                
        # Generate mock sales data
        with st.spinner("Loading sales data..."):
            time.sleep(1)
            sales_data = generate_mock_sales_data(segment)
        
        # Display summary metrics
        st.subheader("Sales Summary")
        
        total_revenue = sales_data["revenue"].sum()
        total_profit = sales_data["profit"].sum()
        avg_margin = sales_data["margin"].mean() * 100
        
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Total Revenue", f"${total_revenue/1000000:.2f}M", "+12.3%")
        metric2.metric("Total Profit", f"${total_profit/1000000:.2f}M", "+8.7%")
        metric3.metric("Average Margin", f"{avg_margin:.1f}%", "+1.5%")
        
        # Revenue by product
        st.subheader("Revenue by Product")
        product_revenue = sales_data.groupby("product")["revenue"].sum().reset_index()
        st.bar_chart(product_revenue.set_index("product"))
        
        # Revenue trend over time
        st.subheader("Revenue Trend")
        revenue_trend = sales_data.groupby("date")["revenue"].sum().reset_index()
        st.line_chart(revenue_trend.set_index("date"))
        
        # Detailed data table
        with st.expander("View Detailed Sales Data"):
            st.dataframe(
                sales_data.groupby(["date", "product"]).agg({
                    "units_sold": "sum",
                    "revenue": "sum",
                    "profit": "sum",
                    "margin": "mean"
                }).reset_index().sort_values("date", ascending=False),
                use_container_width=True
            )
    
    # ---- MARKETING CAMPAIGNS TAB ----
    with marketing_tab:
        st.header("Marketing Campaign Analysis")
        
        # Generate mock campaign data
        with st.spinner("Loading marketing data..."):
            time.sleep(1)
            campaign_data = generate_mock_campaign_data(segment)
        
        # Summary metrics
        total_spend = campaign_data["cost"].sum()
        total_revenue = campaign_data["revenue"].sum()
        overall_roi = (total_revenue - total_spend) / total_spend * 100
        
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Marketing Spend", f"${total_spend/1000:.1f}K")
        metric2.metric("Campaign Revenue", f"${total_revenue/1000:.1f}K")
        metric3.metric("Overall ROI", f"{overall_roi:.1f}%")
        
        # Campaign performance
        st.subheader("Campaign Performance")
        
        campaign_perf = campaign_data.groupby("campaign").agg({
            "impressions": "sum",
            "clicks": "sum",
            "conversions": "sum",
            "cost": "sum",
            "revenue": "sum",
            "roi": "mean"
        }).reset_index()
        
        # Calculate CTR and CVR
        campaign_perf["ctr"] = campaign_perf["clicks"] / campaign_perf["impressions"] * 100
        campaign_perf["cvr"] = campaign_perf["conversions"] / campaign_perf["clicks"] * 100
        
        # Display metrics by campaign
        st.subheader("ROI by Campaign")
        st.bar_chart(campaign_perf.set_index("campaign")["roi"] * 100)
        
        # Channel performance
        st.subheader("Performance by Channel")
        
        channel_perf = campaign_data.groupby("channel").agg({
            "impressions": "sum",
            "clicks": "sum",
            "conversions": "sum",
            "cost": "sum",
            "revenue": "sum",
            "roi": "mean"
        }).reset_index()
        
        channel_perf["ctr"] = channel_perf["clicks"] / channel_perf["impressions"] * 100
        channel_perf["cvr"] = channel_perf["conversions"] / channel_perf["clicks"] * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ROI by Channel")
            st.bar_chart(channel_perf.set_index("channel")["roi"] * 100)
        
        with col2:
            st.subheader("Conversion Rate by Channel")
            st.bar_chart(channel_perf.set_index("channel")["cvr"])
        
        # Detailed campaigns data
        with st.expander("View Detailed Campaign Data"):
            st.dataframe(campaign_data)
    
    # ---- ROI ANALYSIS TAB ----
    with roi_tab:
        st.header("ROI & Investment Analysis")
        
        # Combine sales and marketing data for ROI analysis
        sales_summary = sales_data.groupby("date").agg({
            "revenue": "sum", 
            "profit": "sum"
        }).reset_index()
        
        campaign_summary = campaign_data.groupby("campaign").agg({
            "cost": "sum", 
            "revenue": "sum", 
            "roi": "mean"
        }).reset_index()
        
        # Overall business metrics
        total_sales = sales_summary["revenue"].sum()
        total_marketing = campaign_summary["cost"].sum()
        marketing_percent = (total_marketing / total_sales) * 100
        
        st.subheader("Marketing Investment Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Marketing as % of Sales", f"{marketing_percent:.1f}%")
            st.metric("Marketing Efficiency Ratio", f"{total_sales/total_marketing:.2f}")
            
            st.write("### Investment Recommendations")
            st.markdown(f"""
            Based on the analysis of your {segment} performance:
            
            1. **Increase investment** in channels with highest ROI
            2. **Optimize campaigns** with lower performance
            3. **Test new approaches** in promising segments
            4. **Reallocate budget** from underperforming channels
            """)
            
        with col2:
            # Calculate ROAS by channel
            channel_roas = campaign_data.groupby("channel").agg({
                "cost": "sum",
                "revenue": "sum"
            }).reset_index()
            
            channel_roas["roas"] = channel_roas["revenue"] / channel_roas["cost"]
            
            st.subheader("Return on Ad Spend by Channel")
            st.bar_chart(channel_roas.set_index("channel")["roas"])
            
            st.caption("ROAS = Revenue / Ad Spend")
        
        # Marketing efficiency metrics
        st.subheader("Marketing Efficiency Metrics")
        
        efficiency_data = pd.DataFrame({
            "Metric": [
                "Customer Acquisition Cost (CAC)",
                "Customer Lifetime Value (CLV)",
                "CLV to CAC Ratio",
                "Payback Period (Months)",
                "Marketing ROI"
            ],
            "Value": [
                f"${np.random.uniform(50, 150):.2f}",
                f"${np.random.uniform(300, 900):.2f}",
                f"{np.random.uniform(2.5, 6.0):.1f}",
                f"{np.random.uniform(3, 12):.1f}",
                f"{np.random.uniform(150, 350):.1f}%"
            ]
        })
        
        st.table(efficiency_data)

# Call the main function
show()
