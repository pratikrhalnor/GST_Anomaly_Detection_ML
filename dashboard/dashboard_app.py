import streamlit as st
import pandas as pd
import requests
import io
import plotly.express as px
from datetime import datetime

# Page configuration - Clean and professional
st.set_page_config(
    page_title="GST Fraud Detection",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"  # Remove sidebar
)

# Custom CSS for unique, non-AI look
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
    }
    
    /* Header styling */
    .custom-header {
        background: linear-gradient(135deg, #2c3e50, #34495e);
        color: white;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* File upload area */
    .upload-box {
        background: white;
        padding: 25px;
        border-radius: 10px;
        border: 2px dashed #bdc3c7;
        text-align: center;
        margin: 20px 0;
    }
    
    /* Results container */
    .results-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #e74c3c;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    /* Metrics styling */
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        margin: 8px;
        border-top: 4px solid #3498db;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 6px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #c0392b, #a93226);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(231, 76, 60, 0.3);
    }
    
    /* Success message */
    .success-msg {
        background: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 6px;
        border-left: 4px solid #28a745;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="custom-header">
    <h1 style="margin:0; font-size: 2.5rem;">📋 GST Transaction Analysis</h1>
    <p style="margin:10px 0 0 0; opacity: 0.9;">Detect suspicious patterns in your GST transaction data</p>
</div>
""", unsafe_allow_html=True)

# File Upload Section
st.markdown("""
<div class="upload-box">
    <h3 style="color: #2c3e50; margin-bottom: 15px;">📁 Upload Your Transaction Data</h3>
    <p style="color: #7f8c8d;">Supported format: CSV files with GST transaction records</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose CSV file", 
    type=["csv"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    try:
        # Read and display file info
        df = pd.read_csv(uploaded_file)
        
        # File summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #2c3e50; margin:0;">{len(df):,}</h3>
                <p style="color: #7f8c8d; margin:0;">Total Records</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #2c3e50; margin:0;">{len(df.columns)}</h3>
                <p style="color: #7f8c8d; margin:0;">Columns</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #2c3e50; margin:0;">{uploaded_file.size / 1024:.1f} KB</h3>
                <p style="color: #7f8c8d; margin:0;">File Size</p>
            </div>
            """, unsafe_allow_html=True)

        # Data preview
        with st.expander("🔍 Preview Data", expanded=True):
            st.dataframe(df.head(6), use_container_width=True)

        # Analysis button
        if st.button("🔍 Analyze Transactions for Fraud Patterns"):
            with st.spinner("Analyzing transaction patterns..."):
                try:
                    api_url = "http://127.0.0.1:5000/predict_batch"
                    
                    # Send to API
                    response = requests.post(
                        api_url,
                        files={'file': ('batch.csv', io.BytesIO(uploaded_file.getvalue()), 'text/csv')},
                        timeout=30
                    )

                    if response.status_code == 200:
                        result_json = response.json()
                        result_df = pd.DataFrame(result_json)
                        
                        st.markdown("""
                        <div class="success-msg">
                            <h4 style="margin:0;">✅ Analysis Complete</h4>
                            <p style="margin:5px 0 0 0;">Transaction review finished successfully</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Results Section
                        st.markdown("""
                        <div class="results-box">
                            <h3 style="color: #2c3e50; margin-bottom: 20px;">📈 Analysis Results</h3>
                        </div>
                        """, unsafe_allow_html=True)

                        # Calculate metrics
                        if 'is_anomaly' in result_df.columns:
                            anomalies = result_df[result_df["is_anomaly"] == 1]
                            normal = result_df[result_df["is_anomaly"] == 0]
                            anomaly_rate = (len(anomalies) / len(result_df)) * 100
                        else:
                            anomalies = pd.DataFrame()
                            normal = result_df
                            anomaly_rate = 0

                        # Key metrics in clean layout
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="color: #2c3e50; margin:0;">{len(result_df):,}</h4>
                                <p style="color: #7f8c8d; margin:0;">Total</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="color: #27ae60; margin:0;">{len(normal):,}</h4>
                                <p style="color: #7f8c8d; margin:0;">Normal</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="color: #e74c3c; margin:0;">{len(anomalies):,}</h4>
                                <p style="color: #7f8c8d; margin:0;">Suspicious</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with col4:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="color: #f39c12; margin:0;">{anomaly_rate:.1f}%</h4>
                                <p style="color: #7f8c8d; margin:0;">Risk Rate</p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Only show pie chart (removed bar chart)
                        if len(result_df) > 0 and len(anomalies) > 0:
                            st.markdown("### 📊 Transaction Distribution")
                            
                            # Clean pie chart
                            fig = px.pie(
                                names=['Normal Transactions', 'Suspicious Patterns'],
                                values=[len(normal), len(anomalies)],
                                color=['Normal Transactions', 'Suspicious Patterns'],
                                color_discrete_map={
                                    'Normal Transactions': '#27ae60',
                                    'Suspicious Patterns': '#e74c3c'
                                }
                            )
                            
                            fig.update_layout(
                                showlegend=True,
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(size=12),
                                margin=dict(t=30, b=30)
                            )
                            
                            fig.update_traces(
                                textposition='inside',
                                textinfo='percent+label',
                                marker=dict(line=dict(color='white', width=2))
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)

                        # Brief insights section
                        if len(anomalies) > 0:
                            st.markdown("### 🔍 Key Observations")
                            
                            if anomaly_rate > 20:
                                st.warning("**High attention needed:** Significant number of transactions showing unusual patterns that require review.")
                            elif anomaly_rate > 10:
                                st.info("**Moderate review recommended:** Some transactions display characteristics that merit closer examination.")
                            else:
                                st.success("**Good compliance:** Majority of transactions appear normal with standard business patterns.")
                                
                            st.write(f"- **{len(anomalies)} transactions** flagged for review")
                            st.write(f"- **Risk level:** {anomaly_rate:.1f}% of total transactions")
                            st.write(f"- **Next step:** Download detailed report for investigation")

                        # Download section
                        st.markdown("---")
                        st.markdown("### 📥 Export Results")
                        
                        csv_buffer = io.BytesIO()
                        result_df.to_csv(csv_buffer, index=False)
                        
                        st.download_button(
                            label="💾 Download Full Analysis Report",
                            data=csv_buffer.getvalue(),
                            file_name=f"gst_analysis_{datetime.now().strftime('%d%m%Y')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    else:
                        st.error("Analysis service is temporarily unavailable. Please try again later.")

                except requests.exceptions.RequestException:
                    st.error("Unable to connect to analysis service. Please check your connection.")
                    
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

else:
    # Welcome message when no file uploaded
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px; color: #7f8c8d;">
        <h3>👆 Upload your GST transaction data to begin analysis</h3>
        <p>Get insights into potential suspicious patterns and compliance issues</p>
    </div>
    """, unsafe_allow_html=True)

# Simple footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #95a5a6; font-size: 0.9rem;'>"
    "GST Transaction Analysis Tool • Built for compliance teams"
    "</div>", 
    unsafe_allow_html=True
)