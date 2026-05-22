import streamlit as st
import pickle
import pandas as pd
import os
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="WKU Feedback System", page_icon="🎓", layout="wide")

# 2. Load ML Model and Vectorizer
try:
    model = pickle.load(open('model.pkl', 'rb'))
    vec = pickle.load(open('vec.pkl', 'rb'))
except FileNotFoundError:
    st.error("Please run 'train.py' first to generate the model!")

# 3. Initialize CSV File for Data Storage
CSV_FILE = 'feedbacks.csv'
if not os.path.exists(CSV_FILE):
    df_init = pd.DataFrame(columns=['Feedback', 'Sentiment'])
    df_init.to_csv(CSV_FILE, index=False)

# 4. Sidebar Navigation
page = st.sidebar.selectbox("Select Page", ["Student Portal", "WKU Admin Dashboard"])

# --- PAGE 1: Student Portal ---
if page == "Student Portal":
    st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🎓 Wolkite University</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #4B5563;'>Student Feedback Portal</h3>", unsafe_allow_html=True)
    st.write("---")
    
    text = st.text_area('Enter Student Review / Feedback:', placeholder="Type feedback here (e.g., The campus wifi is very fast)...")
    
    if st.button('Submit Feedback'):
        if text.strip() == "":
            st.warning("Please enter some text first!")
        else:
            # Prediction
            pred = model.predict(vec.transform([text]))
            sentiment_result = 'Positive' if pred[0] == 1 else 'Negative'
            
            # Append to CSV file
            new_data = pd.DataFrame([[text, sentiment_result]], columns=['Feedback', 'Sentiment'])
            new_data.to_csv(CSV_FILE, mode='a', header=False, index=False)
            
            if sentiment_result == 'Positive':
                st.success("🟢 Thank you! Your positive feedback has been recorded.")
            else:
                st.error("🔴 Thank you! Your feedback has been recorded for review.")

# --- PAGE 2: Admin Dashboard ---
elif page == "WKU Admin Dashboard":
    st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>📊 WKU Administration Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #4B5563;'>Overall Student Feedback Analytics</h4>", unsafe_allow_html=True)
    st.write("---")
    
    # Read CSV Data
    df = pd.read_csv(CSV_FILE)
    
    if df.empty:
        st.info("No feedback recorded yet.")
    else:
        # Calculate Metrics
        total_feedback = len(df)
        pos_count = len(df[df['Sentiment'] == 'Positive'])
        neg_count = len(df[df['Sentiment'] == 'Negative'])
        
        # Display Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Feedbacks", total_feedback)
        col2.metric("Positive 🟢", pos_count)
        col3.metric("Negative 🔴", neg_count)
        
        st.write("---")
        
        # Charts Section
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("Sentiment Distribution")
            fig = px.pie(df, names='Sentiment', color='Sentiment',
                         color_discrete_map={'Positive': '#22C55E', 'Negative': '#EF4444'})
            st.plotly_chart(fig, use_container_width=True)
            
        with col_chart2:
            st.subheader("Recent Feedback History")
            st.dataframe(df.tail(10), use_container_width=True)