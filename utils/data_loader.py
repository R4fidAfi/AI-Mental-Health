import pandas as pd
import streamlit as st
from pathlib import Path

@st.cache_data
def load_data():
    """Load and return the cleaned mental health dataset."""
    try:
        df = pd.read_csv("data/Student_Mental_Health.csv")
        
        # Clean column names
        df.rename(columns={
            "Choose your gender": "gender",
            "Age": "age",
            "What is your course?": "course",
            "Your current year of Study": "year",
            "What is your CGPA?": "cgpa",
            "Marital status": "marital_status",
            "Do you have Depression?": "depression",
            "Do you have Anxiety?": "anxiety",
            "Do you have Panic attack?": "panic_attack",
            "Did you seek any specialist for a treatment?": "treatment"
        }, inplace=True)
        
        # Clean data
        df = df.dropna()
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        
        # Convert Yes/No to binary
        for col in ['depression', 'anxiety', 'panic_attack']:
            df[col] = df[col].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}).fillna(0)
        
        df['mental_score'] = df['depression'] + df['anxiety'] + df['panic_attack']
        df['risk_level'] = pd.cut(df['mental_score'], 
                                bins=[-1, 0, 1, 2, 3], 
                                labels=['Low', 'Mild', 'Moderate', 'High'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def get_filtered_data(df, filters):
    """Apply filters to dataframe."""
    filtered = df.copy()
    if filters.get('gender'):
        filtered = filtered[filtered['gender'].isin(filters['gender'])]
    if filters.get('year'):
        filtered = filtered[filtered['year'].isin(filters['year'])]
    if filters.get('course'):
        filtered = filtered[filtered['course'].str.contains('|'.join(filters['course']), case=False, na=False)]
    return filtered