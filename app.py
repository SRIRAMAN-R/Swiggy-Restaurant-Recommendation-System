import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# Caching data loading functions
# -------------------------------
@st.cache_data
def load_cleaned_data(file_path):
    return pd.read_csv(file_path, index_col=0)

@st.cache_data
def load_encoded_data(file_path):
    return joblib.load(file_path, mmap_mode='r')

@st.cache_data
def kmeans_clustering(encoded_data, n_clusters=5):
    numeric_data = encoded_data.select_dtypes(include=[np.number])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    encoded_data['Cluster'] = kmeans.fit_predict(numeric_data)
    return encoded_data, kmeans

@st.cache_data
def merge_data(cleaned_data, encoded_data_with_clusters):
    return pd.merge(cleaned_data, encoded_data_with_clusters[['Cluster', 'name']], on='name', how='left')

# -------------------------------
# Load the data
# -------------------------------
cleaned_data = load_cleaned_data("cleaned_data.csv")

# Optimize memory usage by reducing data types for 'rating' and 'cost'
cleaned_data['rating'] = cleaned_data['rating'].astype('float32')
cleaned_data['cost'] = cleaned_data['cost'].astype('float32')

# Now load the encoded data
encoded_data = load_encoded_data("encoded_data.joblib")

# Perform clustering
encoded_data_with_clusters, kmeans_model = kmeans_clustering(encoded_data)

# Merge for visualization
merged_data = merge_data(cleaned_data, encoded_data_with_clusters)

# -------------------------------
# Streamlit App UI
# -------------------------------
st.title("🍽️ Swiggy Restaurant Recommender")

# Sidebar Navigation
page = st.sidebar.radio("Choose a page", ( "Recommendation"))

if page == "Recommendation":
    st.subheader("🔍 Filter Restaurants Based on Your Preferences")

    city = st.selectbox('Select City', merged_data['city'].unique())
    cuisine = st.selectbox('Select Cuisine', merged_data['cuisine'].unique())
    rating = st.slider('Select Rating Range', 1.0, 5.0, (3.0, 5.0), step=0.5)
    cost = st.slider('Select Cost Range', 0, 3000, (300, 1500), step=200)
    rating_count = st.slider('Select Rating Count Range', 0, 10000, (100, 5000), step=500)

    filtered_data = merged_data[
        (merged_data['city'] == city) &
        (merged_data['cuisine'].str.contains(cuisine, case=False)) &
        (merged_data['rating'].between(rating[0], rating[1])) &
        (merged_data['cost'].between(cost[0], cost[1])) &
        (merged_data['rating_count'].between(rating_count[0], rating_count[1]))
    ]

    st.write("### 🔎 Filtered Restaurants")
    st.dataframe(filtered_data.loc[:, ['name', 'city', 'rating', 'rating_count', 'cost', 'cuisine']])

