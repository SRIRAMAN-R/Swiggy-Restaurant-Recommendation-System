# Swiggy Restaurant Recommendation System

A complete end-to-end recommendation system that helps users discover top restaurants on Swiggy based on their preferences. Built using **Python**, **scikit-learn**, and **Streamlit**, the project includes data preprocessing, encoding, clustering, and a modern UI for exploration and recommendations.

---

## 📌 Features

- ✅ Clean and preprocess raw Swiggy data
- 🔍 One-hot encoding for cities and cuisines
- 🎯 KMeans clustering to group similar restaurants
- 🧠 Recommend top 5 similar restaurants based on cluster
- 🌐 Streamlit UI to explore restaurants with filters like city, cuisine, rating, cost, and rating count
- 📊 Visual representation (optional enhancements supported)

---

## 🛠️ Tech Stack

- **Python** (pandas, numpy, scikit-learn, joblib)
- **Streamlit** for interactive frontend
- **KMeans** clustering algorithm
- **OneHotEncoder** and **MultiLabelBinarizer** for categorical encoding
- **matplotlib** for optional charts/visuals

---

## 📂 Project Structure

- swiggy.csv # Raw dataset
- recommendation.ipynb #jupyter notebook file
- app.py # Streamlit application file

---

## Run the file

- Run ipynb File
- Run streamlit file using ```streamlit run app.py```

## Explore Recommendations

- Use the dropdowns and sliders in the UI to filter restaurants
- Instantly see restaurant recommendations that match your selected city, cuisine, rating, cost, and more!
