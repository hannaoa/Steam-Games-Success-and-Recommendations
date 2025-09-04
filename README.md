# Steam Games: Success and Recommender
## Overview
Steam is a platform that hosts numerous games and operates in a highly competitive market for developers. This project analyzes key factors influencing the success of games on the Steam game platform and builds a recommendation system using game metadata. By combining exploratory analysis (EDA), machine learning classification, and natural language processing (NLP), the main features were identified for positive game reviews and they provided personalized game suggestions. The main features of price, playtime, downloadable content (DLC), and genre are the strongest in predicting success. Additionally, the content-based recommender system was implemented using TP-IDF and cosine similarity to suggest games based on the text description.

Dataset from FronkonGames on Hugging Face: https://huggingface.co/datasets/FronkonGames/steam-games-dataset

<img width="1618" height="791" alt="image" src="https://github.com/user-attachments/assets/6646094e-fa2f-4e54-b598-37c9f0846e6f" />

## Technologies Used
- Programming Language: Python
- Python Libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - plotly
  - wordcloud
  - streamlit

## Features
- Exploratory Data Analysis (EDA): Visualizations of pricing trends, game distributions, and review patterns
- Game Success Prediction: Logistic Regression and Random Forest models trained to predict success (>= 80% positive review ratio)
- Feature Importance Analysis: Identifies which factors like price, playtime, DLC count, and genre matter most
- Content-Based Recommendation System: Suggests 5 similar games based on descriptions using TF-IDF and cosine similarity
- Interactive Dashboard: Streamlit app with plots, correlation analysis, and recommender system demo

## Technical Details
### Data Collection and Processing
- Dataset: More than 83,000 from the dataset obtained through Stream API and Steam Spy
- Data types: Numerical (price, playtime), Boolean (VR support, multiplayer), text (titles, genres, descriptions)
- Preprocessing: Cleaning text, tokenization, TF-IDF vectorization, correlation analysis, and feature scaling

## Modeling 
- Success Prediction:
  - Target Positive review ratio >= 80%
  - Models: Logistic Regression, Random Forest
  - Best performance: Random Forest to rank feature importance
- Recommendation System:
  - Took the 2,000 most reviewed games on Steam, cleaned descriptions and left only key tokens, vectorized and computed cosine similarity, and returned the 5 most similar games with descriptions

## Deployement
- Streamlit dashboard integrates:
  - EDA visualizations: price vs reviews, genre distributions, word clouds
  - Success factor analysis: feature importance, classification results
  - Interactive game recommender demo

## Contact
GitHub: @hannaoa

LinkedIn: https://www.linkedin.com/in/hannaothmal/

Email: hanna.othmal@mavs.uta.edu
