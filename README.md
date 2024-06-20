# Restaurant-Sentiment-Analysis
This project aims to classify restaurant reviews as positive or negative using machine learning techniques and natural language processing (NLP).

## Project Overview
The project involves several key steps:

# 1.Data Collection and Preprocessing
Loading and inspecting the dataset.
Handling missing values and duplicates.
Exploratory data analysis (EDA) including visualizations like bar plots and word clouds.
Text preprocessing tasks such as lowercasing, tokenization, punctuation removal, stop word removal, stemming, lemmatization, and handling special characters.

# 2.Feature Engineering
Transforming text data into numerical vectors using TF-IDF vectorization.
Building the Machine Learning Model

# 3.Splitting the dataset into training and testing sets.
Training a Multinomial Naive Bayes classifier using TF-IDF vectors.
Evaluating the model using accuracy score and classification report.

# 4.Deployment and Prediction
Preprocessing new reviews using the same preprocessing steps applied to the training data.
Using the trained model to predict sentiment (positive or negative) of new reviews.

# Files in the Repository
Reviews.csv: Dataset containing restaurant reviews.
Restaurant_Review_Sentiment_Analysis.ipynb: Jupyter notebook containing the complete code for data preprocessing, model building, and evaluation.
README.md: This file, providing an overview of the project, instructions, and details about each step.

# Libraries Used
pandas: Data manipulation and analysis.
matplotlib and seaborn: Data visualization.
nltk: Natural language processing tasks such as tokenization, stopwords removal, stemming, and lemmatization.
scikit-learn: Machine learning library for building the classification model.
beautifulsoup4: For cleaning HTML tags from text data.
contractions and emoji: For expanding contractions and handling emojis in text data.

# Conclusion
This project demonstrates the process of sentiment analysis on restaurant reviews using machine learning and NLP techniques. It provides a foundation for building more sophisticated sentiment analysis models or integrating sentiment analysis into other applications.
