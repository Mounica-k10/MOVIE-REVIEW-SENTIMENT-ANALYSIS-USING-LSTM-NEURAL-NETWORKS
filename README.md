MOVIE-REVIEW-SENTIMENT-ANALYSIS-USING-LSTM-NEURAL-NETWORKS
DEEP LEARNING PROJECT: MOVIE REVIEW SENTIMENT ANALYSIS USING LSTM NEURAL NETWORKS
Project Overview
Sentiment analysis is a Natural Language Processing (NLP) technique used to identify the opinion or emotion expressed in textual data. In movie reviews, sentiment analysis helps determine whether a viewer has expressed a positive or negative opinion about a movie.
Movie Review Sentiment Analysis Using LSTM Neural Networks is a deep learning project that classifies movie reviews as either Positive or Negative using a Long Short-Term Memory (LSTM) neural network.
The project uses the IMDb Movie Review Dataset available through TensorFlow/Keras. The model processes movie reviews as numerical sequences and learns patterns in the text to predict sentiment.
The trained model can also accept a custom movie review and return its predicted sentiment along with a positive sentiment score.
Objectives
•	Develop a movie review sentiment classification system using deep learning.
•	Use the IMDb movie review dataset for model training and evaluation.
•	Convert textual reviews into numerical sequences.
•	Apply an LSTM neural network to learn sequential patterns in text.
•	Evaluate the model using accuracy, precision, recall, and F1-score.
•	Generate a confusion matrix and training performance graphs.
•	Predict the sentiment of user-provided movie reviews.
Features
•	Text Processing: Converts movie reviews into numerical sequences.
•	Word Embedding: Represents words using dense numerical vectors.
•	LSTM Network: Learns sequential patterns and contextual information from reviews.
•	Binary Classification: Classifies reviews as positive or negative.
•	Model Evaluation: Calculates accuracy, precision, recall, and F1-score.
•	Confusion Matrix: Visualizes correct and incorrect predictions.
•	Training Visualization: Displays training and validation accuracy.
•	Custom Review Prediction: Accepts a movie review and predicts its sentiment.
•	Sentiment Score: Displays the predicted probability of positive sentiment.
Technologies Used
Technology	Purpose
Python	Programming language
Google Colab	Development and execution environment
TensorFlow	Deep learning framework
Keras	LSTM model implementation
NumPy	Numerical operations
Matplotlib	Visualization of training results
Scikit-learn	Classification metrics and model evaluation
Dataset
Dataset Name: IMDb Movie Review Dataset
Source: TensorFlow/Keras
Dataset Loading: tensorflow.keras.datasets.imdb
The IMDb dataset contains 50,000 movie reviews labeled as positive or negative. It is divided into training and testing datasets.
Dataset Details
Dataset	Number of Reviews
Training Set	25,000
Testing Set	25,000
Total	50,000
Data Preparation
The dataset is configured to use the 10,000 most frequent words. The reviews are represented as numerical sequences and padded or truncated to a fixed length of 200 tokens.
This ensures that all reviews have a consistent input length for the LSTM model.
Class Labels
Label	Class
0	Negative
1	Positive
Methodology
The project follows these steps:
1.	Dataset Loading: Load the IMDb movie review dataset using TensorFlow/Keras.
2.	Text Representation: Represent reviews as numerical word-index sequences.
3.	Sequence Preparation: Pad or truncate reviews to a maximum length of 200 tokens.
4.	Dataset Splitting: Use the predefined training and testing sets.
5.	Model Construction: Build an LSTM-based neural network using Keras.
6.	Model Training: Train the network using the Adam optimizer and binary cross-entropy loss.
7.	Validation: Use 20% of the training data for validation.
8.	Model Evaluation: Evaluate performance using test accuracy and classification metrics.
9.	Performance Visualization: Generate training graphs and a confusion matrix.
10.	Sentiment Prediction: Accept a custom review and predict its sentiment.
Dataset Splitting
Dataset	Number of Reviews	Purpose
Training Set	25,000	Train the model
Validation Set	5,000	Monitor training performance
Testing Set	25,000	Evaluate the model on unseen reviews
The validation set is taken from the training data using a 20% validation split during model training.
LSTM Model Architecture
The model uses the following architecture:
Layer	Configuration	Purpose
Input	Numerical review sequence	Accepts the encoded review
Embedding Layer	64-dimensional vectors	Converts word indices into dense vector representations
LSTM Layer	64 units	Learns sequential patterns in the review
Dense Layer	32 neurons, ReLU activation	Learns additional features
Output Layer	1 neuron, Sigmoid activation	Produces the positive sentiment probability
Model Configuration
•	Optimizer: Adam
•	Loss Function: Binary Cross-Entropy
•	Epochs: 5
•	Batch Size: 64
•	Validation Split: 20%
•	Vocabulary Size: 10,000 words
•	Maximum Sequence Length: 200 tokens
•	Output Activation: Sigmoid
•	Classification Threshold: 0.5
The sigmoid output produces a value between 0 and 1. A value greater than or equal to 0.5 is classified as positive, while a value below 0.5 is classified as negative.
Model Evaluation Results
The trained model achieved the following results on the test dataset.
Evaluation Metric	Result
Test Accuracy	80.15%
Positive Class Precision	91.07%
Positive Class Recall	66.86%
Positive Class F1-Score	77.11%
Negative Class Precision	73.82%
Negative Class Recall	93.44%
Negative Class F1-Score	82.48%
Result Analysis
The model achieved an overall test accuracy of 80.15% on 25,000 movie reviews.
The negative class achieved a recall of 93.44%, indicating that the model correctly identified a high proportion of actual negative reviews.
The positive class achieved a precision of 91.07%, indicating that a high proportion of reviews predicted as positive were actually positive.
However, the positive class recall was 66.86%, showing that some positive reviews were incorrectly classified as negative.
These results indicate that the model can identify sentiment in movie reviews, although further improvements could help balance its performance across both classes.
Confusion Matrix Results
The confusion matrix provides a comparison between actual sentiment labels and predicted sentiment labels.
It helps identify correctly classified reviews and errors made by the model.
Note: Insert your actual confusion matrix image here after uploading it to the repository.
Movie Review Prediction
The project includes a prediction function that accepts a movie review and returns the predicted sentiment.
Example
review = input("Enter your movie review: ")
predict_sentiment(review)
Prediction Output
The model returns:
•	Input Review: The movie review entered by the user.
•	Sentiment: Positive or Negative.
•	Positive Sentiment Score: The model's estimated probability of positive sentiment, expressed as a percentage.
For example, a review praising the acting, story, and visuals may receive a positive prediction, while a review describing the movie as boring or disappointing may receive a negative prediction.
The actual prediction and score depend on the trained model.
Visualizations
The project uses visualizations to analyze model performance, including:
•	Training and validation accuracy graph.
•	Training and validation loss graph, if generated.
•	Confusion matrix.
•	Classification report.
•	Custom movie review prediction outputs.
These visualizations help evaluate the model and understand its classification performance.
How to Run the Project
The project can be executed using Google Colab.
Step 1: Open the Notebook
Open Movie_Review_Sentiment_Analysis.ipynb in Google Colab.
Step 2: Import Required Libraries
Ensure that the required libraries are available.
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import classification_report, confusion_matrix
Step 3: Load the Dataset
Load the IMDb dataset using TensorFlow/Keras and configure the vocabulary size.
Step 4: Preprocess the Reviews
Prepare the numerical sequences and pad or truncate them to 200 tokens.
Step 5: Build and Train the Model
Run the model construction and training cells in the notebook.
Step 6: Evaluate the Model
Generate the test accuracy, classification report, and confusion matrix.
Step 7: Predict Sentiment
Execute the prediction cell and enter a custom movie review.
The model displays the predicted sentiment and positive sentiment score.
Note: Run the notebook cells in the correct order. The import statements above are illustrative; use the complete code already present in the project notebook.
Advantages
•	Uses deep learning for text classification.
•	Learns sequential patterns in movie reviews.
•	Processes numerical word sequences using an embedding layer.
•	Supports positive and negative sentiment classification.
•	Provides several evaluation metrics.
•	Accepts custom movie reviews for prediction.
•	Can be developed and executed in Google Colab.
Limitations
•	The model may incorrectly classify some reviews.
•	Performance depends on the quality and diversity of the dataset.
•	The vocabulary is limited to the 10,000 most frequent words.
•	Reviews are padded or truncated to 200 tokens.
•	Sarcasm and complex expressions may be difficult to interpret.
•	The model performs binary classification and does not explicitly identify mixed sentiment.
•	The reported performance may differ on reviews from other datasets or real-world sources.
Future Enhancements
Potential improvements include:
•	Improving performance through hyperparameter tuning.
•	Experimenting with bidirectional LSTM networks.
•	Comparing LSTM with GRU and other deep learning architectures.
•	Exploring advanced text representation techniques.
•	Developing a web application for interactive sentiment prediction.
•	Improving the handling of sarcasm and reviews containing both positive and negative opinions.
•	Evaluating the model on additional movie review datasets.
These are potential enhancements and should not be considered implemented unless they are added and tested.
References
1.	IMDb Movie Review Dataset — TensorFlow/Keras documentation: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb
2.	TensorFlow Documentation: https://www.tensorflow.org/
3.	Keras Documentation: https://keras.io/
4.	Scikit-learn Documentation: https://scikit-learn.org/
Conclusion
This project demonstrates the application of Long Short-Term Memory neural networks to movie review sentiment analysis.
Using the IMDb dataset, the model processes numerical review sequences and learns patterns to classify movie reviews as positive or negative.
The model achieved a test accuracy of 80.15% and was evaluated using precision, recall, F1-score, and a confusion matrix.
The project provides practical experience with text preprocessing, deep learning model development, model evaluation, and sentiment prediction.

