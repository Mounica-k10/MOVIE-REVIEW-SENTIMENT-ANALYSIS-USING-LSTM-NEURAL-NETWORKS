# MOVIE REVIEW SENTIMENT ANALYSIS USING LSTM NEURAL NETWORKS

## Deep Learning Project

## 1. Project Overview

Sentiment analysis is a Natural Language Processing (NLP) technique used to identify opinions and emotions expressed in textual data.

**Movie Review Sentiment Analysis Using LSTM Neural Networks** is a deep learning project that classifies movie reviews as either **Positive or Negative** using a Long Short-Term Memory (LSTM) neural network.

The project uses the IMDb movie review dataset available through TensorFlow/Keras. The model processes reviews as numerical sequences and learns patterns in the text to predict sentiment.

The trained model also accepts custom movie reviews and returns the predicted sentiment along with a positive sentiment score.

## 2. Objectives

- Develop a movie review sentiment classification system using deep learning.
- Use the IMDb dataset for training and evaluation.
- Convert textual reviews into numerical sequences.
- Build and train an LSTM neural network.
- Evaluate the model using classification metrics.
- Predict the sentiment of user-provided movie reviews.

## 3. Features

- **Text Processing:** Converts movie reviews into numerical sequences.
- **Word Embedding:** Represents words as dense numerical vectors.
- **LSTM Network:** Learns sequential patterns in movie reviews.
- **Binary Classification:** Classifies reviews as positive or negative.
- **Model Evaluation:** Calculates accuracy, precision, recall, and F1-score.
- **Confusion Matrix:** Visualizes correct and incorrect predictions.
- **Performance Visualization:** Displays training and validation accuracy.
- **Custom Prediction:** Predicts sentiment for user-provided reviews.
- **Sentiment Score:** Displays the estimated positive sentiment probability.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Google Colab | Development environment |
| TensorFlow | Deep learning framework |
| Keras | LSTM model implementation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Scikit-learn | Model evaluation |

## 5. Dataset

**Dataset Name:** IMDb Movie Review Dataset

**Source:** TensorFlow/Keras

**Dataset Loading:** `tensorflow.keras.datasets.imdb`

The dataset contains 50,000 movie reviews labeled as positive or negative.

### Dataset Details

| Dataset | Number of Reviews |
|---|---:|
| Training Set | 25,000 |
| Testing Set | 25,000 |
| Total | 50,000 |

### Data Preparation

- Vocabulary size: 10,000 words.
- Maximum sequence length: 200 tokens.
- Reviews are represented as numerical sequences.
- Padding and truncation are applied to maintain a fixed input length.

### Class Labels

| Label | Sentiment |
|---|---|
| 0 | Negative |
| 1 | Positive |

## 6. Methodology

The project follows these steps:

1. **Dataset Loading:** Load the IMDb dataset using TensorFlow/Keras.
2. **Text Representation:** Convert reviews into numerical sequences.
3. **Sequence Preparation:** Pad or truncate sequences to 200 tokens.
4. **Dataset Splitting:** Use the predefined training and testing datasets.
5. **Model Construction:** Build an LSTM neural network.
6. **Model Training:** Train the model using the Adam optimizer.
7. **Validation:** Use 20% of the training data for validation.
8. **Model Evaluation:** Evaluate the model on the test dataset.
9. **Performance Visualization:** Generate evaluation graphs and a confusion matrix.
10. **Sentiment Prediction:** Predict the sentiment of custom movie reviews.

### Dataset Splitting

| Dataset | Number of Reviews | Purpose |
|---|---:|---|
| Training Set | 25,000 | Train the model |
| Validation Set | 5,000 | Monitor training performance |
| Testing Set | 25,000 | Evaluate the trained model |

The validation set is taken from the training data using a 20% validation split.

## 7. LSTM Model Architecture

The model consists of the following layers:

| Layer | Configuration | Purpose |
|---|---|---|
| Embedding Layer | 64-dimensional vectors | Converts word indices into dense representations |
| LSTM Layer | 64 units | Learns sequential patterns in reviews |
| Dense Layer | 32 neurons, ReLU activation | Learns additional features |
| Output Layer | 1 neuron, Sigmoid activation | Produces the positive sentiment probability |

### Model Configuration

- **Optimizer:** Adam
- **Loss Function:** Binary Cross-Entropy
- **Epochs:** 5
- **Batch Size:** 64
- **Validation Split:** 20%
- **Vocabulary Size:** 10,000 words
- **Maximum Sequence Length:** 200 tokens
- **Classification Threshold:** 0.5

The sigmoid output produces a value between 0 and 1. A value greater than or equal to 0.5 is classified as positive, while a value below 0.5 is classified as negative.

## 8. Model Evaluation Results

The model achieved the following results on the test dataset.

| Evaluation Metric | Negative | Positive |
|---|---:|---:|
| Precision | 73.82% | 91.07% |
| Recall | 93.44% | 66.86% |
| F1-Score | 82.48% | 77.11% |

**Overall Test Accuracy: 80.15%**

### Result Analysis

The model achieved an accuracy of 80.15% on 25,000 test reviews.

The negative class achieved a recall of 93.44%, indicating that the model correctly identified a high proportion of actual negative reviews.

The positive class achieved a precision of 91.07%, indicating that most reviews predicted as positive were actually positive.

However, the positive class recall was 66.86%, showing that some positive reviews were incorrectly classified as negative.

These results demonstrate the model's ability to classify movie reviews, although further improvements could help balance performance across both classes.

## 9. Confusion Matrix

The confusion matrix compares the actual sentiment labels with the predicted labels.

It helps identify correctly classified reviews and the errors made by the model.

**Add your actual confusion matrix image here after uploading it to the repository.**

## 10. Movie Review Prediction

The project includes a prediction function that accepts a movie review and returns the predicted sentiment.

### Example Code

```python
review = input("Enter your movie review: ")
predict_sentiment(review)
```

### Prediction Output

The model displays:

- **Input Review:** The review entered by the user.
- **Sentiment:** Positive or Negative.
- **Positive Sentiment Score:** The estimated probability of positive sentiment, expressed as a percentage.

The actual prediction depends on the review entered and the trained model.

## 11. Visualizations

The project uses the following visualizations to analyze model performance:

- Training and validation accuracy graph.
- Training and validation loss graph, if generated.
- Confusion matrix.
- Classification report.
- Custom movie review prediction examples.

These visualizations help evaluate the model and understand its classification performance.

## 12. How to Run the Project

The project can be executed using Google Colab.

### Step 1: Open the Notebook

Open `Movie_Review_Sentiment_Analysis.ipynb` in Google Colab.

### Step 2: Load the Dataset

Load the IMDb movie review dataset using TensorFlow/Keras.

### Step 3: Preprocess the Reviews

Convert the reviews into numerical sequences and pad or truncate them to 200 tokens.

### Step 4: Build and Train the Model

Run the notebook cells to construct and train the LSTM model.

### Step 5: Evaluate the Model

Generate the test accuracy, classification report, and confusion matrix.

### Step 6: Predict Sentiment

Run the prediction cell and enter a movie review to obtain the predicted sentiment.

**Note:** Execute the notebook cells in order to ensure the model and required variables are initialized correctly.

## 13. Advantages

- Uses deep learning for sentiment classification.
- Learns sequential patterns in textual data.
- Supports positive and negative sentiment classification.
- Provides multiple evaluation metrics.
- Accepts custom movie reviews for prediction.
- Can be executed using Google Colab.

## 14. Limitations

- The model may incorrectly classify some reviews.
- Performance depends on the quality and diversity of the dataset.
- The vocabulary is limited to the 10,000 most frequent words.
- Reviews are padded or truncated to 200 tokens.
- Sarcasm and complex expressions may be difficult to interpret.
- The current model does not explicitly classify mixed sentiment.
- Performance may vary on reviews from other sources.

## 15. Future Enhancements

- Improve performance through hyperparameter tuning.
- Experiment with bidirectional LSTM networks.
- Compare LSTM with GRU and other deep learning models.
- Explore advanced text representation techniques.
- Develop a web application for interactive predictions.
- Investigate methods for handling sarcasm and mixed opinions.
- Evaluate the model on additional movie review datasets.

These are proposed enhancements and are not claimed as implemented features.

## 16. References

1. [IMDb Dataset — TensorFlow Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb)
2. [TensorFlow Documentation](https://www.tensorflow.org/)
3. [Keras Documentation](https://keras.io/)
4. [Scikit-learn Documentation](https://scikit-learn.org/)

## 17. Conclusion

This project demonstrates the application of Long Short-Term Memory neural networks to movie review sentiment analysis.

Using the IMDb dataset, the model processes numerical review sequences and learns patterns to classify reviews as positive or negative.

The model achieved a test accuracy of 80.15% and was evaluated using precision, recall, F1-score, and a confusion matrix.

The project provides practical experience with text preprocessing, deep learning model development, model evaluation, and sentiment prediction.
