# README: Comparing Naive Bayes Models

## Overview

This repository provides an in-depth comparison of different Naive Bayes models, including **Gaussian Naive Bayes**, **Multinomial Naive Bayes**, and **Bernoulli Naive Bayes**. We aim to analyze their performance across various scenarios, including different datasets, feature distributions, and classification tasks. The repository includes sample code, explanations, and comparison tables to help understand which Naive Bayes model is best suited for a given problem.

## Table of Contents

1. [Introduction](#introduction)
2. [Naive Bayes Models Overview](#naive-bayes-models-overview)
   - Gaussian Naive Bayes
   - Multinomial Naive Bayes
   - Bernoulli Naive Bayes
3. [Dataset Preparation](#dataset-preparation)
4. [Model Implementation](#model-implementation)
5. [Performance Evaluation](#performance-evaluation)
   - Accuracy
   - Precision, Recall, F1-Score
   - Confusion Matrix
6. [Comparison and Conclusions](#comparison-and-conclusions)
7. [Sample Code](#sample-code)
8. [References](#references)

## Introduction

Naive Bayes is a family of simple yet effective probabilistic classifiers based on Bayes' theorem. These models assume strong (naive) independence between features, which simplifies the computation of probabilities. Despite their simplicity, Naive Bayes models often perform remarkably well, particularly for text classification problems.

This project explores three popular variants of Naive Bayes models:

- **Gaussian Naive Bayes**: Assumes that the features follow a normal distribution.
- **Multinomial Naive Bayes**: Suitable for discrete data, commonly used in text classification with word counts or term frequencies.
- **Bernoulli Naive Bayes**: Designed for binary/boolean features, often used for text classification with binary word occurrence vectors.

## Naive Bayes Models Overview

### Gaussian Naive Bayes

Gaussian Naive Bayes assumes that the features are normally distributed. It is typically used when dealing with continuous data. The probability distribution of each feature is modeled using a Gaussian (normal) distribution.

### Multinomial Naive Bayes

Multinomial Naive Bayes is appropriate for discrete data, particularly when dealing with text data. It models the distribution of each feature as a multinomial distribution, making it ideal for document classification tasks based on word counts or term frequencies.

### Bernoulli Naive Bayes

Bernoulli Naive Bayes is suitable for binary features. It assumes that the features are binary (i.e., either 0 or 1) and is also widely used in text classification, especially when the presence or absence of words (rather than their frequency) is important.

## Dataset Preparation

To effectively compare the different Naive Bayes models, we use multiple datasets with varying characteristics:

1. **Iris Dataset** (for Gaussian Naive Bayes): A classic dataset for classification problems involving continuous features.
2. **20 Newsgroups Dataset** (for Multinomial and Bernoulli Naive Bayes): A popular dataset for text classification tasks.
3. **Custom Binary Feature Dataset**: A synthetic dataset with binary features to test Bernoulli Naive Bayes.

### Data Preprocessing

Each dataset undergoes appropriate preprocessing steps:
- For the **Iris Dataset**, features are scaled to fit the Gaussian assumption.
- For the **20 Newsgroups Dataset**, text is vectorized using `CountVectorizer` and `TfidfVectorizer`.
- For the **Binary Feature Dataset**, data is binarized to create binary feature vectors.

## Model Implementation

The Naive Bayes models are implemented using Python and the `scikit-learn` library. Below are the steps for model implementation:

1. **Gaussian Naive Bayes**:
   ```python
   from sklearn.naive_bayes import GaussianNB
   model = GaussianNB()
   model.fit(X_train, y_train)
   predictions = model.predict(X_test)
   ```

2. **Multinomial Naive Bayes**:
   ```python
   from sklearn.naive_bayes import MultinomialNB
   model = MultinomialNB()
   model.fit(X_train, y_train)
   predictions = model.predict(X_test)
   ```

3. **Bernoulli Naive Bayes**:
   ```python
   from sklearn.naive_bayes import BernoulliNB
   model = BernoulliNB()
   model.fit(X_train, y_train)
   predictions = model.predict(X_test)
   ```

## Performance Evaluation

Each model's performance is evaluated using the following metrics:

- **Accuracy**: The proportion of correctly classified instances.
- **Precision, Recall, F1-Score**: Metrics for evaluating the balance between precision and recall.
- **Confusion Matrix**: A summary of prediction results on the classification problem.

### Example Performance Metrics Table

| Model               | Dataset            | Accuracy | Precision | Recall | F1-Score |
|---------------------|--------------------|----------|-----------|--------|----------|
| Gaussian Naive Bayes| Iris               | 0.95     | 0.96      | 0.95   | 0.95     |
| Multinomial Naive Bayes | 20 Newsgroups | 0.83     | 0.84      | 0.83   | 0.83     |
| Bernoulli Naive Bayes | Binary Feature  | 0.88     | 0.89      | 0.88   | 0.88     |

## Comparison and Conclusions

### Strengths and Weaknesses

1. **Gaussian Naive Bayes**:
   - Strengths: Effective with continuous data, simple to implement.
   - Weaknesses: Assumes normal distribution, may perform poorly if the assumption is violated.

2. **Multinomial Naive Bayes**:
   - Strengths: Excellent for text classification with discrete features.
   - Weaknesses: Not suitable for continuous data.

3. **Bernoulli Naive Bayes**:
   - Strengths: Works well with binary/boolean features.
   - Weaknesses: May not perform well with non-binary features or when feature frequency matters.

### Conclusion

- **Gaussian Naive Bayes** is best suited for problems where features are continuous and approximately normally distributed.
- **Multinomial Naive Bayes** excels in text classification tasks with discrete word counts or frequencies.
- **Bernoulli Naive Bayes** is optimal for binary feature sets, particularly when the presence or absence of features is critical.

## Sample Code

Here’s a brief example of how to implement the different Naive Bayes models in Python using `scikit-learn`:

```python
from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report

# Example: Gaussian Naive Bayes on Iris Dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
predictions = gnb.predict(X_test)
print("Gaussian NB Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# Example: Multinomial Naive Bayes on 20 Newsgroups Dataset
newsgroups = fetch_20newsgroups(subset='train')
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(newsgroups.data)
X_train, X_test, y_train, y_test = train_test_split(X_train_counts, newsgroups.target, test_size=0.3, random_state=42)
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
predictions = mnb.predict(X_test)
print("Multinomial NB Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# Example: Bernoulli Naive Bayes on Binary Feature Dataset
# Assume X_train_binary, X_test_binary are preprocessed binary datasets
bnb = BernoulliNB()
bnb.fit(X_train_binary, y_train_binary)
predictions = bnb.predict(X_test_binary)
print("Bernoulli NB Accuracy:", accuracy_score(y_test_binary, predictions))
print(classification_report(y_test_binary, predictions))
```

## References

1. Scikit-learn Documentation: [https://scikit-learn.org/stable/modules/naive_bayes.html](https://scikit-learn.org/stable/modules/naive_bayes.html)
2. Iris Dataset: UCI Machine Learning Repository
3. 20 Newsgroups Dataset: Scikit-learn Datasets

---

This README file is intended to serve as a comprehensive guide for understanding and comparing different Naive Bayes models. By following the explanations and sample code provided, users can experiment with these models and understand their applicability to various classification tasks.
