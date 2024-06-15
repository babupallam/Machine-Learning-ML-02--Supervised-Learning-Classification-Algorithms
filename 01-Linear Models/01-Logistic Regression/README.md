 # Logistic Regression
## Understanding Logistic Regression

- **Purpose**: Used for binary classification problems (e.g., predicting whether an email is spam or not).
- **Output**: Probability that a given input belongs to a particular class, typically between 0 and 1.
- **Decision Rule**: Classify the input as class 1 if the probability is greater than 0.5, otherwise class 0.

## Mathematical Formulation

### Logit Function

The logit function is a linear combination of input features. It represents the relationship between the features and the log-odds of the dependent binary variable. The logit function is defined as:

\[
\text{logit}(P) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n
\]

Where:
- \( \text{logit}(P) \) is the log-odds of the probability \( P \) of the binary outcome.
- \( \beta_0 \) is the intercept.
- \( \beta_1, \beta_2, \ldots, \beta_n \) are the coefficients of the input features.
- \( x_1, x_2, \ldots, x_n \) are the input features.

### Sigmoid Function

The sigmoid function converts the logit (linear combination of input features) to a probability value between 0 and 1. The probability \( P(y=1|X) \) of the dependent binary variable \( y \) being 1 given the input features \( X \) is defined as:

\[
P(y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n)}}
\]

Where:
- \( P(y=1|X) \) is the probability that the outcome is 1 given the input features \( X \).
- \( e \) is the base of the natural logarithm.
- \( \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n \) is the logit, the linear combination of the input features.

## Summary

The logistic regression model uses the logit function to create a linear combination of input features and the sigmoid function to convert this linear combination into a probability. This probability is then used to classify the input data into one of the two binary classes.

By understanding and applying these mathematical formulations, you can effectively use logistic regression for binary classification tasks.



==========================================
# Detailing Mathematical Formulation Concepts
# Logistic Regression README

## 1. The Logit Function

The core idea of logistic regression is to model the probability that a given input \( \mathbf{X} \) belongs to a particular class \( y = 1 \).

The logit function (log-odds) is a linear combination of the input features:
\[ \text{logit}(P) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n \]

where:
- \( \beta_0 \) is the intercept term.
- \( \beta_i \) are the coefficients for the features.
- \( x_i \) are the input features.

## 2. The Sigmoid Function

To convert the logit to a probability, we use the sigmoid function, also known as the logistic function:
\[ \sigma(z) = \frac{1}{1 + e^{-z}} \]

where \( z \) is the logit. This function outputs values between 0 and 1, which can be interpreted as probabilities.

Applying the sigmoid function to the logit, we get the probability \( P \) that the output \( y = 1 \):
\[ P(y=1|\mathbf{X}) = \sigma(\text{logit}(P)) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n)}} \]

## 3. Decision Boundary

To make a classification decision, we set a threshold, commonly 0.5:
\[ \hat{y} = 
\begin{cases} 
1 & \text{if } P(y=1|\mathbf{X}) \geq 0.5 \\
0 & \text{if } P(y=1|\mathbf{X}) < 0.5 
\end{cases}
\]

## 4. The Loss Function

Logistic regression is typically trained using maximum likelihood estimation. The loss function used is the logistic loss or binary cross-entropy loss:

For a single training example:
\[ L(\beta; \mathbf{x}, y) = - \left[ y \log(P(y=1|\mathbf{x})) + (1 - y) \log(1 - P(y=1|\mathbf{x})) \right] \]

where \( y \) is the actual label.

For a dataset with \( m \) training examples, the total loss is:
\[ L(\beta) = - \frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(P(y=1|\mathbf{x}^{(i)})) + (1 - y^{(i)}) \log(1 - P(y=1|\mathbf{x}^{(i)})) \right] \]

## 5. Gradient Descent

To minimize the loss function, we use gradient descent. The update rule for the parameters \( \beta \) is:
\[ \beta_j := \beta_j - \alpha \frac{\partial L}{\partial \beta_j} \]

where \( \alpha \) is the learning rate.

The gradient of the loss function with respect to the parameters \( \beta \) is:
\[ \frac{\partial L}{\partial \beta_j} = \frac{1}{m} \sum_{i=1}^{m} \left( P(y=1|\mathbf{x}^{(i)}) - y^{(i)} \right) x_j^{(i)} \]

## 6. Regularization

To prevent overfitting, regularization can be added to the loss function. Common types are L1 (Lasso) and L2 (Ridge) regularization.

**L2 Regularization (Ridge)**:
\[ L(\beta) = - \frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(P(y=1|\mathbf{x}^{(i)})) + (1 - y^{(i)}) \log(1 - P(y=1|\mathbf{x}^{(i)})) \right] + \frac{\lambda}{2m} \sum_{j=1}^{n} \beta_j^2 \]

**L1 Regularization (Lasso)**:
\[ L(\beta) = - \frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(P(y=1|\mathbf{x}^{(i)})) + (1 - y^{(i)}) \log(1 - P(y=1|\mathbf{x}^{(i)})) \right] + \frac{\lambda}{m} \sum_{j=1}^{n} |\beta_j| \]
