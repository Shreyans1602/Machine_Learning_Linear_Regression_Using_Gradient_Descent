# Linear_Regression_Using_Gradient_Descent

# Instructions
- There are two parts to this assignment. The first part requires you to write code that uses gradient descent for linear regression. In the second part, you will use a ML library on the same dataset and compare your results.
- For the programming part, it’s your responsibility to find the best set of parameters. Please include a README file detailing how to compile and run your program.

# 1 Linear Regression using Gradient Descent
## Coding in Python
For this part, you will write your own code in Python for implementing the gradient descent algorithm, and apply it to a linear regression problem. You are free to use any data loading, pre-processing, parsing, and graphing library, such as numpy, pandas, graphics. However, you cannot use any library that implements gradient descent or linear regression.

You will need to perform the following:
1. Choose a dataset suitable for regression from UCI ML Repository: https://archive.ics.uci.edu/ml/datasets.php. If the above link doesn’t work, you can go to the main page: https://archive.ics.uci.edu/ml/index.php and choose “view all datasets option”. Host the dataset on a public location e.g. GitHub. Please do not hard code paths to your local computer.

2. Pre-process your dataset. Pre-processing includes the following activities:
- Remove null or NA values
- Remove any redundant rows
- Convert categorical variables to numerical variables
- If you feel an attribute is not suitable or is not correlated with the outcome, you might want to get rid of it.
- Any other pre-processing that you may need to perform.

3. After pre-processing split the dataset into training and test parts. It is up to you to choose the train/test ratio, but commonly used values are 80/20, 90/10, etc.

4. Use the training dataset to construct a linear regression model. We discussed creating a linear regression model using a single attribute in class. For this assignment, you will need to extend that model to consider multiple attributes. You may want to think of the vector form of the weight update equation. Note again: you cannot use a library that implements gradient descent or linear regression.
There are various parameters such as learning rate, number of iterations or other stopping condition, etc. You need to tune these parameters to achieve the optimum error value. Tuning involves testing various combi- nations, and then using the best one. You need to create a log file that indicates parameters used and error (MSE) value obtained for various trials.

5. Apply the model you created in the previous step to the test part of the dataset. Report the test dataset error values for the best set of parameters obtained from previous part. If you are not satisfied with your answer, you can repeat the training step.

6. Answer this question: Are you satisfied that you have found the best solution? Explain.
