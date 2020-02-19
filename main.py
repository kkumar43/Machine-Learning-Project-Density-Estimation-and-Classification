import numpy as np
import scipy.io

# Importing the Dataset for the Project
dataset = scipy.io.loadmat("C:\\Users\\kunal\\Desktop\\Statistical Machine Learning\\Project 1\\mnist_data.mat")

# Extracting the 4 individual Matrix from Dataset
training_set = dataset["trX"]
training_set_label = dataset["trY"]
testing_set = dataset["tsX"]
testing_set_label = dataset["tsY"]

# splitting the train dataset into Class - 7 and Class - 8
training_set_7 = training_set[0:6265]
training_set_8 = training_set[6265:]

# Calculating the Mean and Standard Deviation of each image belonging to class - 7 and class - 8
mean_7 = np.mean(training_set_7, axis=1)
mean_8 = np.mean(training_set_8, axis=1)
sd_7 = np.std(training_set_7, axis=1)
sd_8 = np.std(training_set_8, axis=1)

# Mean and Standard Daviation 'array' for the training Dataset and testing Dataset
training_set_mean = np.mean(training_set, axis=1)
testing_set_mean = np.mean(testing_set, axis=1)
training_set_sd = np.std(training_set, axis=1)
testing_set_sd = np.std(testing_set, axis=1)


# "" NAIVE BAYES CLASSIFIER ""

# P(Y=Class/X)-- Posterior Probability of the Class
# P(X/Y=Class)-- Likelihood
# P(Y=Class)--   Class Prior Probability
# P(X)--         Predictor Prior Probability


# calculating the probability that an image belongs to class 7 and class 8 respectively
prior_prob_7 = mean_7.size / training_set_mean.size
prior_prob_8 = mean_8.size / training_set_mean.size


# Declaring the Univariate Normal Distribution (PDF) [P(y/x)] where y represents the label and x represent the features

def p_x_given_y(x, mean, variance):
    p_x_give_y = (1 / (np.sqrt(2 * np.pi * variance))) * np.exp(-(x - mean) ** 2 / (2 * variance))
    return p_x_give_y


# Calculating the 'numerator' for posterior probability for 7

post_prob_7 = prior_prob_7 \
              * p_x_given_y(testing_set_mean, mean_7.mean(), mean_7.var()) \
              * p_x_given_y(testing_set_sd, sd_7.mean(), sd_7.var())

# Calculating the 'numerator' for posterior probability for 8
post_prob_8 = prior_prob_8 \
              * p_x_given_y(testing_set_mean, mean_8.mean(), mean_8.var()) \
              * p_x_given_y(testing_set_sd, sd_8.mean(), sd_8.var())

# Comparing the Values of Posterior Prob of 7 and Posterior Prob of 8
# If post_prob_7 > post_prob_8 = Image belong to 7
# If post_prob_7 < post_prob_8 = Image belong to 8

value_compare = np.greater(post_prob_8, post_prob_7)

# Converting the True Values to 1 & False Values to 0
value_compare_numeric = value_compare.astype(np.int)

# calculating the accuracy for 7
accuracy_7 = ((np.count_nonzero(np.equal(value_compare_numeric[0:1028], np.squeeze(testing_set_label)[0:1028]))
              / np.squeeze(testing_set_label)[0:1028].size)
              * 100)
print('The Accuracy of the Naive Bayes for predicting "7" is ', accuracy_7, "%")

# calculating the accuracy for class - 8
accuracy_8 = ((np.count_nonzero(np.equal(value_compare_numeric[1028:], np.squeeze(testing_set_label)[1028:]))
              / np.squeeze(testing_set_label)[1028:].size)
              * 100)
print('The Accuracy of the Naive Bayes for predicting "8" is ', accuracy_8, "%")

# Comparing and finding tot number of correctly matched and unmatched using testing dataset label

value_compare_numeric_and_label = np.equal(value_compare_numeric, testing_set_label)

#Total number of correct prediction
tot_correct_prediction = np.count_nonzero(value_compare_numeric_and_label)

# The overall accuracy (Total Number of Images Predicted Correctly of the Test Dataset/Total Number of Images in the Test Dataset)

accuracy_naive_bayes = ((tot_correct_prediction / testing_set_label.size) * 100)
print("The Accuracy of the Naive Bayes is ", accuracy_naive_bayes, "%")


#   "" LOGISTIC REGRESSION CLASSIFIER ""

# Defining function for Logistic Regression

def log_reg(X, y, num_iterations, learning_rate, add_intercept=False):
    if add_intercept:
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X))

    theta = np.zeros(X.shape[1])

    for iter in range(num_iterations):
        temp = np.dot(X, theta)
        prds = sig(temp)

        diff = y - prds

        gradient = np.dot(X.T, diff)
        theta = (theta + (learning_rate * gradient))

    return theta

def sig(temp):
    return 1 / (1 + np.exp(-temp))

# Stacking from Training and Testing
train_feature = np.column_stack((training_set_mean, training_set_sd))
train_label = np.squeeze(training_set_label.transpose())
test_feature = np.column_stack((testing_set_mean, testing_set_sd))
test_label = np.squeeze(testing_set_label.transpose())

# Calculating Theta for L.R
thetas = log_reg(train_feature, train_label, num_iterations=150000, learning_rate=0.001,
                             add_intercept=True)

final_val = np.dot(np.hstack((np.ones((test_feature.shape[0], 1)),
                            test_feature)), thetas)
prds = np.round(sig(final_val))

# Calculating accuracy for 7
accuracy_log_reg_7 = (prds[0:1028] == test_label[0:1028]).sum().astype(int) / len(test_label[0:1028])
print('The accuracy of the Logistic Regression for predicting "7" is ', accuracy_log_reg_7 * 100, '%')

#Calculating accuracy for 8
accuracy_log_reg_8 = (prds[1028:] == test_label[1028:]).sum().astype(int) / len(test_label[1028:])
print('The accuracy of the Logistic Regression for predicting "8" is ', accuracy_log_reg_8 * 100, '%')

# Calculating accuracy for L.R
accuracy_log_reg = (prds == test_label).sum().astype(int) / len(test_label)
print('The accuracy of the Logistic Regression is ', accuracy_log_reg * 100, '%')





