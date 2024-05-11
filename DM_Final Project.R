# Data Preprocessing

# Load necessary libraries
library(dplyr)
library(tidyr)

# Load dataset
data <- read.csv("D:/Semester/13) Spring 2024/Data Mining/Data Mining Final/heart (1).csv")
data

# Display the structure of the dataset
str(data)

# Display the column names
head(data)

# Display the summary statistics of the dataset
summary(data)

# Check for duplicate values in the dataset
sum(duplicated(data))

# Check for missing values in the dataset
sum(is.na(data))

# Data Preprocessing
# Handle missing values
data <- data %>%
  mutate_all(~ ifelse(is.na(.), mean(., na.rm = TRUE), .))
data

# Handle duplicates
data <- data %>% distinct()
data

# Encode categorical variables
# data <- data %>% mutate_if(is.character, as.factor)
# data

# Normalize numerical features
# numerical_features <- sapply(data, is.numeric)
# data[, numerical_features] <- scale(data[, numerical_features])

# Convert categorical variables to factors
categorical_columns <- c('sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal')
data[categorical_columns] <- lapply(data[categorical_columns], as.factor)

# Scale numerical columns
numerical_columns <- c('age', 'trestbps', 'chol', 'thalach', 'oldpeak')
data[numerical_columns] <- scale(data[numerical_columns])

# Split Dataset into training and test sets
set.seed(123)
train_indices <- sample(1:nrow(data), 0.7 * nrow(data))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# Exploratory Data Analysis (EDA)

# Load necessary libraries
library(ggplot2)

# Distribution of target variable
ggplot(train_data, aes(x = target)) + geom_bar(fill = "blue") + ggtitle("Distribution of Target Variable")

# Correlation matrix
library(corrplot)
cor_matrix <- cor(train_data[numerical_columns])
corrplot(cor_matrix, method = "circle")

# Distributions of selected numerical features
for (feature in numerical_columns) {
  print(
    ggplot(train_data, aes_string(x = feature)) +
      geom_histogram(fill = "blue", alpha = 0.7, bins = 30) +
      ggtitle(paste("Distribution of", feature)) +
      theme_minimal()
  )
}


# Cluster Analysis

# Load necessary libraries
library(cluster)
library(factoextra)

# K-means clustering
set.seed(42)
kmeans_result <- kmeans(train_data[, numerical_columns], centers = 3)
fviz_cluster(kmeans_result, data = train_data[, numerical_columns])



# Classification

# Decision Tree Classifier
# Load necessary library
library(rpart)
library(caret)
library(ggplot2)
library(e1071)

# Train Decision Tree model
tree_model <- rpart(target ~ ., data = train_data, method = "class")

# Predict and evaluate
tree_predictions <- predict(tree_model, test_data, type = "class")
confusion_tree <- confusionMatrix(tree_predictions, test_data$target)
train_data$target <- as.factor(train_data$target)
test_data$target <- as.factor(test_data$target)

# Print confusion matrix
print(confusion_tree)

# Function to plot confusion matrix
plot_confusion_matrix <- function(cm, title) {
  cm_data <- as.data.frame(cm$table)
  ggplot(data = cm_data, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), color = "black", size = 4) +
    scale_fill_gradient(low = "white", high = "red") +
    ggtitle(title) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# Plot confusion matrix
plot_confusion_matrix(confusion_tree, "Decision Tree Confusion Matrix")

# Naïve Bayes Classifier
# Load necessary library
library(e1071)

# Train Naïve Bayes model
nb_model <- naiveBayes(target ~ ., data = train_data)

# Predict and evaluate
nb_predictions <- predict(nb_model, test_data)
confusion_nb <- confusionMatrix(nb_predictions, test_data$target)

# Print confusion matrix
print(confusion_nb)

# Plot confusion matrix
plot_confusion_matrix(confusion_nb, "Naïve Bayes Confusion Matrix")



# Support Vector Machine (SVM) Classifier
# Load necessary library

# Train SVM model
svm_model <- svm(target ~ ., data = train_data)

# Predict and evaluate
svm_predictions <- predict(svm_model, test_data)
confusion_svm <- confusionMatrix(svm_predictions, test_data$target)

# Print confusion matrix
print(confusion_svm)

# Plot confusion matrix
plot_confusion_matrix(confusion_svm, "SVM Confusion Matrix")





model <- naiveBayes(target ~ ., data = train_data)
saveRDS(model, file = "model.rds")
library(Rserve)
Rserve()