# Read in the data
# source: https://www.kaggle.com/andrewmvd/heart-failure-clinical-data
heart = read.csv("heart_failure_clinical_records_dataset.csv")
heart$DEATH_EVENT = as.factor(heart$DEATH_EVENT)
str(heart)

# Calculate baseline accuracy based on most frequent value
max(table(heart$DEATH_EVENT)) / nrow(heart)

# Split the data
library(caTools)
set.seed(1)
spl = sample.split(heart$DEATH_EVENT, SplitRatio = 0.7)
Train = subset(heart, spl==TRUE)
Test = subset(heart, spl==FALSE)

# CART model
library(rpart)
library(rpart.plot)
HeartTree = rpart(DEATH_EVENT ~ ., data = Train, method="class", minbucket=5)
prp(HeartTree)
PredictCART = predict(HeartTree, newdata = Test, type = "class")
confusionMatrix = table(Test$DEATH_EVENT, PredictCART)
confusionMatrix
# Calculate accuracy from confusion matrix
sum(diag(confusionMatrix)) / nrow(Test)

# ROC curve
library(ROCR)
PredictROC = predict(HeartTree, newdata = Test)
PredictROC
pred = prediction(PredictROC[,2], Test$DEATH_EVENT)
perf = performance(pred, "tpr", "fpr")
plot(perf)

# Random Forests
library(randomForest)
HeartForest = randomForest(DEATH_EVENT ~ ., data = Train, ntree=200, nodesize=25 )
PredictForest = predict(HeartForest, newdata = Test)
forestMatrix = table(Test$DEATH_EVENT, PredictForest)
# Calculate accuracy from confusion matrix
sum(diag(forestMatrix)) / nrow(Test)

# Cross-validation
library(caret)
library(e1071)
numFolds = trainControl( method = "cv", number = 10 )
cpGrid = expand.grid( .cp = seq(0.005,0.06,0.005)) 
train(DEATH_EVENT ~ ., data = Train, method = "rpart", trControl = numFolds, tuneGrid = cpGrid )

# Cross-validated CART model
HeartTreeCV = rpart(DEATH_EVENT ~ ., data = Train, method="class", cp = 0.055)
PredictCV = predict(HeartTreeCV, newdata = Test, type = "class")
treeMatrixCV = table(Test$DEATH_EVENT, PredictForest)
# Calculate accuracy from confusion matrix
sum(diag(treeMatrixCV)) / nrow(Test)

