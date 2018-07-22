setwd("~/Desktop/project/da/project 3")

data = read.csv('Customer Churn Data.csv')
data$Id = NULL
data$total_day_minutes = NULL
data$total_eve_minutes = NULL
data$total_night_minutes = NULL
data$total_intl_minutes = NULL
data$phone_number = NULL

set.seed(123)
sample <- sample.int(n = nrow(data), size = floor(.75*nrow(data)), replace = F)
train <- data[sample, ]
test  <- data[-sample, ]

x_train = train[,1:15]
y_train = as.integer(train$churn)
x_test = test[,1:15]
y_test = as.integer(test$churn)

#naive bayes
library(e1071)
nbayes <- naiveBayes(train$churn~., data = train)
naive_pred = predict(nbayes, newdata = test[,-16], type = 'class')
  
matrix1 = table(test$churn, naive_pred)
TN1 = matrix1[1]
FN1 = matrix1[2]
FP1 = matrix1[3]
TP1 = matrix1[4]

Recall_NB = TP1/(TP1+FN1)
Precision_NB = TP1/(TP1+FP1)
Accuracy_NB = (TP1+TN1)/(TP1+TN1+FP1+FN1)
synth1 = as.data.frame(table(Recall_NB,Precision_NB,Accuracy_NB))
synth1 = synth1[,-c(4)]

#decision tree
library(rpart)
library(rpart.plot)
tree <- rpart(churn~.,train)
rpart.plot(tree)

tree_pred = predict(tree , newdata = test[,-16], type = 'class')

matrix2 = table(test$churn, tree_pred)
TN2 = matrix2[1]
FN2 = matrix2[2]
FP2 = matrix2[3]
TP2 = matrix2[4]

Recall_DT = TP2/(TP2+FN2)
Precision_DT = TP2/(TP2+FP2)
Accuracy_DT = (TP2+TN2)/(TP2+TN2+FP2+FN2)
synth2 = as.data.frame(table(Recall_DT,Precision_DT,Accuracy_DT))
synth2 = synth2[,-c(4)]

#SVM
svm_model <- svm(formula = churn ~ ., data = train, type = 'C-classification', kernel = 'radial')

svm_pred = predict(svm_model , newdata = test[,-16])

matrix3 = table(test$churn, svm_pred)
TN3 = matrix3[1]
FN3 = matrix3[2]
FP3 = matrix3[3]
TP3 = matrix3[4]

Recall_svm = TP3/(TP3+FN3)
Precision_svm = TP3/(TP3+FP3)
Accuracy_svm = (TP3+TN3)/(TP3+TN3+FP3+FN3)
synth3 = as.data.frame(table(Recall_svm,Precision_svm,Accuracy_svm))
synth3 = synth3[,-c(4)]