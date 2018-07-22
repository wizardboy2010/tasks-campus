# Set Working Directory
path = "/home/spandan/DataAnalytics"
setwd(path)
dataset = read.csv('Customer Churn Data.csv')

# Check for missing data
summary(is.na(dataset))

# Remove Irrelevant columns
dataset = dataset[,-c(1,2,5)]

# Rectify the dataset
# Convert labels to categorical, and boolean
dataset$international_plan = (as.numeric(dataset$international_plan))-1
dataset$voice_mail_plan = (as.numeric(dataset$voice_mail_plan))-1
dataset$churn = (as.numeric(dataset$churn))-1
dataset$international_plan = (as.factor(dataset$international_plan))
dataset$voice_mail_plan = (as.factor(dataset$voice_mail_plan))
dataset$churn = (as.factor(dataset$churn))
dataset$area_code = (as.factor(dataset$area_code))


# Univariate Analysis
summary(dataset)

# Split on basis of the data$churn for further univariate analysis
out = split(dataset, dataset$churn)
false_val = out[[1]]
true_val = out[[2]]

# Remove insignificant variables
keeps = c("international_plan", "voice_mail_plan", "number_vmail_messages", "total_day_minutes", "total_day_charge", "total_eve_minutes", "total_night_calls", "total_intl_calls", "number_customer_service_calls", "churn")
dataset = dataset[keeps]

# Split test and train data sets. We're using caTools here.
install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$churn, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#[SVM] Applying SVM model
install.packages('e1071')
library(e1071)
classifier = svm(formula = churn ~ ., data = training_set, type = 'C-classification', kernel = 'radial')

#[SVM] Predicting the Test set results
y_pred1 = predict(classifier, newdata = test_set[-10])

#[SVM] Confusion Matrix
cm1 = table(test_set[, 10], y_pred1)
cm1 = as.data.frame(cm1)
TN1 = cm1$Freq[1]
FN1 = cm1$Freq[2]
FP1 = cm1$Freq[3]
TP1 = cm1$Freq[4]

#[SVM] Validation Analysis
Recall_svm = TP1/(TP1+FN1)
Precision_svm = TP1/(TP1+FP1)
Accuracy_svm = (TP1+TN1)/(TP1+TN1+FP1+FN1)
synth1 = as.data.frame(table(Recall_svm,Precision_svm,Accuracy_svm))
synth1 = synth1[,-c(4)]

#[DECISIN TREE] Fitting Decision Tree Classification to the Training set
install.packages('rpart')
library(rpart)
classifier = rpart(formula = churn ~ ., data = training_set)

#[DECISIN TREE] Predicting the Test set results
y_pred2 = predict(classifier, newdata = test_set[-10], type = 'class')

#[DECISIN TREE] Confusion Matrix
cm2 = as.data.frame(table(test_set[, 10], y_pred2))
TN2 = cm2$Freq[1]
FN2 = cm2$Freq[2]
FP2 = cm2$Freq[3]
TP2 = cm2$Freq[4]

#[DECISIN TREE] Validation Analysis
Recall_DT = TP2/(TP2+FN2)
Precision_DT = TP2/(TP2+FP2)
Accuracy_DT = (TP2+TN2)/(TP2+TN2+FP2+FN2)
synth2 = as.data.frame(table(Recall_DT,Precision_DT,Accuracy_DT))
synth2 = synth2[,-c(4)]

#[NAIVE BAYES] Naive Bayes Model
library(e1071)
model = naiveBayes(dataset$churn ~ ., data = dataset)
y_pred3 = predict(model, newdata = test_set[-10], type = 'class')

#[NAIVE BAYES] Confusion Matrix
cm2 = as.data.frame(table(test_set[, 10], y_pred3))
TN2 = cm2$Freq[1]
FN2 = cm2$Freq[2]
FP2 = cm2$Freq[3]
TP2 = cm2$Freq[4]

#[NAIVE BAYES] Validation Analysis
Recall_DT = TP2/(TP2+FN2)
Precision_DT = TP2/(TP2+FP2)
Accuracy_DT = (TP2+TN2)/(TP2+TN2+FP2+FN2)
synth2 = as.data.frame(table(Recall_DT,Precision_DT,Accuracy_DT))
synth2 = synth2[,-c(4)]

