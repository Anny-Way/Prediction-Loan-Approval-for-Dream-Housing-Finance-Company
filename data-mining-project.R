#input data
rm(list=ls())
setwd("C:\\Users\\Ziwei Wang\\Desktop\\study\\data mining\\project")
loan = read.csv("loan_data.csv", stringsAsFactors=FALSE)
Loan_Status  = loan$Loan_Status

#proportion of "Y" status
sum(loan$Loan_Status=="Y")/dim(loan[1])

length(unique(loan$Gender))
unique(loan$Gender)  # missing value exists
loan$Gender[loan$Gender==''] <- NA
loan$Gender[loan$Gender=='Male'] <- 1
loan$Gender[loan$Gender=='Female'] <- 0
sum(is.na(loan$Gender))
loan$Gender <- as.numeric(loan$Gender)

length(unique(loan$Married))
unique(loan$Married)  # missing value exists
loan$Married[loan$Married==''] <- NA
loan$Married[loan$Married=='Yes'] <- 1
loan$Married[loan$Married=='No'] <- 0
sum(is.na(loan$Married))
loan$Married <- as.numeric(loan$Married)

length(unique(loan$Dependents))
unique(loan$Dependents)  # missing value exists
loan$Dependents[loan$Dependents==''] <- NA
loan$Dependents[loan$Dependents=='3+'] <- 3
sum(is.na(loan$Dependents))
loan$Dependents <- as.numeric(loan$Dependents)

length(unique(loan$Education))
unique(loan$Education)  # no missing value
loan$Education[loan$Education=='Graduate'] <- 1
loan$Education[loan$Education=='Not Graduate'] <- 0
loan$Education <- as.numeric(loan$Education)

length(unique(loan$Self_Employed))
unique(loan$Self_Employed)  # missing value exists
loan$Self_Employed[loan$Self_Employed==''] <- NA
loan$Self_Employed[loan$Self_Employed=='No'] <- 0
loan$Self_Employed[loan$Self_Employed=='Yes'] <- 1
sum(is.na(loan$Self_Employed))
loan$Self_Employed <- as.numeric(loan$Self_Employed)

length(unique(loan$Property_Area))
unique(loan$Property_Area)  # no missing value
loan$Property_Area[loan$Property_Area=='Rural'] <- 1
loan$Property_Area[loan$Property_Area=='Semiurban'] <- 2
loan$Property_Area[loan$Property_Area=='Urban'] <- 3
loan$Property_Area <- as.numeric(loan$Property_Area)

unique(loan$Loan_Status)
loan$Loan_Status[loan$Loan_Status=='Y'] <- 1
loan$Loan_Status[loan$Loan_Status=='N'] <- 0
loan$Loan_Status <- as.numeric(loan$Loan_Status)

# Removing data
loan_noca <- subset(loan, select = -c(Loan_ID, Loan_Status))
summary(loan_noca)

library(mice)
# impute the missing values
imputed_noca <- mice(loan_noca, m=5, maxit=50, method = 'cart', seed = 666)
summary(imputed_noca)

library(VIM)
mice_plot <- aggr(loan_noca, col=c('navyblue','yellow'),
                  numbers=TRUE, sortVars=TRUE,
                  labels=names(loan_noca), cex.axis=.7,
                  gap=3, ylab=c("Missing data","Pattern"))

# Check imputed values
imputed_noca$imp$Loan_Amount_Term

# Get complete data ( 3rd out of 5)
completeData <- complete(imputed_noca, 3)
sum(is.na(completeData))
head(completeData)

#check linearity
corr = cor(completeData)

## Factor Analysis for a mix of continuous and categorical data
library(FactoMineR)
library(factoextra)
## change data type in order to identify continuous and categorical data type
c = completeData
c$Gender[c$Gender=='1'] <- 'Male'
c$Gender[c$Gender=='0'] <- 'Female'
c$Married[c$Married=='1'] <- 'Yes'
c$Married[c$Married=='0'] <- 'No'
c$Dependents[c$Dependents=='3'] <- 'three more'
c$Dependents[c$Dependents=='2'] <- 'two'
c$Dependents[c$Dependents=='1'] <- 'one'
c$Dependents[c$Dependents=='0'] <- 'none'
c$Education[c$Education=='1'] <- 'Graduate'
c$Education[c$Education=='0'] <- 'Not Graduate'
c$Self_Employed[c$Self_Employed=='0'] <- 'No'
c$Self_Employed[c$Self_Employed=='1'] <- 'Yes'
c$Property_Area[c$Property_Area=='1'] <- 'Rural'
c$Property_Area[c$Property_Area=='2'] <- 'Semiurban'
c$Property_Area[c$Property_Area=='3'] <- 'Urban'
c$Credit_History[c$Credit_History=='1'] <- 'Yes'
c$Credit_History[c$Credit_History=='0'] <- 'No'

res.famd <- FAMD(c, ncp = 14, graph = FALSE)
print(res.famd)

eign.val <- get_eigenvalue(res.famd)
eign.val
fviz_screeplot(res.famd)
# Plot of variables
fviz_famd_var(res.famd, repel = TRUE) 
# Contribution to the first dimension
fviz_contrib(res.famd, "var", axes = 1)
# Contribution to the second dimension
fviz_contrib(res.famd, "var", axes = 2)

q <- get_famd(res.famd, element = c("ind", "var", "quanti.var", "quali.var"))
famd_data = q$coord

newdata = completeData

#extreme value
#Loan Amount
TotalIncome = log(newdata$ApplicantIncome + newdata$CoapplicantIncome)
newdata = data.frame(newdata, TotalIncome, Loan_Status)

#ROC
library(ROCR)
library(pROC)
rocplot = function(pred, truth){
  predob = prediction(pred, truth)
  perf = performance(predob, "tpr", "fpr")
  plot(perf)
}

#######################################################################
#famd data
famd_data = data.frame(famd_data, Loan_Status)
set.seed(1)
n_train1 = sample.int(n = nrow(famd_data), size = floor(.60*nrow(famd_data)), replace = F)
n_test1 = -n_train1
train1 = famd_data[n_train1,]
test1 = famd_data[n_test1,]
n1 = dim(test1)[1]

#######################################################################
#original data
#train&test
set.seed(666)
n_train = sample.int(n = nrow(newdata), size = floor(.60*nrow(newdata)), replace = F)
n_test = -n_train
train = newdata[n_train,]
test = newdata[n_test,]
n = dim(test)[1]

#choose variable
#lasso
x = model.matrix(Loan_Status~Gender + Married + Dependents + Credit_History + Education + Self_Employed + Property_Area + LoanAmount + Loan_Amount_Term + TotalIncome, data = newdata)[,-1]
mlasso = cv.glmnet(x, newdata[,13], alpha = 1, family = "binomial")
lambda_best = mlasso$lambda.min
lambda_best
plot(mlasso)
#fit the model on entire data using best lambda
mbest = glmnet(x, newdata[,13], alpha = 1, family = "binomial")
predict(mbest, s = lambda_best, type = "coefficients")

#logistic regression
#dim
m_lr1 = glm(Loan_Status~., data = famd_data, family = binomial, subset = n_train1)
summary(m_lr1)
prob_lr1 = predict(m_lr1, test1, type = "response")
pre_lr1 = rep(0,n1)
pre_lr1[prob_lr1>0.75] = "Y"
pre_lr1[prob_lr1<0.75] = "N"
#test error
mean(pre_lr1 != test1[,"Loan_Status"])
c1 = table(test1$Loan_Status, pre_lr1)
c1
#plot ROC
pred_lr1 = pre_lr1
pred_lr1[pre_lr1 == "Y"] = 1
pred_lr1[pre_lr1 == "N"] = 0
pred_lr1 = as.numeric(pred_lr1)
rocplot(pred_lr1, loan[n_test1,"Loan_Status"])
title(main = "ROC of Logistic regression")
#auc
auc(roc(prob_lr1, loan[n_test1,"Loan_Status"]))
#0.6333

#logistic regression(2 income)
m_lr2 = glm(Loan_Status~Gender + Married + Dependents + Credit_History + Education + Self_Employed + LoanAmount + Loan_Amount_Term + Property_Area + ApplicantIncome + CoapplicantIncome, data = newdata, family = binomial, subset = n_train)
summary(m_lr2)
prob_lr2 = predict(m_lr2, test, type = "response")
pre_lr2 = rep(0,n)
pre_lr2[prob_lr2>0.75] = "Y"
pre_lr2[prob_lr2<0.75] = "N"
#test error
mean(pre_lr2 != test[,"Loan_Status"])
c2 = table(test$Loan_Status, pre_lr2)
c2
#plot ROC
rocplot(prob_lr2, loan[n_test,"Loan_Status"])
title(main = "ROC of Logistic regression")
#auc
auc(roc(loan[n_test,"Loan_Status"], prob_lr2))
#0.6432

#logistic regression(total income)
m_lr3 = glm(Loan_Status~Gender + Married + Dependents + Credit_History + Education + Self_Employed + LoanAmount + Loan_Amount_Term + Property_Area + TotalIncome, data = newdata, family = binomial, subset = n_train)
summary(m_lr3)
prob_lr3 = predict(m_lr3, test, type = "response")
pre_lr3 = rep(0,n)
pre_lr3[prob_lr3>0.75] = "Y"
pre_lr3[prob_lr3<0.75] = "N"
#test error
mean(pre_lr3 != test[,"Loan_Status"])
c3 = table(test$Loan_Status, pre_lr3)
c3
#plot ROC
pred_lr3 = pre_lr3
pred_lr3[pre_lr3 == "Y"] = 1
pred_lr3[pre_lr3 == "N"] = 0
pred_lr3 = as.numeric(pred_lr3)
rocplot(pred_lr3, loan[n_test,"Loan_Status"])
title(main = "ROC of Logistic regression")
#auc
auc(roc(pred_lr3, loan[n_test,"Loan_Status"]))
#0.635

#LDA
library(MASS)
#dim
m_lda1 = lda(Loan_Status~Dim.1+Dim.2+Dim.3+Dim.4+Dim.5+Dim.6+Dim.7+Dim.8+Dim.9+Dim.10+Dim.11+Dim.12+Dim.13+Dim.14, famd_data, subset = n_train1)
pre_lda1 = predict(m_lda1, test1)
#test error
mean(pre_lda1$class != test1[,"Loan_Status"])
c4 = table(test1$Loan_Status, pre_lda1$class)
c4
#plot ROC
pred_lda1 = as.character(pre_lda1$class)
pred_lda1[pre_lda1$class == "Y"] = 1
pred_lda1[pre_lda1$class == "N"] = 0
pred_lda1 = as.numeric(pred_lda1)
rocplot(pred_lda1, loan[n_test1,"Loan_Status"])
title(main = "ROC of LDA")
#auc
auc(roc(pred_lda1, loan[n_test1,"Loan_Status"]))
#0.8341

#2 income
m_lda2 = lda(Loan_Status~Gender + Married + Dependents + Credit_History + Education + Self_Employed + LoanAmount + Loan_Amount_Term + Property_Area + ApplicantIncome + CoapplicantIncome, newdata, subset = n_train)
pre_lda2 = predict(m_lda2, test)
#test error
mean(pre_lda2$class != test[,"Loan_Status"])
c5 = table(test$Loan_Status, pre_lda2$class)
c5
#plot ROC
pred_lda2 = as.character(pre_lda2$class)
pred_lda2[pre_lda2$class == "Y"] = 1
pred_lda2[pre_lda2$class == "N"] = 0
pred_lda2 = as.numeric(pred_lda2)
rocplot(pred_lda2, loan[n_test,"Loan_Status"])
title(main = "ROC of LDA")
#auc
auc(roc(pred_lda2, loan[n_test,"Loan_Status"]))
#0.8341

#total income
m_lda3 = lda(Loan_Status~Gender + Married + Dependents + Credit_History + Education + Self_Employed + LoanAmount + Loan_Amount_Term + Property_Area + TotalIncome, newdata, subset = n_train)
pre_lda3 = predict(m_lda3, test)
#test error
mean(pre_lda3$class != test[,"Loan_Status"])
c6 = table(test$Loan_Status, pre_lda3$class)
c6
#plot ROC
pred_lda3 = as.character(pre_lda3$class)
pred_lda3[pre_lda3$class == "Y"] = 1
pred_lda3[pre_lda3$class == "N"] = 0
pred_lda3 = as.numeric(pred_lda3)
rocplot(pred_lda3, loan[n_test,"Loan_Status"])
title(main = "ROC of LDA")
#auc
auc(roc(pred_lda3, loan[n_test,"Loan_Status"]))
#0.8341

#QDA
#2 income
m_qda1 = qda(Loan_Status~Gender + Married + Dependents + Credit_History + Education + Self_Employed + LoanAmount + Loan_Amount_Term + Property_Area + ApplicantIncome + CoapplicantIncome, newdata, subset = n_train)
pre_qda1 = predict(m_qda1, test)
#test error
#The test error rate for QDA is 
mean(pre_qda1$class != test[,"Loan_Status"])
c7 = table(test$Loan_Status, pre_qda1$class)
c7
#ROC
pred_qda1 = as.character(pre_qda1$class)
pred_qda1[pre_qda1$class == "Y"] = 1
pred_qda1[pre_qda1$class == "N"] = 0
pred_qda1 = as.numeric(pred_qda1)
rocplot(pred_qda1, loan[n_test,"Loan_Status"])
title(main = "ROC of QDA")
#auc
auc(roc(pred_qda1, loan[n_test,"Loan_Status"]))
#0.7809

#total income
m_qda2 = qda(Loan_Status~Gender + Married + Dependents + Credit_History + Education + Self_Employed + LoanAmount + Loan_Amount_Term + Property_Area + TotalIncome, newdata, subset = n_train)
pre_qda2 = predict(m_qda2, test)
#test error
#The test error rate for QDA is 
mean(pre_qda2$class != test[,"Loan_Status"])
c8 = table(test$Loan_Status, pre_qda2$class)
c8
#ROC
pred_qda2 = as.character(pre_qda2$class)
pred_qda2[pre_qda2$class == "Y"] = 1
pred_qda2[pre_qda2$class == "N"] = 0
pred_qda2 = as.numeric(pred_qda2)
rocplot(pred_qda2, loan[n_test,"Loan_Status"])
title(main = "ROC of QDA")
#auc
auc(roc(pred_qda2, loan[n_test,"Loan_Status"]))
#0.7708

#total income
m_qda3 = qda(Loan_Status~Dim.1+Dim.2+Dim.3+Dim.4+Dim.5+Dim.6+Dim.7+Dim.8+Dim.9+Dim.10+Dim.11+Dim.12+Dim.13+Dim.14, famd_data, subset = n_train1)
pre_qda3 = predict(m_qda3, test1)
#test error
#The test error rate for QDA is 
mean(pre_qda3$class != test1[,"Loan_Status"])
c9 = table(test1$Loan_Status, pre_qda3$class)
c9
#ROC
pred_qda3 = as.character(pre_qda3$class)
pred_qda3[pre_qda3$class == "Y"] = 1
pred_qda3[pre_qda3$class == "N"] = 0
pred_qda3 = as.numeric(pred_qda3)
rocplot(pred_qda3, loan[n_test1,"Loan_Status"])
title(main = "ROC of QDA")
#auc
auc(roc(pred_qda3, loan[n_test1,"Loan_Status"]))
#0.7982

#knn (total income)
library(class)
training1 = newdata[n_train,c("Gender", "Married", "Dependents", "Credit_History", "Education", "Self_Employed", "LoanAmount", "Loan_Amount_Term", "Property_Area", "TotalIncome")]
testing1 = newdata[n_test,c("Gender", "Married", "Dependents", "Credit_History", "Education", "Self_Employed", "LoanAmount", "Loan_Amount_Term", "Property_Area", "TotalIncome")]
LoanStatus_train = newdata$Loan_Status[n_train]

training2 = newdata[n_train,c("Gender", "Married", "Dependents", "Credit_History", "Education", "Self_Employed", "LoanAmount", "Loan_Amount_Term", "Property_Area", "ApplicantIncome", "CoapplicantIncome")]
testing2 = newdata[n_test,c("Gender", "Married", "Dependents", "Credit_History", "Education", "Self_Employed", "LoanAmount", "Loan_Amount_Term", "Property_Area", "ApplicantIncome", "CoapplicantIncome")]

training3 = famd_data[n_train1,c("Dim.1","Dim.2","Dim.3","Dim.4","Dim.5","Dim.6","Dim.7","Dim.8","Dim.9","Dim.10","Dim.11","Dim.12","Dim.13","Dim.14")]
testing3 = famd_data[n_test1,c("Dim.1","Dim.2","Dim.3","Dim.4","Dim.5","Dim.6","Dim.7","Dim.8","Dim.9","Dim.10","Dim.11","Dim.12","Dim.13","Dim.14")]
LoanStatus_train3 = famd_data$Loan_Status[n_train1]

#k = 1(total)
pre_knn1 = knn(training1, testing1, LoanStatus_train, k = 1)
#The test error for k = 1 is 
mean(pre_knn1 != test[,"Loan_Status"])
c10 = table(test$Loan_Status, pre_knn1)
c10
#ROC
pred_knn1 = as.character(pre_knn1)
pred_knn1[pre_knn1 == "Y"] = 1
pred_knn1[pre_knn1 == "N"] = 0
pred_knn1 = as.numeric(pred_knn1)
rocplot(pred_knn1, loan[n_test,"Loan_Status"])
title(main = "ROC of KNN (k=1)")
#auc
auc(roc(pred_knn1, loan[n_test,"Loan_Status"]))
#0.6361

#k = 10(total)
pre_knn2 = knn(training1, testing1, LoanStatus_train, k = 10)
#The test error for k = 10 is 
mean(pre_knn2 != test[,"Loan_Status"])
c11 = table(test$Loan_Status, pre_knn2)
c11
#ROC
pred_knn2 = as.character(pre_knn2)
pred_knn2[pre_knn2 == "Y"] = 1
pred_knn2[pre_knn2 == "N"] = 0
pred_knn2 = as.numeric(pred_knn2)
rocplot(pred_knn2, loan[n_test,"Loan_Status"])
title(main = "ROC of KNN (k=10)")
#auc
auc(roc(pred_knn2, loan[n_test,"Loan_Status"]))
#0.7206

#k = 100(total)
pre_knn3 = knn(training1, testing1, LoanStatus_train, k = 100)
#The test error for k = 100 is 
mean(pre_knn3 != test[,"Loan_Status"])
#From the above outputs, we can see that k=10 performs the best.
c12 = table(test$Loan_Status, pre_knn3)
c12
#ROC
pred_knn3 = as.character(pre_knn3)
pred_knn3[pre_knn3 == "Y"] = 1
pred_knn3[pre_knn3 == "N"] = 0
pred_knn3 = as.numeric(pred_knn3)
rocplot(pred_knn3, loan[n_test,"Loan_Status"])
title(main = "ROC of KNN (k=100)")
#auc
#auc = 0.5(?), because all the predictor values are "Y".

#k = 10(2 income)
pre_knn4 = knn(training2, testing2, LoanStatus_train, k = 10)
#The test error for k = 10 is 
mean(pre_knn4 != test[,"Loan_Status"])
c13 = table(test$Loan_Status, pre_knn4)
c13
#ROC
pred_knn4 = as.character(pre_knn4)
pred_knn4[pre_knn2 == "Y"] = 1
pred_knn4[pre_knn2 == "N"] = 0
pred_knn4 = as.numeric(pred_knn4)
rocplot(pred_knn4, loan[n_test,"Loan_Status"])
title(main = "ROC of KNN (k=10)")
#auc
auc(roc(pred_knn4, loan[n_test,"Loan_Status"]))
#0.7206

#k = 1(dim)
pre_knn5 = knn(training3, testing3, LoanStatus_train3, k = 1)
#The test error for k = 1 is 
mean(pre_knn5 != test1[,"Loan_Status"])
c14 = table(test1$Loan_Status, pre_knn5)
c14
#ROC
pred_knn5 = as.character(pre_knn1)
pred_knn5[pre_knn5 == "Y"] = 1
pred_knn5[pre_knn5 == "N"] = 0
pred_knn5 = as.numeric(pred_knn5)
rocplot(pred_knn5, loan[n_test1,"Loan_Status"])
title(main = "ROC of KNN (k=1)")
#auc
auc(roc(pred_knn5, loan[n_test1,"Loan_Status"]))
#0.6257

#k = 10(dim)
pre_knn6 = knn(training3, testing3, LoanStatus_train3, k = 10)
#The test error for k = 10 is 
mean(pre_knn6 != test1[,"Loan_Status"])
c15 = table(test1$Loan_Status, pre_knn6)
c15
#ROC
pred_knn6 = as.character(pre_knn6)
pred_knn6[pre_knn6 == "Y"] = 1
pred_knn6[pre_knn6 == "N"] = 0
pred_knn6 = as.numeric(pred_knn6)
rocplot(pred_knn6, loan[n_test1,"Loan_Status"])
title(main = "ROC of KNN (k=10)")
#auc
auc(roc(pred_knn6, loan[n_test1,"Loan_Status"]))
#0.7952

#k = 100(dim)
pre_knn7 = knn(training3, testing3, LoanStatus_train3, k = 100)
#The test error for k = 100 is 
mean(pre_knn7 != test1[,"Loan_Status"])
#From the above outputs, we can see that k=10 performs the best.
c16 = table(test1$Loan_Status, pre_knn7)
c16
#ROC
pred_knn7 = as.character(pre_knn7)
pred_knn7[pre_knn3 == "Y"] = 1
pred_knn7[pre_knn3 == "N"] = 0
pred_knn7 = as.numeric(pred_knn7)
rocplot(pred_knn7, loan[n_test1,"Loan_Status"])
title(main = "ROC of KNN (k=100)")
#auc
#auc = 0.5(?), because all the predictor values are "Y".

#svm
library(e1071)
set.seed(1)
# total income
m_svm1 = tune(svm, Loan_Status~Gender + Married + Dependents + Credit_History + Education + Self_Employed + LoanAmount + Loan_Amount_Term + Property_Area + TotalIncome, data = train, kernel = "radial", ranges = list(cost = c(0.1, 1, 10, 100), gamma =c(0.5, 1, 2, 3)))
summary(m_svm1)
pre_svm1 = predict(m_svm1$best.model, test)
mean(pre_svm1 != test[,"Loan_Status"])
c17 = table(test$Loan_Status, pre_svm1)
c17
#ROC
pred_svm1 = as.character(pre_svm1)
pred_svm1[pre_svm1 == "Y"] = 1
pred_svm1[pre_svm1 == "N"] = 0
pred_svm1 = as.numeric(pred_svm1)
rocplot(pred_svm1, loan[n_test,"Loan_Status"])
title(main = "ROC of SVM")
#auc
auc(roc(pred_svm1, loan[n_test,"Loan_Status"]))
#0.7191

#2 income
m_svm2 = tune(svm, Loan_Status~Gender + Married + Dependents + Credit_History + Education + Self_Employed + LoanAmount + Loan_Amount_Term + Property_Area + ApplicantIncome + CoapplicantIncome, data = train, kernel = "radial", ranges = list(cost = c(0.1, 1, 10, 100), gamma =c(0.5, 1, 2, 3)))
pre_svm2 = predict(m_svm2$best.model, test)
mean(pre_svm2 != test[,"Loan_Status"])
c18 = table(test$Loan_Status, pre_svm2)
c18
#ROC
pred_svm2 = as.character(pre_svm2)
pred_svm2[pre_svm2 == "Y"] = 1
pred_svm2[pre_svm2 == "N"] = 0
pred_svm2 = as.numeric(pred_svm2)
rocplot(pred_svm2, loan[n_test,"Loan_Status"])
title(main = "ROC of SVM")
#auc
auc(roc(pred_svm2, loan[n_test,"Loan_Status"]))
#0.7317

#dim
m_svm3 = tune(svm, Loan_Status~Dim.1+Dim.2+Dim.3+Dim.4+Dim.5+Dim.6+Dim.7+Dim.8+Dim.9+Dim.10+Dim.11+Dim.12+Dim.13+Dim.14, data = train1, kernel = "radial", ranges = list(cost = c(0.1, 1, 10, 100), gamma =c(0.5, 1, 2, 3)))
pre_svm3 = predict(m_svm3$best.model, test1)
mean(pre_svm3 != test1[,"Loan_Status"])
c19 = table(test1$Loan_Status, pre_svm3)
c19
#ROC
pred_svm3 = as.character(pre_svm3)
pred_svm3[pre_svm3 == "Y"] = 1
pred_svm3[pre_svm3 == "N"] = 0
pred_svm3 = as.numeric(pred_svm3)
rocplot(pred_svm2, loan[n_test1,"Loan_Status"])
title(main = "ROC of SVM")
#auc
auc(roc(pred_svm3, loan[n_test1,"Loan_Status"]))
#0.6889

#random forest
library(randomForest)
set.seed(1)
#total income
m_rf1 = randomForest(Loan_Status~Gender + Married + Dependents + Credit_History + Education + Self_Employed + Property_Area + LoanAmount + Loan_Amount_Term + TotalIncome, train, ntree = 500, mtry = 3, importance = TRUE)
pre_rf1 = predict(m_rf1, test)
#test MSE
mean(pre_rf1 != test[,"Loan_Status"])
importance(m_rf1)
c20 = table(test$Loan_Status, pre_rf1)
c20
#ROC
pred_rf1 = as.character(pre_rf1)
pred_rf1[pre_rf1 == "Y"] = 1
pred_rf1[pre_rf1 == "N"] = 0
pred_rf1 = as.numeric(pred_rf1)
rocplot(pred_rf1, loan[n_test,"Loan_Status"])
title(main = "ROC of Random Forest")
#auc
auc(roc(pred_rf1, loan[n_test,"Loan_Status"]))
#0.7906

#2 income
m_rf2 = randomForest(Loan_Status~Gender + Married + Dependents + Credit_History + Education + Self_Employed + Property_Area + LoanAmount + Loan_Amount_Term + ApplicantIncome + CoapplicantIncome, train, ntree = 500, mtry = 3, importance = TRUE)
pre_rf2 = predict(m_rf2, test)
#test MSE
mean(pre_rf2 != test[,"Loan_Status"])
importance(m_rf2)
c21 = table(test$Loan_Status, pre_rf2)
c21
#ROC
pred_rf2 = as.character(pre_rf2)
pred_rf2[pre_rf2 == "Y"] = 1
pred_rf2[pre_rf2 == "N"] = 0
pred_rf2 = as.numeric(pred_rf2)
rocplot(pred_rf2, loan[n_test,"Loan_Status"])
title(main = "ROC of Random Forest")
#auc
auc(roc(pred_rf2, loan[n_test,"Loan_Status"]))
#0.7834

#dim
m_rf3 = randomForest(Loan_Status~Dim.1+Dim.2+Dim.3+Dim.4+Dim.5+Dim.6+Dim.7+Dim.8+Dim.9+Dim.10+Dim.11+Dim.12+Dim.13+Dim.14, train1, ntree = 500, mtry = 4, importance = TRUE)
pre_rf3 = predict(m_rf3, test1)
#test MSE
mean(pre_rf3 != test1[,"Loan_Status"])
importance(m_rf3)
c22 = table(test1$Loan_Status, pre_rf3)
c22
#ROC
pred_rf3 = as.character(pre_rf3)
pred_rf3[pre_rf3 == "Y"] = 1
pred_rf3[pre_rf3 == "N"] = 0
pred_rf3 = as.numeric(pred_rf3)
rocplot(pred_rf3, loan[n_test1,"Loan_Status"])
title(main = "ROC of Random Forest")
#auc
auc(roc(pred_rf3, loan[n_test1,"Loan_Status"]))
#0.7227

