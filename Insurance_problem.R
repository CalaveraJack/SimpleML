mydata_ori <- read.csv("insurance.csv")

#************
#Preliminary actions
#************


library(caret)

#Packages installation
install.packages("NeuralNetTools")



#Data inspection and preparation

#Check dimensions
dim(mydata_ori)
head(mydata_ori)
str(mydata_ori)

#Check for missing values
colSums(is.na(mydata_ori))
#No missing values

#Which values there is?
table(mydata_ori$sex)
table(mydata_ori$smoker)
table(mydata_ori$region)

#Outliers of non-factor (non-dummy) vars
#For BMI
boxplot(mydata_ori$bmi,
        horizontal = FALSE,
        lwd = 1,
        col = "darkblue",
        xlab = "",
        ylab = "",
        main = "bmi",
        notch = FALSE,
        border = "grey",
        outpch = 22,
        outbg = "orange",
        whiskcol = "blue",
        whisklty = 2,
        lty = 1)
mean(mydata_ori$bmi)
median(mydata_ori$bmi)
max(mydata_ori$bmi)
min(mydata_ori$bmi)
##Below 18.5	Underweight
#18.5 – 24.9	Healthy Weight
#25.0 – 29.9	Overweight
#30.0 and Above	Obesity
#Mean is !!!!obese!!!! - quite a selection, huh? 
#However, In the United States, the average adult man has a BMI of 26.6 and the average adult woman has a BMI of 26.5
#So it is fine

#For age
boxplot(mydata_ori$age,
        horizontal = FALSE,
        lwd = 1,
        col = "darkblue",
        xlab = "",
        ylab = "",
        main = "Age",
        notch = FALSE,
        border = "grey",
        outpch = 22,
        outbg = "orange",
        whiskcol = "blue",
        whisklty = 2,
        lty = 1)

mean(mydata_ori$age)
median(mydata_ori$age)
max(mydata_ori$age)
min(mydata_ori$age)

#For childen
boxplot(mydata_ori$children,
        horizontal = FALSE,
        lwd = 1,
        col = "darkblue",
        xlab = "",
        ylab = "",
        main = "Children",
        notch = FALSE,
        border = "grey",
        outpch = 22,
        outbg = "orange",
        whiskcol = "blue",
        whisklty = 2,
        lty = 1)

mean(mydata_ori$children)
median(mydata_ori$children)
max(mydata_ori$children)
min(mydata_ori$children)

#For expenses
boxplot(mydata_ori$expenses,
        horizontal = FALSE,
        lwd = 1,
        col = "darkblue",
        xlab = "",
        ylab = "",
        main = "expenses",
        notch = FALSE,
        border = "grey",
        outpch = 22,
        outbg = "orange",
        whiskcol = "blue",
        whisklty = 2,
        lty = 1)

mean(mydata_ori$expenses)
median(mydata_ori$expenses)
max(mydata_ori$expenses)
min(mydata_ori$expenses)
#No extreme unexplained outliers

#Data preparation:
#Dummy variables
library(caret)
mydata <- mydata_ori
dmy <- dummyVars(" ~ .", data = mydata, fullRank=T) #Tildedot means taking everythig, Fullrank means removing one of the dummies
mydata_v1 <- data.frame(predict(dmy, newdata = mydata))

#Scaling
mydata_v2<-preProcess(mydata_v1,method="range")
mydata_v3<-predict(mydata_v2,mydata_v1)


#Colellation check
#Start with Scatterplot
pairs(mydata_v3)
#We see that corellation which might concern us is in age, bmi, children
corr.matrix <- cor(mydata_v3)
View(corr.matrix)
max(corr.matrix)
#Top-5 correlations between IV's: bmi/age (0.107457087), bmi/southeast (0.269927945), all the regions with each-other (0.345395934)
#Let's plot interesting corellations: regional corellation is explained by selection

ggplot(mydata_v1, aes(x=age, y=bmi) ) + geom_hex(bins = 20) + scale_fill_continuous(type = "viridis") + theme_bw() +  geom_smooth(method = "lm", se = FALSE,col="orange")

#For binary variables better using boxplot
ggplot(mydata_v3) + geom_bar(aes(regionsoutheast, bmi, fill = regionsoutheast), position = "dodge", stat = "summary", fun = "mean")

#Partitioning
dim(mydata_v3)
head(mydata_v3)
str(mydata_v3)

set.seed(123)
index <- sample(1:nrow(mydata_v3), nrow(mydata_v3)*0.8, replace=F)
index
mydata_v3.train <- mydata_v3[index,]
mydata_v3.test <- mydata_v3[-index,]
summary(mydata_v3.train)
summary(mydata_v3.test)
#80% to 20% - now we have a training and the test sets



#Preliminary calculations

#Mean of the dependent variable:
default.pred <- mean(mydata_v3.train$expenses)
default.pred

#ESS for training and training set compared to default
default.train.rss <- sum((mydata_v3.train$expenses-default.pred)^2)
default.train.rss
default.test.rss <- sum((mydata_v3.test$expenses-default.pred)^2)
default.test.rss




#Cross-validation and repeats
TControl <- trainControl(method="repeatedcv", number=10,repeats = 2)
#We used repeatedcv parameter with 2 repetition because when choosing the tunegrid for nnet the RMSE and Rsquared started to diverge between tunegrid sizes



#Report creation
report <- data.frame(Model=character(), R2.Train=numeric(), R2.Test=numeric())


#************
#Models
#************


#OLS
set.seed(123)
ols <- train(expenses ~ ., data=mydata_v3.train, method="lm", trControl=TControl, metric="Rsquared")
ols
summary(ols)
ols$finalModel #Inspecting final model


prediction.train <- predict(ols, newdata = mydata_v3.train)
train.rss <- sum((mydata_v3.train$expenses-prediction.train)^2)
ols.r2 <- 1.0-train.rss/default.train.rss
ols.r2
prediction.test <- predict(ols, newdata = mydata_v3.test)
test.rss <- sum((mydata_v3.test$expenses-prediction.test)^2)
ols.pseudoR2 <- 1.0-test.rss/default.test.rss
ols.pseudoR2
report <- rbind(report, data.frame(Model="Linear Regression", R2.Train=ols.r2, R2.Test=ols.pseudoR2))




#Ridge
ridgeGrid <- expand.grid(alpha=0.00, lambda=seq(from=0.0, to=0.05, by=0.001))
set.seed(123)
ridge <- train(expenses ~ ., data=mydata_v3.train, method="glmnet", tuneGrid=ridgeGrid, trControl=TControl, metric="Rsquared")
ridge
#We decided against using the cycle for alpha adjustment, because the results seem to be diverging with every increase
summary(ridge)
plot(ridge,xvar = "lambda")
ridge$bestTune$alpha

prediction.train <- predict(ridge, newdata = mydata_v3.train)
train.rss <- sum((mydata_v3.train$expenses-prediction.train)^2)
ridge.r2 <- 1.0-train.rss/default.train.rss
ridge.r2
prediction.test <- predict(ridge, newdata = mydata_v3.test)
test.rss <- sum((mydata_v3.test$expenses-prediction.test)^2)
ridge.pseudoR2 <- 1.0-test.rss/default.test.rss
ridge.pseudoR2
report <- rbind(report, data.frame(Model="Ridge Regression", R2.Train=ridge.r2, R2.Test=ridge.pseudoR2))



#Kernell Linear+Radial
svlGrid <- expand.grid(C=seq(from=0.01, to=1, by=0.1))
eps <- 0.1
set.seed(123)
svr.linear <- train(expenses ~ ., data=mydata_v3.train, method="svmLinear", tuneGrid=svlGrid, trControl=TControl, metric="Rsquared", epsilon = eps)
svr.linear
plot(svr.linear,xvar = "C")
svlres<-cbind(svr.linear$results$C, svr.linear$results$Rsquared,svr.linear$results$MAE)

prediction.train <- predict(svr.linear, newdata = mydata_v3.train)
train.rss <- sum((mydata_v3.train$expenses-prediction.train)^2)
svrlin.r2 <- 1.0-train.rss/default.train.rss
svrlin.r2
prediction.test <- predict(svr.linear, newdata = mydata_v3.test)
test.rss <- sum((mydata_v3.test$expenses-prediction.test)^2)
svrlin.pseudoR2 <- 1.0-test.rss/default.test.rss
svrlin.pseudoR2
report <- rbind(report, data.frame(Model="Linear SVR", R2.Train=svrlin.r2, R2.Test=svrlin.pseudoR2))
#We can see that results are low - this is non-linear dependency

# radial
svrGrid <- expand.grid(sigma = c(.01, .015, 0.2), C = c(1, 10, 50, 100))
eps <- 0.1
set.seed(123)
svr.rbf <- train(expenses ~ ., data=mydata_v3.train, method="svmRadial", tuneGrid=svrGrid, trControl=TControl, metric="Rsquared", epsilon = eps)
svr.rbf
plot(svr.rbf)
# We use sigma = 0.01 and C = 100
prediction.train <- predict(svr.rbf, newdata = mydata_v3.train)
train.rss <- sum((mydata_v3.train$expenses-prediction.train)^2)
svrrbf.r2 <- 1.0-train.rss/default.train.rss
svrrbf.r2
prediction.test <- predict(svr.rbf, newdata = mydata_v3.test)
test.rss <- sum((mydata_v3.test$expenses-prediction.test)^2)
svrrbf.pseudoR2 <- 1.0-test.rss/default.test.rss
svrrbf.pseudoR2
report <- rbind(report, data.frame(Model="Radial SVR", R2.Train=svrrbf.r2, R2.Test=svrrbf.pseudoR2))






#NN type 1 -(2 hyperpar: size + decay) - simple 1-hidden layer neural network
nnGrid <- expand.grid(size=1:8, decay=seq(from=0.1, to=0.5, by=0.1))
set.seed(123)
nnmodel <- train(expenses ~ ., data=mydata_v3.train, method="nnet", tuneGrid=nnGrid, trControl=TControl, trace=FALSE, metric="Rsquared")
nnmodel

library(NeuralNetTools)
plotnet(nnmodel)
nnmodel$results
plot(nnmodel)
nnmodel$finalModel
#Using (8)-2-(1) model with decay=0.1

prediction.train <- predict(nnmodel, newdata = mydata_v3.train)
train.rss <- sum((mydata_v3.train$expenses-prediction.train)^2)
nn.r2 <- 1.0-train.rss/default.train.rss
nn.r2

prediction.test <- predict(nnmodel, newdata = mydata_v3.test)
test.rss <- sum((mydata_v3.test$expenses-prediction.test)^2)
nn.pseudoR2 <- 1.0-test.rss/default.test.rss
nn.pseudoR2
report <- rbind(report, data.frame(Model="Neural Network 1", R2.Train=nn.r2, R2.Test=nn.pseudoR2))

#NN type 1.1 - we can adjust the decay: it is between 0.1 and 0.2 (let's use 0.05 as a margin):
nnGrid_1 <- expand.grid(size=1:8, decay=seq(from=0.05, to=0.25, by=0.01))
set.seed(123)
nnmodel_1 <- train(expenses ~ ., data=mydata_v3.train, method="nnet", tuneGrid=nnGrid_1, trControl=TControl, trace=FALSE, metric="Rsquared")
nnmodel_1
#Using (8)-5-(1) model with decay=0.05
plotnet(nnmodel_1)
nnmodel_1$finalModel


prediction.train <- predict(nnmodel_1, newdata = mydata_v3.train)
train.rss <- sum((mydata_v3.train$expenses-prediction.train)^2)
nn.r2 <- 1.0-train.rss/default.train.rss
nn.r2
prediction.test <- predict(nnmodel_1, newdata = mydata_v3.test)
test.rss <- sum((mydata_v3.test$expenses-prediction.test)^2)
nn.pseudoR2 <- 1.0-test.rss/default.test.rss
nn.pseudoR2
report <- rbind(report, data.frame(Model="Neural Network DecAdj", R2.Train=nn.r2, R2.Test=nn.pseudoR2))
#Model is better!




#NN type 2 - Multi-layered network
nn2Grid <- expand.grid(layer1=c(1,3,5,7,8), layer2=c(0,1,3,5,7,8), layer3=c(0,1,3,5,7,8))
set.seed(123)
nnmodel2 <- train(expenses ~ ., data=mydata_v3.train, method="neuralnet", tuneGrid=nn2Grid, trControl=TControl, metric="Rsquared")
nnmodel2
plot(nnmodel2$finalModel)
nnmodel2$bestTune
nnmodel2$finalModel
#Using (8)-3-3-7-(1) model
prediction.train <- predict(nnmodel2, newdata = mydata_v3.train)
train.rss <- sum((mydata_v3.train$expenses-prediction.train)^2)
nn.r2 <- 1.0-train.rss/default.train.rss
nn.r2
prediction.test <- predict(nnmodel2, newdata = mydata_v3.test)
test.rss <- sum((mydata_v3.test$expenses-prediction.test)^2)
nn.pseudoR2 <- 1.0-test.rss/default.test.rss
nn.pseudoR2
report <- rbind(report, data.frame(Model="Neural Network 2", R2.Train=nn.r2, R2.Test=nn.pseudoR2))





#RF (non-optimized)
set.seed(123)
rfmodel <- train(expenses ~ ., data=mydata_v3.train, method="rf", trControl=TControl, metric="Rsquared")
rfmodel

summary(rfmodel)
plot(rfmodel)

prediction.train <- predict(rfmodel, newdata = mydata_v3.train)
train.rss <- sum((mydata_v3.train$expenses-prediction.train)^2)
rf.r2 <- 1.0-train.rss/default.train.rss
rf.r2
prediction.test <- predict(rfmodel, newdata = mydata_v3.test)
test.rss <- sum((mydata_v3.test$expenses-prediction.test)^2)
rf.pseudoR2 <- 1.0-test.rss/default.test.rss
rf.pseudoR2
report <- rbind(report, data.frame(Model="Random Forest", R2.Train=rf.r2, R2.Test=rf.pseudoR2))
#Model is overfit: we need to optimize maxnodes
#Mtry is 5

#Optimized RF 
#ZOOMING IN:Choosing initial maxnode step 50
for (maxnodes in c(10,60,110,160,210)) {
  set.seed(123)
  rfmodel <- train(expenses ~ ., data=mydata_v3.train, method="rf", metric="Rsquared",
                   trControl=TControl, maxnodes=maxnodes)
  prediction.train <- predict(rfmodel, newdata = mydata_v3.train)
  train.rss <- sum((mydata_v3.train$expenses-prediction.train)^2)
  rf.r2 <- 1.0-train.rss/default.train.rss
  prediction.test <- predict(rfmodel, newdata = mydata_v3.test)
  test.rss <- sum((mydata_v3.test$expenses-prediction.test)^2)
  rf.pseudoR2 <- 1.0-test.rss/default.test.rss
  cat("maxnodes = ", maxnodes, "\n")
  cat(rf.r2, rf.pseudoR2, "\n\n")
}
#110 - max R2 for test: The needed maxnode parameter is located somewhere between 60 and 160
#ZOOMING IN:Step 25
for (maxnodes in c(60,85,110,135,160)) {
  set.seed(123)
  rfmodel <- train(expenses ~ ., data=mydata_v3.train, method="rf", metric="Rsquared",
                   trControl=TControl, maxnodes=maxnodes)
  prediction.train <- predict(rfmodel, newdata = mydata_v3.train)
  train.rss <- sum((mydata_v3.train$expenses-prediction.train)^2)
  rf.r2 <- 1.0-train.rss/default.train.rss
  prediction.test <- predict(rfmodel, newdata = mydata_v3.test)
  test.rss <- sum((mydata_v3.test$expenses-prediction.test)^2)
  rf.pseudoR2 <- 1.0-test.rss/default.test.rss
  cat("maxnodes = ", maxnodes, "\n")
  cat(rf.r2, rf.pseudoR2, "\n\n")
}
#Needed result is between 85 and 135
#ZOOMING IN:Step is somewhere around 12
for (maxnodes in c(85,97,110,122,135)) {
  set.seed(123)
  rfmodel <- train(expenses ~ ., data=mydata_v3.train, method="rf", metric="Rsquared",
                   trControl=TControl, maxnodes=maxnodes)
  prediction.train <- predict(rfmodel, newdata = mydata_v3.train)
  train.rss <- sum((mydata_v3.train$expenses-prediction.train)^2)
  rf.r2 <- 1.0-train.rss/default.train.rss
  prediction.test <- predict(rfmodel, newdata = mydata_v3.test)
  test.rss <- sum((mydata_v3.test$expenses-prediction.test)^2)
  rf.pseudoR2 <- 1.0-test.rss/default.test.rss
  cat("maxnodes = ", maxnodes, "\n")
  cat(rf.r2, rf.pseudoR2, "\n\n")
}

#Needed result is between 97 and 122
#Step is somewhere around 12
#ZOOMING IN: Step is 6
for (maxnodes in c(97,103,110,116,122)) {
  set.seed(123)
  rfmodel <- train(expenses ~ ., data=mydata_v3.train, method="rf", metric="Rsquared",
                   trControl=TControl, maxnodes=maxnodes)
  prediction.train <- predict(rfmodel, newdata = mydata_v3.train)
  train.rss <- sum((mydata_v3.train$expenses-prediction.train)^2)
  rf.r2 <- 1.0-train.rss/default.train.rss
  prediction.test <- predict(rfmodel, newdata = mydata_v3.test)
  test.rss <- sum((mydata_v3.test$expenses-prediction.test)^2)
  rf.pseudoR2 <- 1.0-test.rss/default.test.rss
  cat("maxnodes = ", maxnodes, "\n")
  cat(rf.r2, rf.pseudoR2, "\n\n")
}
#Needed result is between 110 and 122
#ZOOMING IN: Step is 3
for (maxnodes in c(110,113,116,119,122)) {
  set.seed(123)
  rfmodel <- train(expenses ~ ., data=mydata_v3.train, method="rf", metric="Rsquared",
                   trControl=TControl, maxnodes=maxnodes)
  prediction.train <- predict(rfmodel, newdata = mydata_v3.train)
  train.rss <- sum((mydata_v3.train$expenses-prediction.train)^2)
  rf.r2 <- 1.0-train.rss/default.train.rss
  prediction.test <- predict(rfmodel, newdata = mydata_v3.test)
  test.rss <- sum((mydata_v3.test$expenses-prediction.test)^2)
  rf.pseudoR2 <- 1.0-test.rss/default.test.rss
  cat("maxnodes = ", maxnodes, "\n")
  cat(rf.r2, rf.pseudoR2, "\n\n")
}

#Needed result is between 113 and 119
#ZOOMING IN: Step is 3
for (maxnodes in c(113,114,115,116,117,118,119)) {
  set.seed(123)
  rfmodel <- train(expenses ~ ., data=mydata_v3.train, method="rf", metric="Rsquared",
                   trControl=TControl, maxnodes=maxnodes)
  prediction.train <- predict(rfmodel, newdata = mydata_v3.train)
  train.rss <- sum((mydata_v3.train$expenses-prediction.train)^2)
  rf.r2 <- 1.0-train.rss/default.train.rss
  prediction.test <- predict(rfmodel, newdata = mydata_v3.test)
  test.rss <- sum((mydata_v3.test$expenses-prediction.test)^2)
  rf.pseudoR2 <- 1.0-test.rss/default.test.rss
  cat("maxnodes = ", maxnodes, "\n")
  cat(rf.r2, rf.pseudoR2, "\n\n")
}
#Max R2 for test was achieved in maxnodes = 116

set.seed(123)
rfomodel <- train(expenses ~ ., data=mydata_v3.train, method="rf", trControl=TControl, 
                  metric="Rsquared", maxnodes=116)
rfomodel

summary(rfomodel)
rfomodel$finalModel

prediction.train <- predict(rfomodel, newdata = mydata_v3.train)
train.rss <- sum((mydata_v3.train$expenses-prediction.train)^2)
rf.r2 <- 1.0-train.rss/default.train.rss
rf.r2
prediction.test <- predict(rfomodel, newdata = mydata_v3.test)
test.rss <- sum((mydata_v3.test$expenses-prediction.test)^2)
rf.pseudoR2 <- 1.0-test.rss/default.test.rss
rf.pseudoR2
report <- rbind(report, data.frame(Model="Random Forest opt.", R2.Train=rf.r2, R2.Test=rf.pseudoR2))




#KNN
knnGrid <- expand.grid(k=seq(from=2, to=15, by=1))
set.seed(123)
knnmodel <- train(expenses ~ ., data=mydata_v3.train, method="knn", tuneGrid=knnGrid, trControl=TControl, metric="Rsquared")
knnmodel

summary(knnmodel)
plot(knnmodel)
knnmodel$finalModel

prediction.train <- predict(knnmodel, newdata = mydata_v3.train)
train.rss <- sum((mydata_v3.train$expenses-prediction.train)^2)
knn.r2 <- 1.0-train.rss/default.train.rss
knn.r2
prediction.test <- predict(knnmodel, newdata = mydata_v3.test)
test.rss <- sum((mydata_v3.test$expenses-prediction.test)^2)
knn.pseudoR2 <- 1.0-test.rss/default.test.rss
knn.pseudoR2
report <- rbind(report, data.frame(Model="k-NN", R2.Train=knn.r2, R2.Test=knn.pseudoR2))







#************
#Final Report
#************

report

results <- resamples(list(OLS=ols, Ridge=ridge, SVM.L=svr.linear, SVM.R=svr.rbf, 
                          NeuNet=nnmodel,NeuNetAdj=nnmodel_1, NeuNet2=nnmodel2, RFor=rfmodel, RFor.o=rfomodel, KNN=knnmodel))
summary(results)
dotplot(results)

write.csv(report, "report.csv", row.names=FALSE)
write.csv(corr.matrix, "correlation.csv", row.names=FALSE)
write.csv(svlGrid, "svlgrid.csv", row.names=FALSE)
write.csv(svrGrid, "svrgrid.csv", row.names=FALSE)
write.csv(svr.linear$results, "svlres.csv", row.names=FALSE)
write.csv(nnmodel$results, "nnmodres.csv", row.names=FALSE)
write.csv(nnGrid, "nngrid.csv", row.names=FALSE)
write.csv(rfmodel$results, "rfres.csv", row.names=FALSE)
write.csv(knnmodel$results, "knnres.csv", row.names=FALSE)
