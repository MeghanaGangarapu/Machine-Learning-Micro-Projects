## Titanic Survival Prediction

setwd("D:/MSBA/Projects/Titanic/")
rm(list=ls())

library(data.table)
library(tidyverse)
library(dplyr)
library(stringr)
library(caret)
library(randomForest)
library(e1071)
library(rpart)

train <- fread("train.csv") %>% data.table()
test <-  fread("test.csv") %>% data.table()

str(train)
str(test)

## PClass VS Survived bar graph
ggplot(train, aes(x = factor(Pclass), fill = factor(Survived))) +
  geom_bar(width = 0.5, position="dodge") +
  xlab("Pclass") +
  ylab("Total Count") +
  labs(fill = "Survived")

## Sex VS Survived bar graph
ggplot(train, aes(x = factor(Sex), fill = factor(Survived))) +
  geom_bar(width = 0.5, position="dodge") +
  xlab("Sex") +
  ylab("Total Count") +
  labs(fill = "Survived")

## Age VS Survived 

ggplot(subset(train,!is.na(Age)), aes(x = Age, fill = factor(Survived))) +
 geom_histogram(bins = 30) +
  xlab("Age") +
  ylab("Total Count") 

## Age Vs Sex
ggplot(subset(train,!is.na(Age)), aes(Age, fill = factor(Survived))) + 
  geom_histogram(bins=30) + 
  xlab("Age") +
  ylab("Count") +
  facet_grid(.~Sex)+
  scale_fill_discrete(name = "Survived") +
  ggtitle("Age vs Sex vs Survived")

# Pclass vs Survived
ggplot(train, aes(Pclass, fill = factor(Survived))) + 
  geom_bar(stat = "count",position = "dodge")+
  xlab("Pclass") +
  facet_grid(.~Sex)+
  ylab("Count") +
  scale_fill_discrete(name = "Survived") + 
  ggtitle("Pclass vs Sex vs Survived")

# Pclass vs Sex vs Age vs Survived
ggplot(train, aes(x = Age, y = Sex)) + 
  geom_jitter(aes(colour = factor(Survived))) + 
  theme(legend.title = element_blank())+
  facet_wrap(~Pclass) + 
  labs(x = "Age", y = "Sex", title = "Pclass vs Sex vs Age vs Survived")+
  scale_fill_discrete(name = "Survived") + 
  scale_x_continuous(name="Age",limits=c(0, 81)) 

table(train$SibSp)
ggplot(train, aes(x = SibSp, fill = factor(Survived))) +
  geom_histogram(binwidth=0.5, position="dodge") +
  xlab("Number of Siblings/Spouses") +
  ylab("Total Count") +
  labs(fill = "Survived")

## Fare Vs Pclass

ggplot(train, aes(x=Fare , y = Pclass)) + 
  geom_jitter(aes(color = factor(Survived))) + 
  scale_x_continuous(limits = c(0,300), breaks = seq(0,300,40))



table(train$Parch)
ggplot(train, aes(x = Parch, fill = factor(Survived))) +
  geom_histogram(binwidth=0.5, position="dodge") +
  facet_grid(.~Sex)+
  xlab("Number of Parents/Children") +
  ylab("Total Count") +
  labs(fill = "Survived")

# finding percentage of missing values in each column
sapply(train, function(x) {ifelse(sum(is.na(x))!=0 , round(sum(is.na(x))*100/nrow(train),2) , round(sum(x=="")*100/nrow(train),2))})

# Ignoring attribute Cabin since 77% of data is missing

# replacing NA in Embarked to S as only 2 are missing
table(train$Embarked)
train$Embarked[train$Embarked ==""] <- 'S'

train$Fsize <- train$SibSp + train$Parch + 1

ggplot(train , aes(x=Fsize , fill = factor(Survived)))+geom_bar(stat = "count" , position = "dodge")+
  scale_x_continuous(breaks = seq(0,max(train$Fsize),1)) +
  ggtitle("Family Size vs Survived")

## Embarked vs Pclass vs Survived
ggplot(train, aes(Embarked, fill = factor(Survived))) + 
  geom_bar(stat = "count",position = "dodge")+
  xlab("Pclass") +
  ylab("Count") +
  facet_grid(.~Pclass) + 
  scale_fill_discrete(name = "Survived") + 
  ggtitle("Embarked vs Pclass vs Survived")

train$Title <- NA
train$Title <- sapply(train$Name , function(x) str_trim(str_split(x,"[,.]")[[1]][2],side ="both"))


unique(train$Title)
train$Title[train$Title%in%c("Mme")] <- "Mrs"
train$Title[train$Title%in%c("Mlle","Ms")] <- "Miss"
officer <- c('Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev')
royalty <- c('Dona', 'Lady', 'the Countess','Sir', 'Jonkheer')
train$Title[train$Title %in% royalty]  <- 'Royalty'
train$Title[train$Title %in% officer]  <- 'Officer'

ggplot(train , aes(x=Title , fill = factor(Survived))) + 
  geom_histogram(bins = 6 , stat = "count") +
  ggtitle("Title By Survived")


## imputing missing age by predicting the age based on variables Pclass,Sex,title,SibSp,Parch,fare,Title
agefit <- rpart(Age~Pclass+Sex+Embarked+SibSp+Parch+Fsize + Fare + Title,
                data = train[!is.na(train$Age)] , method = "anova")
summary(agefit)
train$Age[is.na(train$Age)] <- predict(agefit , train[is.na(train$Age)])


## child or adult based on age
train$Child[train$Age < 18] <- 'Child'
train$Child[train$Age >= 18] <- 'Adult'


ggplot(data = train , aes(x=Child , fill = factor(Survived))) + 
  geom_bar(stat = "count", position = "dodge") + facet_grid(.~Sex)



train$Pclass <- factor(train$Pclass)
train$Sex <- as.integer(train$Sex=="male")
train$Child <- as.integer(train$Child=="Child")
train$Embarked <- factor(train$Embarked)
train$Title <- as.factor(train$Title)


set.seed(123) 

#creating indices
trainIndex <- createDataPartition(train$Survived,p=0.75,list=FALSE)

#splitting data into training/testing data using the trainIndex object
train_titanic <- train[trainIndex,] #training data (75% of data)
test_titanic <- train[-trainIndex,] #testing data (25% of data)

## Logistic Regression
model_glm <- glm(Survived ~ Pclass+Sex+Fsize+Child+Fare+Embarked+Title ,data = train_titanic, family = binomial(link = "logit") )
summary(model_glm)
predglm <- predict(model_glm, test_titanic , type ="response" )
logit_survived = as.numeric(predglm >= 0.5)
table(logit_survived)
confusionMatrix(as.factor(test_titanic$Survived) ,as.factor(logit_survived))
print(RMSE(test_titanic$Survived,logit_survived))

#Random FOrest
set.seed(123)
cvCtrl = trainControl(method = "repeatedcv", number = 5, repeats = 5) 
mtry <- round(sqrt(ncol(train_titanic) -1))

RFgrid <- expand.grid(
 mtry = mtry)

rf_model <-train(Survived ~ Pclass+Sex+Fsize+Child+Fare+Embarked+Title, data=train_titanic,
                 tuneGrid = RFgrid,
                 method = "rf" ,
                 trControl = cvCtrl,
                 preProcess = c("center", "scale"))

rf_pred <- predict(rf_model , test_titanic )
rf_pred = as.numeric(rf_pred >= 0.5)
confusionMatrix(as.factor(test_titanic$Survived) ,as.factor(rf_pred))
print(RMSE(test_titanic$Survived,rf_pred))


## SVM
# Set up the 5-fold CV
fitControl <- caret::trainControl(method = "repeatedcv",
                                  number = 5,
                                  repeats = 5)

# Define ranges for the two parameters
C_range =     sapply(seq(-1,3,0.0125), function(x) 10^x)
sigma_range = sapply(seq(-3,1,0.0125), function(x) 10^x)

# Create the grid of parameters
fitGrid <- expand.grid(C= C_range,
                       sigma = sigma_range)


Rsvm <- caret::train(Survived ~ Pclass+Sex+Fsize+Child+Fare+Embarked+Title, data=train_titanic,
                     method = "svmLinear",
                     trControl = fitControl,
                     preProcess = c("center", "scale"))
svm_pred <- predict(Rsvm , test_titanic )
svm_pred = as.numeric(svm_pred >= 0.5)
confusionMatrix(as.factor(test_titanic$Survived) ,as.factor(svm_pred))
print(RMSE(test_titanic$Survived,svm_pred))

#gradient boosting
fitControl <- trainControl(method = 'repeatedcv',
                           number = 5,
                           repeats = 5)
# for caret, there are only four tuning parameters below.

# tune n.trees
newGrid <- expand.grid(n.trees = c(50, 100, 200, 300), 
                       interaction.depth = c(6),
                       shrinkage = 0.01,
                       n.minobsinnode = 10
)
fit_gbm <- train(Survived ~ Pclass+Sex+Fsize+Child+Fare+Embarked+Title, data=train_titanic, 
                 method = 'gbm', 
                 trControl = fitControl,
                 tuneGrid =  newGrid,
                 bag.fraction = 0.5,
                 verbose = FALSE,
                 preProcess = c("center", "scale"))
fit_gbm$bestTune
gbm_pred <- predict(fit_gbm , test_titanic )
gbm_pred = as.numeric(gbm_pred >= 0.5)
confusionMatrix(as.factor(test_titanic$Survived) ,as.factor(gbm_pred))
print(RMSE(test_titanic$Survived,gbm_pred))



#test Data
test_glm = predict(model_glm,newdata = test , type = "response")
test_survived <- as.numeric(test_glm >= 0.5)

test$Survived<-predict(rf_model , test)
test$Survived <-as.integer(test$Survived>=0.5)
fwrite(test[,c("PassengerId","Survived")] , file = "gender_submission.csv") 




