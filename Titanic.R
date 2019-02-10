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
train =  train[,-c("Cabin")]


# replacing NA in Embarked to S as only 2 are missing
table(train$Embarked)
train$Embarked[train$Embarked ==""] <- 'S'

## imputing missing age by predicting the age based on variables Pclass,Sex,title,SibSp,Parch,fare
ageNA <- lm(Age~factor(Pclass)+factor(Sex)+factor(Embarked)+SibSp+Parch + Fare, data = train[!is.na(train$Age)])
summary(ageNA)
train$Age[is.na(train$Age)] <- predict(ageNA , train[is.na(train$Age)])

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

## child or adult based on age
train$Child[train$Age < 18] <- 'Child'
train$Child[train$Age >= 18] <- 'Adult'

ggplot(data = train , aes(x=Child , fill = factor(Survived))) + 
  geom_bar(stat = "count", position = "dodge") + facet_grid(.~Sex)


train$Pclass <- factor(train$Pclass)
train$Sex <- factor(train$Sex)
train$Fsize <- factor(train$Fsize)
train$Child <- factor(train$Child)
train$Embarked <- factor(train$Embarked)

set.seed(123) 

#creating indices
trainIndex <- createDataPartition(train$Survived,p=0.75,list=FALSE)

#splitting data into training/testing data using the trainIndex object
train_titanic <- train[trainIndex,] #training data (75% of data)
test_titanic <- train[-trainIndex,] #testing data (25% of data)

## Logistic Regression
model1 <- glm(Survived ~ Pclass+Sex+Fsize+Child+Fare+Embarked ,data = train_titanic, family = binomial(link = "logit") )
summary(model1)

## Logistic regression validation
predglm <- predict(model1, test_titanic[,-c("Survived")] , type ="response" )
logit_survived = as.numeric(predglm >= 0.5)
table(logit_survived)
confusionMatrix(test_titanic$Survived ,as.factor(logit_survived))



#Random FOrest

rf_model <- randomForest(Survived ~ Pclass+Sex+Fsize+Child+Fare+Embarked, data = train_titanic)
rf_pred <- predict(rf_model , test_titanic[,-c("Survived")] , type = "response")
confusionMatrix(test_titanic$Survived ,rf_pred)

## Random Forest validation
sapply(test, function(x) {ifelse(sum(is.na(x))!=0 , round(sum(is.na(x))*100/nrow(test),2) , round(sum(x=="")*100/nrow(test),2))})
test$Age[is.na(test$Age)] <- predict(ageNA , test[is.na(test$Age)])
test$Fsize <- test$SibSp + test$Parch + 1
test$Child[test$Age < 18] <- 'Child'
test$Child[test$Age >= 18] <- 'Adult'
test <- test[,-c("Cabin")]

test$Pclass <- factor(test$Pclass)
test$Sex <- factor(test$Sex)
test$Fsize <- factor(test$Fsize)
test$Child <- factor(test$Child)
test$Embarked <- factor(test$Embarked)

test_glm = predict(model1,newdata = test , type = "response")
test_survived <- as.numeric(test_glm >= 0.5)
#test <- test[,-c("Survived")]
test$Survived<-predict(rf_model , test_titanic[,-c("Survived")] , type = "response")
fwrite(test[,c("PassengerId","Survived")] , file = "gender_submission.csv" , append = FALSE) 

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

