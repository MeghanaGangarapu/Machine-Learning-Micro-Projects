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

train <- fread("train.csv")%>% data.table()
test <-  fread("test.csv") %>% data.table()
test$Survived <- NA

combi = rbind(train, test)
ntrain <- nrow(train)
str(train)
str(test)

## PClass VS Survived bar graph
ggplot(combi[1:ntrain,], aes(x = factor(Pclass), fill = factor(Survived))) +
  geom_bar(width = 0.5, position="dodge") +
  xlab("Pclass") +
  ylab("Total Count") +
  labs(fill = "Survived")

## Sex VS Survived bar graph
ggplot(combi[1:ntrain,], aes(x = factor(Sex), fill = factor(Survived))) +
  geom_bar(width = 0.5, position="dodge") +
  xlab("Sex") +
  ylab("Total Count") +
  labs(fill = "Survived")

## Age VS Survived 

ggplot(subset(combi[1:ntrain,],!is.na(Age)), aes(x = Age, fill = factor(Survived))) +
 geom_histogram(bins = 30) +
  xlab("Age") +
  ylab("Total Count") 

## Age Vs Sex
ggplot(subset(combi[1:ntrain,],!is.na(Age)), aes(Age, fill = factor(Survived))) + 
  geom_histogram(bins=30) + 
  xlab("Age") +
  ylab("Count") +
  facet_grid(.~Sex)+
  scale_fill_discrete(name = "Survived") +
  ggtitle("Age vs Sex vs Survived")

# Pclass vs Survived
ggplot(combi[1:ntrain,], aes(Pclass, fill = factor(Survived))) + 
  geom_bar(stat = "count",position = "dodge")+
  xlab("Pclass") +
  facet_grid(.~Sex)+
  ylab("Count") +
  scale_fill_discrete(name = "Survived") + 
  ggtitle("Pclass vs Sex vs Survived")

# Pclass vs Sex vs Age vs Survived
ggplot(combi[1:ntrain,], aes(x = Age, y = Sex)) + 
  geom_jitter(aes(colour = factor(Survived))) + 
  theme(legend.title = element_blank())+
  facet_wrap(~Pclass) + 
  labs(x = "Age", y = "Sex", title = "Pclass vs Sex vs Age vs Survived")+
  scale_fill_discrete(name = "Survived") + 
  scale_x_continuous(name="Age",limits=c(0, 81)) 

table(combi[1:ntrain,]$SibSp)
ggplot(combi[1:ntrain,], aes(x = SibSp, fill = factor(Survived))) +
  geom_histogram(binwidth=0.5, position="dodge") +
  xlab("Number of Siblings/Spouses") +
  ylab("Total Count") +
  labs(fill = "Survived")

## Fare Vs Pclass

ggplot(combi[1:ntrain,], aes(x=Fare , y = Pclass)) + 
  geom_jitter(aes(color = factor(Survived))) 


table(combi[1:ntrain,]$Parch)
ggplot(combi[1:ntrain,], aes(x = Parch, fill = factor(Survived))) +
  geom_histogram(binwidth=0.5, position="dodge" , stat = "count") +
  facet_grid(.~Sex)+
  xlab("Number of Parents/Children") +
  ylab("Total Count") +
  labs(fill = "Survived")

# finding percentage of missing values in each column
sapply(combi, function(x) {ifelse(sum(is.na(x))!=0 , round(sum(is.na(x))*100/nrow(combi),2) , round(sum(x=="")*100/nrow(combi),2))})

# Ignoring attribute Cabin since 77% of data is missing

# replacing NA in Embarked to S as only 2 are missing
table(combi$Embarked)
combi$Embarked[combi$Embarked ==""] <- 'S'
combi$Embarked[combi$Embarked ==""] <- 'S'

combi$Fsize <- combi$SibSp + combi$Parch + 1

ggplot(combi[1:ntrain,] , aes(x=Fsize , fill = factor(Survived)))+geom_bar(stat = "count" , position = "dodge")+
  scale_x_continuous(breaks = seq(0,max(combi[1:ntrain,]$Fsize),1)) +
  ggtitle("Family Size vs Survived")

table(combi$Pclass)

## Embarked vs Pclass vs Survived
ggplot(combi[1:ntrain,], aes(Embarked, fill = factor(Survived))) + 
  geom_bar(stat = "count",position = "dodge")+
  xlab("Pclass") +
  ylab("Count") +
  facet_grid(.~Pclass) + 
  scale_fill_discrete(name = "Survived") + 
  ggtitle("Embarked vs Pclass vs Survived")

combi$Title <- NA
combi$Title <- sapply(combi$Name , function(x) str_trim(str_split(x,"[,.]")[[1]][2],side ="both"))


unique(combi$Title)
combi$Title[combi$Title%in%c("Mme")] <- "Mrs"
combi$Title[combi$Title%in%c("Mlle","Ms")] <- "Miss"
officer <- c('Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev')
royalty <- c('Dona', 'Lady', 'the Countess','Sir', 'Jonkheer')
combi$Title[combi$Title %in% royalty]  <- 'Royalty'
combi$Title[combi$Title %in% officer]  <- 'Officer'

ggplot(combi[1:ntrain,] , aes(x=Title , fill = factor(Survived))) + 
  geom_histogram(bins = 6 , stat = "count") +
  ggtitle("Title By Survived")+
  theme(axis.text.x=element_text(angle=60, hjust=1))





## imputing missing age by predicting the age based on variables Pclass,Sex,title,SibSp,Parch,fare,Title
agefit <- rpart(Age~Pclass+Sex+Embarked+SibSp+Parch+Fsize + Fare + Title,
                data = combi[!is.na(combi$Age),] , method = "anova")
summary(agefit)
combi$Age[is.na(combi$Age)] <- predict(agefit , combi[is.na(combi$Age),])


## child or adult based on age
combi$Child[combi$Age < 18] <- 'Child'
combi$Child[combi$Age >= 18] <- 'Adult'


ggplot(data = combi[1:ntrain,] , aes(x=Child , fill = factor(Survived))) + 
  geom_bar(stat = "count", position = "dodge") + facet_grid(.~Sex)



combi$Pclass <- factor(combi$Pclass)
combi$Sex <- as.integer(combi$Sex=="male")
combi$Child <- as.integer(combi$Child=="Child")
combi$Embarked <- factor(combi$Embarked)
combi$Title <- as.factor(combi$Title)

sapply(combi, function(x) {ifelse(sum(is.na(x))!=0 , round(sum(is.na(x))*100/nrow(combi),2) , round(sum(x=="")*100/nrow(combi),2))})
set.seed(123)

#creating indices
trainIndex <- createDataPartition(combi[1:ntrain,]$Survived,p=1,list=FALSE)

#splitting data into training/testing data using the trainIndex object
train_titanic <- combi[trainIndex,] 
test_titanic <- combi[-trainIndex,] 

#creating indices to split train into train and validation
Index2 <- createDataPartition(train_titanic$Survived,p=0.8,list=FALSE)
train <- train_titanic[Index2,] 
validation <- train_titanic[-Index2,] 

## Logistic Regression
model_glm <- glm(Survived ~ Pclass+Sex+Fsize+Child+Fare+Embarked+Title ,data = train, family = binomial(link = "logit") )
summary(model_glm)
# validation
predglm <- predict(model_glm, validation , type ="response" )
logit_survived = as.numeric(predglm >= 0.5)
table(logit_survived)
confusionMatrix(as.factor(validation$Survived) ,as.factor(logit_survived))
# predicting Test
test_glm <- predict(model_glm, test_titanic , type ="response" )
print(RMSE(validation$Survived,logit_survived))
#Random FOrest
set.seed(123)
cvCtrl = trainControl(method = "repeatedcv", number = 5, repeats = 5) 
mtry <- round(sqrt(ncol(train) -1))

RFgrid <- expand.grid(
 mtry = mtry)

rf_model <-train(Survived ~ Pclass+Sex+Fsize+Child+Fare+Embarked+Title, data=train,
                 tuneGrid = RFgrid,
                 method = "rf" ,
                 trControl = cvCtrl,
                 preProcess = c("center", "scale"))

rf_pred <- predict(rf_model , validation )
rf_pred = as.numeric(rf_pred >= 0.5)
confusionMatrix(as.factor(validation$Survived) ,as.factor(rf_pred))
print(RMSE(validation$Survived,rf_pred))


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


Rsvm <- caret::train(Survived ~ Pclass+Sex+Fsize+Child+Fare+Embarked+Title, data=train,
                     method = "svmLinear",
                     trControl = fitControl,
                     preProcess = c("center", "scale"))
svm_pred <- predict(Rsvm , validation )
svm_pred = as.numeric(svm_pred >= 0.5)
confusionMatrix(as.factor(validation$Survived) ,as.factor(svm_pred))
print(RMSE(validation$Survived,svm_pred))

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
fit_gbm <- train(Survived ~ Pclass+Sex+Fsize+Child+Fare+Embarked+Title, data=train, 
                 method = 'gbm', 
                 trControl = fitControl,
                 tuneGrid =  newGrid,
                 bag.fraction = 0.5,
                 verbose = FALSE,
                 preProcess = c("center", "scale"))
fit_gbm$bestTune
gbm_pred <- predict(fit_gbm , validation )
gbm_pred = as.numeric(gbm_pred >= 0.5)
confusionMatrix(as.factor(validation$Survived) ,as.factor(gbm_pred))
print(RMSE(validation$Survived,gbm_pred))



