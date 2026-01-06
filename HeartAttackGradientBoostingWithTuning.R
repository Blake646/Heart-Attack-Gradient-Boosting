library(xgboost)
library(caret)
library(gbm)
library(pROC)

heart_attack <- read.csv('heart_attack.csv')
#Train Test
idx = createDataPartition(heart_attack$heart_attack, p=0.75, list = F)
heart.train = heart_attack[idx,]
heart.test = heart_attack[-idx,]



#Boost no tuning
boost.model = gbm(heart_attack~., data=heart.train,
                  distribution = "bernoulli", n.trees = 500,
                  interaction.depth = 4)  

summary(boost.model)

plot(boost.model, i="systolic_bp") #exploration


#Accuracy
yhat <- predict(boost.model, heart.test, type = "response")
yhat <- ifelse(yhat > 0.5, 1, 0)
yhat <- as.factor(yhat)
confusionMatrix(yhat, as.factor(heart.test$heart_attack)) 


#for auc
yhat_probs <- predict(boost.model, heart.test, n.trees = 500, type = "response")
roc_obj <- roc(response = heart.test$heart_attack,
               predictor = yhat_probs)

auc_value <- auc(roc_obj)
print(auc_value)

#---------------------------------

heart.train$heart_attack <- as.factor(heart.train$heart_attack)
heart.test$heart_attack  <- as.factor(heart.test$heart_attack)

#Tuning
tunegrid = expand.grid(interaction.depth = 1:5, 
                       n.trees = seq(100,1000,100), 
                       shrinkage = c(0.05,0.4,.05),
                       n.minobsinnode = 5:10)
gbm_tuning = train(heart_attack~.,  #takes a minute to run
                   data=heart.train, 
                   method='gbm', 
                   tuneGrid = tunegrid,
                   verbose=FALSE,
                   trControl=trainControl(method="cv", number=10))
gbm_tuning


plot(gbm_tuning)


#Accuracy
yhat1 = predict(gbm_tuning, heart.test)
cm <- confusionMatrix(yhat1, as.factor(heart.test$heart_attack))
cm$table
yhat_probs <- predict(gbm_tuning, heart.test, type = "prob")
roc_obj <- roc(response = heart.test$heart_attack,
               predictor = yhat_probs[,"1"])  
auc_value <- auc(roc_obj)
auc_value
plot(roc_obj, print.auc=T, main='ROC Curve (Tuned Model)')
colnames(heart_attack)



#Changing shrinkage
#----------------------------


heart.train$heart_attack <- as.factor(heart.train$heart_attack)
tunegrid = expand.grid(interaction.depth = 1:5, 
                       n.trees = seq(100,1000,100), 
                       shrinkage = c(0.01,0.2,.02),
                       n.minobsinnode = 5:10)
gbm_tuning1 = train(heart_attack~., 
                    data=heart.train, 
                    method='gbm', 
                    tuneGrid = tunegrid,
                    verbose=FALSE,
                    trControl=trainControl(method="cv", number=10))
summary(gbm_tuning1)
#Accuracy
yhat1.1 = predict(gbm_tuning1, heart.test)
confusionMatrix(yhat1.1, as.factor(heart.test$heart_attack))