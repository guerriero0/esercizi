library(MASS)
library(neuralnet)
library(fansi)
library(tidyr)
library(dplyr)
library(randomForest)
library(rpart)
library(rpart.plot)


healthcare.dataset.stroke.data <- read.csv("~/Downloads/healthcare-dataset-stroke-data.csv")
dati<-healthcare.dataset.stroke.data
attach(dati)
str(dati)
dati$stroke<-as.factor(dati$stroke)
dati$gender<-as.factor(dati$gender)
dati$ever_married<-as.factor(dati$ever_married)
dati$work_type<-as.factor(dati$work_type)
dati$Residence_type<-as.factor(dati$Residence_type)
dati$smoking_status<-as.factor(dati$smoking_status)
dati$hypertension<-as.factor(dati$hypertension)
dati$heart_disease<-as.factor(dati$heart_disease)



apply(dati,2,function(x) sum(is.na(x)))
dati<-dati %>% drop_na(bmi)
dati<-dati %>%
  select(!(id)) 

index<-sample(1:nrow(dati),round(0.8*nrow(dati)))

training<-dati[index,]
test<-dati[-index,]

datist<-dati %>% 
  mutate(bmi=scale(bmi,center=min(bmi),scale=max(bmi)-min(bmi)),age=scale(age,center=min(age),
  scale=max(age)-min(age)),
  avg_glucose_level=scale(avg_glucose_level,center=min(avg_glucose_level),scale=max(avg_glucose_level)-min(avg_glucose_level)))


datist<-model.matrix(~stroke+gender+age+hypertension+heart_disease+ever_married+work_type+
                       Residence_type+avg_glucose_level+bmi+smoking_status,data=datist)
datist<-as.data.frame(datist)
trainst<-datist[index,]
testst<-datist[-index,]
nomi<-names(trainst) 
form<-as.formula(paste("stroke~",paste(nomi[!nomi %in% c("stroke","gender","ever_married","Residence_type","work_type","smoking_status")],
                                     collapse="+")))

form
retneur<-neuralnet(stroke1~age+hypertension+heart_disease+
       bmi+avg_glucose_level+genderMale+genderOther
  ,data=trainst,hidden=c(3),linear.output = FALSE,threshold = 0.05,algorithm="backprop",learningrate = 0.1)

plot(retneur)
prev.retneur<-neuralnet::compute(retneur,testst)

pred<-ifelse(prev.retneur$net.result>0.5,1,0)
conf<-table(pred,testst$stroke1)
sum(diag(conf))/sum(conf)

library(pROC)
pred<-as.factor(pred)
curvaROC<-plot(roc(pred,testst$stroke1,print.AUC=TRUE))
curvaROC

tree<-rpart(stroke~age+hypertension+heart_disease+bmi+avg_glucose_level,data=trainst,method="class")
rpart.plot(tree,type=5,fallen.leaves = TRUE)


library(datasets)
data()
