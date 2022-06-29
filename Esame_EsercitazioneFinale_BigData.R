library(tidyverse)
library(caret)
library(FactoMineR)

#ACP,variabili quantitative,ML non supervisionato cioè senza una variabile di risposta
data("decathlon")
head(decathlon)
ris.pca<-PCA(decathlon[,1:10],scale.unit=TRUE)

ris.pca<-PCA(decathlon,scale.unit=TRUE, quanti.sup=c(11:12),quali.sup=13)

plot.PCA(ris.pca,axes=c(3,4),choir="var")
ris.pca
ris.pca$eig

ris.clus<-HCPC(ris.pca)


#K means
x<-matrix(rnorm(50*2),ncol=2)
x[1:25,1]=x[1:25,1]+3
x[1:25,2]=x[1:25,2]-4
plot(x)

ris.km<-kmeans(x,2,nstart=20)
ris.km

ris.km$cluster
plot(x,col=ris.km$cluster+1)

ris.km<-kmeans(decathlon[,c(1,6)],3,nstart =20 )
plot(decathlon[,c(1,6)],col=ris.km$cluster+1)

####
library(tidyverse)
data("swiss")
head(swiss)

train.control<-trainControl(method="LOOCV")
train.control<-trainControl(method="cv",number=10,classProbs = TRUE)
train.control<-trainControl(method="repeatedcv",number=10,repeats = 3)
#ripete la procedura di CV 3 volte per pulire eventuali differenze tra le Cross validation


model1<-train(Fertility~.,data=swiss,method="lm",trControl=train.control)
model2<-train(Fertility~.,data=swiss,method="lasso",trControl=train.control)

data(iris)
train.control<-trainControl(method="repeatedcv",number=5,repeats = 2)
model3<-train(Species~.,data=iris,method="multinom",trControl=train.control)

model4<-train(Species~.,data=iris,method="qda",trControl=train.control)

model5<-train(Species~.,data=iris,method="svmLinear",trControl=train.control)

model6<-train(Species~.,data=iris,method="svmRadial",trControl=train.control)

getModelInfo("qda")
help("getModelInfo")

#Alb.di regr.
library(rpart)
library(rpart.plot)
data(kyphosis)
head(kyphosis)

set.seed(123)
ind<-sample(1:nrow(kyphosis),round(0.8*nrow(kyphosis)))
base<-kyphosis[ind,]
test<-kyphosis[-ind,]

#Parto dalla regr. logistica

fit.logit<-glm(factor(Kyphosis)~.,data=base,family="binomial")
summary(fit.logit)

prev.logit<-predict(fit.logit,newdata=test,type="response")
prev.logit
pred.logit<-ifelse(prev.logit<0.5,"absent","present")
mat.conf1<-table(test$Kyphosis,pred.logit)

sum(diag(mat.conf1))/sum(mat.conf1)

#Albero di regressione

fit.tree<-rpart(Kyphosis~.,data=base,method="class")

pred.tree<-predict(fit.tree,test,type="class")

mat.conf2<-table(test$Kyphosis,pred.tree)
sum(diag(mat.conf2))/sum(mat.conf2)

prev.tree<-predict(fit.tree,test,type="prob")
auc.tree<-auc(test$Kyphosis,prev.tree[,2])
auc.tree

#Discriminant
library(MASS)
fit.lda<-lda(Kyphosis~.,data=base)
pred.lda<-predict(fit.lda,newdata=test)
mat.conf3<-table(test$Kyphosis,pred.lda$class)
sum(diag(mat.conf3))/sum(mat.conf3)

library(pROC)
auc.lda<-auc(test$Kyphosis,pred.lda$posterior[,2])
auc.lda

#MSE per confrontare modelli per var.quant.

library(datasets)
data()

##ESERCIZIO COLLEGE
library(ISLR)
data("College")
str(College)

set.seed(2467)
indice<-sample(1:nrow(College),0.75*nrow(College))
train<-College[indice,]
test<-College[-indice,]

mod.lm<-lm(Expend~.-Grad.Rate,data=train)
summary(mod.lm)
prev<-predict(mod.lm,newdata=test)
MSE<-mean((test$Expend-prev)^2)
sqrt(MSE)

library(caret)
library(mlbench)
set.seed(2467)
train.control<-trainControl(method="repeatedcv",number=10,repeats = 3)
model.lm<-train(Expend~.-Grad.Rate,data=College,method="lm",trControl=train.control)
print(model.lm)

library(mgcv)
mod.loess<-gam(Expend~.-Grad.Rate,data=train,sp=0.1)
summary(mod.loess)

set.seed(2467)
train.control<-trainControl(method="cv",number=10)
model.loess<-train(Expend~.-Grad.Rate+Private,data=College,method="gam",trControl=train.control)
print(model.loess)

library(rpart)
library(rpart.plot)

mod.alb<-rpart(Expend~.-Grad.Rate,data=train,method="anova")
rpart.plot(mod.alb,digits=3,5,fallen.leaves = TRUE)
prev<-predict(mod.alb,newdata=test)
MSE<-mean((test$Expend-prev)^2)
sqrt(MSE)

set.seed(2467)
train.control<-trainControl(method="cv",number=10)
model.albregr<-train(Expend~.-Grad.Rate,data=College,method="rpart",trControl=train.control)
print(model.albregr)


a<-as.factor(c("A","B","C"))
set.seed(2467)
b<-sample(a,777,replace=TRUE)
b
table(b)
