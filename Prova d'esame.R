#Esercizio 1

library(ISLR)
data("Caravan")
attach(Caravan)
#Logistic

set.seed(2468)
mod<-glm(Purchase~.,data=Caravan,family="binomial")
summary(mod)

library(caret)
library(mlbench)

train.control <- trainControl(method = "cv", number = 10)
model<-train(Purchase~.,data=Caravan,family="binomial",method= "glm",
             metric = "Accuracy", trControl = train.control)

print(model)

train.control <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)
model<-train(Purchase~.,data=Caravan,family="binomial",method= "glm",
             metric = "ROC", trControl = train.control)

print(model)

MatricediConfusione_rel<-confusionMatrix(model,reference=Caravan$Purchase,
                                         positive='si',"none")
#Decision Tree

library(rpart)
library(rpart.plot)
library(caret)
library(mlbench)

mod.alb<-rpart(Purchase~.,data=Caravan,method="class",minbucket=1,minsplit=1)
summary(mod.alb)

set.seed(2468)

train.control <- trainControl(method = "cv", number = 10)
model<-train(Purchase~.,data=Caravan,method= "rpart",
             metric = "Accuracy", trControl = train.control)

print(model)

set.seed(2468)
train.control <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)
model<-train(Purchase~.,data=Caravan,method= "rpart",
             metric = "ROC", trControl = train.control)

print(model)

rpart.plot(mod.alb,type=5,digits=3,fallen.leaves=TRUE)

set.seed(2468)
train<-sample(1:nrow(Caravan),3/4*nrow(Caravan))
traindata<-Caravan[train,]
testdata<-Caravan[-train,]
mod.alb.train<-rpart(Purchase~.,data=traindata,method="class")
rpart.plot(mod.alb.train,type=5,digits=3,fallen.leaves=TRUE)
pred<-predict(mod.alb.train,newdata = testdata,type="class")
mat.conf<-table(testdata$Purchase,pred)
mat.conf
mean(testdata$Purchase==pred)

train<-sample(1:nrow(Caravan),3/4*nrow(Caravan))
traindata<-Caravan[train,]
testdata<-Caravan[-train,]
mod.alb.train<-rpart(Purchase~.,data=traindata,method="class",cp= 0.000003563)
rpart.plot(mod.alb.train,type=5,digits=3,fallen.leaves=TRUE)

pred<-predict(mod.alb.train,newdata = testdata,type="class")
mat.conf<-table(testdata$Purchase,pred)
mat.conf
mean(testdata$Purchase==pred)

#Boosting (Adaboost)

library(caret)
library(mlbench)

set.seed(2468)
indice<-sample(1:nrow(Caravan), 0.75*nrow(Caravan))

train<-Caravan[indice,]
test<-Caravan[-indice,]

set.seed(2468)
train.control <- trainControl(method = "cv", number = 10)
model<-train(Purchase~., data=train,method= "adaboost",
             metric = "Accuracy", trControl = train.control)
print(model)

pred<-predict(model, newdata=test,type="class")
mean(test$Purchase!=pred)

library(adabag)
help("adabag")
mod<-boosting(Purchase~.,data=train)
summary(mod)
#RIDGE

library(leaps)
library(Matrix)
library(glmnet)

x<-model.matrix(Purchase~.,Caravan)
y<-Caravan$Purchase

indice<-sample(1:nrow(x),3/4*(nrow(x)))
xtrain<-x[indice,]
xtest<-x[-indice,]
ytrain<-y[indice]
ytest<-y[-indice]

fit.cv.ridge<-cv.glmnet(xtrain,ytrain,alpha=0,family="binomial")
coef(fit.cv.ridge)
plot(fit.cv.ridge)

fit.cv.ridge
summary(fit.cv.ridge)
lambda.sel<-fit.cv.ridge$lambda.min
prev<-predict(fit.cv.ridge,s=lambda.sel,newx=xtest)
pred<-ifelse(prev>0.5,1,0)
mat.conf<-table(ytest,pred)
tbc<-sum(diag(mat.conf))/sum(mat.conf)
tbc

lambda.sel1se<-fit.cv.ridge$lambda.1se
prev.1se<-predict(fit.cv.ridge,s=lambda.sel1se,newx=xtest)
pred.1se<-ifelse(prev.1se>0.5,1,0)
mat.conf.1se<-table(ytest,pred.1se)
tbc.1se<-sum(diag(mat.conf.1se))/sum(mat.conf.1se)
tbc.1se


griglia<-10^seq(10,-2,length=100)
out.ridge<-glmnet(xtrain,ytrain,lambda=griglia,alpha=0,family="binomial")
ridge.coef<-predict(out.ridge,type="coefficients",s=lambda.sel1se)
ridge.coef


#LASSO

fit.cv.lasso<-cv.glmnet(xtrain,ytrain,alpha=1,family="binomial")
coef(fit.cv.lasso)
plot(fit.cv.lasso)
fit.cv.lasso$lambda.min
fit.cv.lasso$lambda.1se

lambda.sel.lasso<-fit.cv.lasso$lambda.min
prev.lasso<-predict(fit.cv.lasso,s=lambda.sel.lasso,newx=xtest)
pred.lasso<-ifelse(prev.lasso>0.5,1,0)
mat.conf<-table(ytest,pred.lasso)
tbc<-sum(diag(mat.conf))/sum(mat.conf)
tbc

lambda.sel.lasso.1se<-fit.cv.lasso$lambda.1se
prev.lasso.1se<-predict(fit.cv.lasso,s=lambda.sel.lasso.1se,newx=xtest)
pred.1se<-ifelse(prev.lasso.1se>0.5,1,0)
mat.conf.1se<-table(ytest,pred.1se)
tbc.1se<-sum(diag(mat.conf.1se))/sum(mat.conf.1se)
tbc.1se



coef(fit.cv.lasso,lambda=lambda.1se)

out.lasso<-glmnet(xtrain,ytrain,lambda=griglia,alpha=1,family="binomial")
lasso.coef<-predict(out.lasso,type="coefficients",s=lambda.sel.lasso.1se)
lasso.coef

#LDA
library(MASS)

indice<-sample(1:nrow(Caravan),3/4*(nrow(Caravan)))
train_data<-Caravan[indice,]
test_data<-Caravan[-indice,]

mod.lda<-lda(Purchase~.,data=train_data)
summary(mod.lda)

pred.lda<-predict(mod.lda,newdata=test_data)
mat.conf.lda<-table(test_data$Purchase,pred.lda$class)
mat.conf.lda

mean(test_data$Purchase!=pred.lda$class)




#QDA
library(MASS)

mod.qda<-qda(Purchase~.,data=train_data)
summary(mod.qda)

pred.qda<-predict(mod.qda,newdata=test_data)
mat.conf.qda<-table(test_data$Purchase,pred.qda$class)
mat.conf.qda

mean(test_data$Purchase!=pred.qda$class)

#SVM
library(e1071)


set.seed(2468)
train<-sample(1:nrow(Caravan),0.75*nrow(Caravan))

svmfit<-svm(Purchase~.,data=Caravan[train,],kernel="radial",cost=1,gamma=2)
plot(svmfit,Caravan[train,])
#plot qui funziona solo con variabili esplicative continue e in data devo mettere al massimo 3 var.
#(inclusa la y)
pred<-predict(svmfit,newdata=Caravan[-train,])
table(true=Caravan[-train,"Purchase"],pred=predict(svmfit,
                                                   newdata=Caravan[-train,]))
mean(Caravan[-train,"Purchase"]==pred)
#per capire il miglior modello si usa la CV
tune.out<-tune(svm,Purchase~.,data=Caravan[train,],kernel="radial",ranges=list(cost=c(0.1,1,5,10,100),gamma=c(1,2,3,4,5)))
summary(tune.out)
plot(tune.out$best.model,Caravan[train,])

table(true=Caravan[-train,"Purchase"],pred=predict(tune.out$best.model,
                                         newdata=Caravan[-train,]))


### RETI NEURALI

library(neuralnet)
library(MASS)
library(dplyr)
library(ISLR)

data(Caravan)
dati<-Caravan
head(dati)
apply(dati,2,function(x) sum(is.na(x)))

set.seed(2468)
index<-sample(1:nrow(dati),round(0.8*nrow(dati)))
train<-dati[index,]
test<-dati[-index,]

nomi<-names(train) 
form<-as.formula(paste("Purchase~",paste(nomi[!nomi %in% "Purchase"],
                                     collapse="+")))
retneur<-neuralnet(form,data=train,hidden=c(30),act.fct = "logistic",
                   linear.output = F,algorithm = "backprop",learningrate = 0.01
                   )
#sag è stochastic average gradient
retneur$net.result

str(retneur)
summary(retneur)
plot(retneur)
retneur$act.fct

dim(test)
prev.retneur<-neuralnet::compute(retneur,test[,1:85])
pred.retneur<-ifelse(prev.retneur$net.result[,1]>0.5,1,0)

###oppure
pred.retneur<-apply(prev.retneur$net.result,1,which.max)
table(pred.retneur)
is.integer(pred.retneur)
pred.retneur[pred.retneur==1]<-0
pred.retneur[pred.retneur==2]<-1
pred.retneur<-as.factor(pred.retneur)

str(prev.retneur)
#prev.retneur.or<-prev.retneur$net.result*(max(dati$medv)-min(dati$medv))+min(dati$Purchase)
#test.or<-(testst$medv)*(max(dati$medv)-min(dati$medv))+min(dati$medv)

mat.conf<-table(test$Purchase,pred.retneur)
mat.conf
sum(diag(mat.conf))/sum(mat.conf)

softplus <- function(x) log(1 + exp(x))
#tanh<-function(x) (exp(x)-exp(-x))/(exp(x)+exp(-x))

##
par(mfrow=c(2,2))
gwplot(retneur,selected.covariate = "",min=,max=)
gwplot(retneur,selected.covariate = "",min=,max=)
gwplot(retneur,selected.covariate = "",min=,max=)
gwplot(retneur,selected.covariate = "",min=,max=)

###

library(tidyverse)
library(caret)
library(auc)
set.seed(2468)

train.control<-trainControl(method="repeatedcv",number=10,repeats = 3)
model6<-train(Purchase~.,data=Caravan,method="svmRadial",trControl=train.control)

print(model6)


#Esercizio 2

library(datarium)
data("marketing")
attach(marketing)
set.seed(13579)
limiti<-range(youtube)

dim(marketing)
str(marketing)

library(ggplot2)
ggplot(marketing, aes(youtube,sales))+
  geom_point()+
  geom_smooth()

library(MASS)
ggplot(marketing, aes(facebook,sales))+
  geom_point()+
  geom_smooth(method="gam")

ggplot(marketing, aes(newspaper,sales))+
  geom_point()+
  geom_smooth(method="gam")

ggplot(marketing, aes(youtube,sales))+ geom_boxplot()
ggplot(marketing, aes(sales))+
  geom_histogram(binwidth = 1)
mean(sales)

#RIDGE

library(leaps)
library(Matrix)
library(glmnet)

x<-model.matrix(sales~.,marketing)
y<-marketing$sales

indice<-sample(1:nrow(x),3/4*(nrow(x)))
xtrain<-x[indice,]
xtest<-x[-indice,]
ytrain<-y[indice]
ytest<-y[-indice]

fit.cv.ridge<-cv.glmnet(xtrain,ytrain,alpha=0)
coef(fit.cv.ridge)
plot(fit.cv.ridge)

fit.cv.ridge
summary(fit.cv.ridge)
lambda.sel<-fit.cv.ridge$lambda.min
prev<-predict(fit.cv.ridge,s=lambda.sel,newx=xtest)
MSE.ridge.min<-mean((prev-ytest)^2)
MSE.ridge.min

lambda.sel1se<-fit.cv.ridge$lambda.1se
prev.1se<-predict(fit.cv.ridge,s=lambda.sel1se,newx=xtest)
MSE.ridge.1se<-mean((prev-ytest)^2)
MSE.ridge.1se

griglia<-10^seq(10,-2,length=100)
out.ridge<-glmnet(xtrain,ytrain,lambda=griglia,alpha=0)
ridge.coef<-predict(out.ridge,type="coefficients",s=lambda.sel1se)
ridge.coef


#LASSO

fit.cv.lasso<-cv.glmnet(xtrain,ytrain,alpha=1)
coef(fit.cv.lasso)
plot(fit.cv.lasso)
fit.cv.lasso$lambda.min
fit.cv.lasso$lambda.1se

lambda.sel.lasso<-fit.cv.lasso$lambda.min
prev.lasso<-predict(fit.cv.lasso,s=lambda.sel.lasso,newx=xtest)
MSE.lasso.min<-mean((prev.lasso-ytest)^2)
MSE.lasso.min

lambda.sel.lasso.1se<-fit.cv.lasso$lambda.1se
prev.lasso.1se<-predict(fit.cv.lasso,s=lambda.sel.lasso.1se,newx=xtest)
MSE.lasso.1se<-mean((prev.lasso-ytest)^2)
MSE.lasso.1se



coef(fit.cv.lasso,lambda=lambda.1se)

out.lasso<-glmnet(xtrain,ytrain,lambda=griglia,alpha=1)
lasso.coef<-predict(out.lasso,type="coefficients",s=lambda.sel.lasso.1se)
lasso.coef

#PCR
library(pls)
train<-sample(1:nrow(marketing),3/4*(nrow(marketing)))

x<-model.matrix(sales~.,marketing)[,-1]
y<-na.omit(marketing$sales)

fit.pcr<-pcr(sales~.,data=marketing[train,],scale=TRUE,validation="CV")
help("pcr")
summary(fit.pcr)
validationplot(fit.pcr,val.type="MSEP")

pcr.prev<-predict(fit.pcr,x[-train,],ncomp=2)

MSE.pcr<-mean((pcr.prev-y[-train])^2)
MSE.pcr


#PLS
library(pls)

x<-model.matrix(sales~.,marketing)[,-1]
y<-na.omit(marketing$sales)


fit.pls<-plsr(sales~.,data=marketing,scale=TRUE,subset=train,validation="CV")

summary(fit.pls)
validationplot(fit.pls,val.type="MSEP")

pls.prev<-predict(fit.pls,x[-train,],ncomp=1)

MSE.pls<-mean((pls.prev-y[-train])^2)
MSE.pls

biplot(fit.pls)

par(mfrow = c(2,2))
biplot(fit.pls, which = "x") # Default
biplot(fit.pls, which = "y")
biplot(fit.pls, which = "scores")
biplot(fit.pls, which = "loadings")
#Local

set.seed(1234)
indice<-sample(1:nrow(marketing),0.7*nrow(marketing))
train<-marketing[indice,]
test<-marketing[-indice,]

mod.loc<-loess(sales~.,data=train,span=2,degree=2)
summary(mod.loc)
pred.loc<-predict(mod.loc,se=TRUE,newdata = test)
head(marketing)
plot(test$youtube,test$sales,xlim=limiti,col="grey")
lines(test$youtube[order(test$youtube)],pred.loc$fit[order(test$youtube)],col="red",lwd=2)
MSE.loc<-mean((test$sales-pred.loc$fit)^2)
MSE.loc

mod.lin<-lm(sales~.,data=train)
summary(mod.lin)
prev<-predict(mod.lin,se=TRUE,newdata=test)
head(prev$fit)
plot(youtube,sales,xlim=limiti,col="grey")
lines(youtube[order(youtube)],prev$fit[order(youtube)],col="red",lwd=2)
MSE.lin<-mean((test$sales-prev$fit)^2)
MSE.lin
plot(mod.lin)
res.stud<-rstandard(mod.lin)
qqnorm(res.stud)
qqline(res.stud)
shapiro.test(res.stud)
#non è verificata la normalità e la linearità

X<-marketing[,1:3]
autov<-eigen(cor(X))
CI<-sqrt(max(autov$values)/min(autov$values))
#non c'è multicollinearità

#Linear
library(lmtest)
bptest(mod.lin)

residui<-mod.lin$residuals
par(mfrow=c(2,2))
plot(youtube,residui)
abline(h=0)
plot(facebook,residui)
abline(h=0)
mean(residui)
#media residui pari a zero

#splines
library(splines2)
library(splines)

fiss<-attr(bs(c(youtube,facebook,newspaper),df=8),"knots")
head(marketing)
mod.spline<-lm(sales~bs(youtube+facebook+newspaper,knots=c(20.01,42.36,91.59),degree=3),data=marketing)
summary(mod.spline)
pred.spline<-predict(mod.spline,se=TRUE)
plot(youtube[order(youtube)],pred.spline$fit[order(youtube)],col="red",lwd=2)

plot(youtube[order(youtube)],sales,col="grey")
lines(youtube[order(youtube)],pred.spline$fit[order(youtube)],col="red",lwd=2)

mse<-mean((sales-pred.spline$fit)^2)
mse
#
attr(bs(c(youtube),df=6),"knots")

mod.spline<-lm(sales~bs(youtube,knots=c(89.25,179.70,262.59),degree=3),data=marketing)
summary(mod.spline)
pred.spline<-predict(mod.spline,se=TRUE)
plot(youtube[order(youtube)],pred.spline$fit[order(youtube)],col="red",lwd=2)

plot(youtube[order(youtube)],sales,col="grey")
abline(v=89.25)
abline(v=179.7)
abline(v=262.59)
lines(youtube[order(youtube)],pred.spline$fit[order(youtube)],col="red",lwd=2)

mse<-mean((sales-pred.spline$fit)^2)
mse
#natural splines
head(marketing)

mod.nspline<-lm(sales~ns(youtube+facebook+newspaper,df=6),data=marketing)
summary(mod.nspline)
pred.nspline<-predict(mod.nspline,se=TRUE)
plot(youtube[order(youtube)],pred.nspline$fit[order(youtube)],col="red",lwd=2)

plot(youtube[order(youtube)],sales,col="grey")
lines(youtube[order(youtube)],pred.nspline$fit[order(youtube)],col="red",lwd=2)

mse<-mean((sales-pred.nspline$fit)^2)
mse
#
mod.nspline<-lm(sales~ns(youtube,df=6),data=marketing)
summary(mod.nspline)
pred.nspline<-predict(mod.nspline,se=TRUE)
plot(youtube[order(youtube)],pred.nspline$fit[order(youtube)],col="red",lwd=2)

plot(youtube[order(youtube)],sales,col="grey")
lines(youtube[order(youtube)],pred.nspline$fit[order(youtube)],col="red",lwd=2)

lines(youtube[order(youtube)],pred.nspline$fit[order(youtube)]+2*pred.nspline$se[order(youtube)],col="green",lwd=2)
lines(youtube[order(youtube)],pred.nspline$fit[order(youtube)]-2*pred.nspline$se[order(youtube)],col="green",lwd=2)

mse<-mean((sales-pred.nspline$fit)^2)
mse

#mars
library(earth)
library(caret)
library(mlbench)
set.seed(123)
train<-sample(1:nrow(marketing),0.7*nrow(marketing))
marketing.train<-marketing[train,]
marketing.test<-marketing[-train,]

mod.mars<-earth(sales~.,data=marketing.train)
mod.mars.int<-earth(sales~.,data=marketing.train,degree=1,nprune=4)
print(mod.mars)


summary(mod.mars,style="pmax")
summary(mod.mars)
plotmo(mod.mars)
#Youtube
# per i valori fino a 53.64 contribuisce solo -0.1495289 + 18.04
#mentre per i valori dopo 53.64 contribuisce +0.03767 + 18.04

# facebook
# stesso ragionamento

plot(mod.mars,which=1)

pred<-predict(mod.mars,newdata=marketing.test,se=TRUE)
pred.int<-predict(mod.mars.int,newdata=marketing.test,se=TRUE)

mse<-mean((marketing.test$sales-pred)^2)
mse

#CV per trovare il miglior modello di MARS con degree=1
library(caret)
library(mlbench)

set.seed(13579)
train.control <- trainControl(method = "cv", number = 10)
model<-train(sales~.,data=marketing,method = "gam",
             metric ="RMSE", trControl = train.control)
print(model)

#GAM,Generalized Additive Models

library(gam)
library(mgcv)
head(marketing)
par(mfrow=c(3,3))
plot(youtube,sales)
plot(facebook,sales)
plot(newspaper,sales)
mod.gam<-gam(sales~s(youtube)+s(facebook)+ns(newspaper),data=marketing.train)

plot(mod.gam)
mod.gam2<-gam(sales~s(youtube,spar=0.6)+s(facebook,spar=0.6)+newspaper
  ,data=marketing.train)
plot(mod.gam2)

prev.1<-predict(mod.gam,newdata=marketing.test,se=TRUE)
prev.2<-predict(mod.gam2,newdata=marketing.test,se=TRUE)

MSE1<-mean((marketing.test$sales-prev.1)^2)
MSE1
sqrt(MSE1)
MSE2<-mean((marketing.test$sales-prev.2)^2)
MSE2
dim(marketing)



anova(mod.gam,mod.gam2,test="F")

#Albero di regressione
set.seed(13579)

library(rpart)
library(rpart.plot)
library(caret)
library(mlbench)

mod.tree<-rpart(sales~.,data=marketing,method="anova")
rpart.plot(mod.tree,type=5,digits=3,fallen.leaves = TRUE)


train.control <- trainControl(method = "cv", number = 10)
model<-train(sales~.,data=marketing,method= "rpart",
             metric = "RMSE", trControl = train.control)

print(model)

train.control <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)


print(model)
par(mfrow=c(1,1))
mod.tree2<-prune(mod.tree,cp=0.06431633)
rpart.plot(mod.tree2,type=5,digits=3,fallen.leaves = TRUE)

prev<-predict(mod.tree2,se=TRUE)
rmse<-sqrt(mean((sales-prev)^2))

mod.tree3<-rpart(sales~youtube,data=marketing,method="anova")
prev3<-predict(mod.tree3,se=TRUE)

plot(youtube,sales,col="grey")
lines(youtube[order(youtube)],prev3[order(youtube)],col="blue",lwd=2)

### RETI NEURALI
library(neuralnet)
library(datarium)
data("marketing")
dati<-marketing
attach(dati)

set.seed(13579)
index<-sample(1:nrow(dati),round(0.8*nrow(dati)))
train<-dati[index,]
test<-dati[-index,]

maxs<-apply(dati,2,max)
mins<-apply(dati,2,min)
datist<-as.data.frame(scale(dati,center=mins,scale=maxs-mins))
#centratura fatta con il minimo della colonna e poi diviso la diff tra max e min delal col

trainst<-datist[index,]
testst<-datist[-index,]

nomi<-names(train) 
form<-as.formula(paste("sales~",paste(nomi[!nomi %in% "sales"],
                                         collapse="+")))
retneur<-neuralnet(form,data=trainst,hidden=c(2),act.fct=softplus,
                   linear.output = T)
retneur$net.result

str(retneur)
summary(retneur)
plot(retneur)
retneur$act.fct

dim(test)
prev.retneur<-neuralnet::compute(retneur,testst[,1:3])
prev.retneur.or<-prev.retneur$net.result*(max(dati$sales)-min(dati$sales))+min(dati$sales)
test.or<-(testst$sales)*(max(dati$sales)-min(dati$sales))+min(dati$sales)

sqrt(mean((test.or-prev.retneur.or)^2))

RMSE<- test %>%
  mutate(residual=test$sales-prev.retneur$net.result) %>%
  summarize(sqrt(mean(residual^2)))
RMSE


