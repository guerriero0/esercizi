bank <- read.csv("~/Downloads/archive/bank.csv")
head(bank)
attach(bank)

bank$y<-as.factor(bank$y)
bank$education<-as.factor(bank$education)
bank$month<-factor(bank$month, levels=c("jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"))

library(dplyr)
a<-bank %>%
  group_by(month,education) %>%
  summarize(media=mean(balance))

library(ggplot2)
ggp <- ggplot(data=a, aes(x = month , y = media,group=1)) + 
  geom_line() +
  ggtitle("Grafico di serie storica") +
  xlab("Mese") + ylab("Balance")+
  expand_limits(y=0)
ggp


a<-bank %>%
  select(month,education,balance) %>%
  group_by(month,education) %>%
  summarize(media=mean(balance))
            
a$month<-as.numeric(a$month)

mesi<-c("jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec")
ggp <- ggplot(data=a, aes(x = month,y=media,color=education)) + 
  geom_line(lwd=1.1) +
  ggtitle("Grafico di serie storica") +
  xlab("Mese") + ylab("Balance")+
  xlim(mesi)+
  expand_limits(y=0)
ggp

str(bank)
balance<-as.numeric(balance)

ggplot(bank, aes(age,balance))+
  geom_point()+
  facet_wrap(~education)

ggplot(bank, aes(balance,age, color=education))+
  geom_point()

medie<-bank %>%
  group_by(education) %>%
  summarise(medie=mean(balance))

#ACP,variabili quantitative,ML non supervisionato cioè senza una variabile di risposta
library(FactoMineR)

ris.pca<-PCA(bank,scale.unit=TRUE, quanti.sup=c(11:12),quali.sup=13)

plot.PCA(ris.pca,axes=c(3,4),choir="var")
ris.pca
ris.pca$eig

ris.clus<-HCPC(ris.pca)


#K means

ris.km<-kmeans(bank[,c("balance","age","duration")],3,nstart =20 )
plot(bank[,c("balance","age","duration")],col=ris.km$cluster+1)

ggplot(bank, aes(age,balance, color=y))+
  geom_point()
sum(is.na(bank))


#RETI NEURALI

library(neuralnet)
library(MASS)
library(ISLR)

maxs<-apply(bank[,c("balance","duration","pdays")],2,max)
mins<-apply(bank[,c("balance","duration","pdays")],2,min)

bankist<- bank %>%
  mutate(balance=scale(balance,center=mins[1],scale=maxs[1]-mins[1]),
         duration=scale(duration,center=mins[2],scale=maxs[2]-mins[2]),
         pdays=scale(pdays,center=mins[3],scale=maxs[3]-mins[3]))

set.seed(2468)
index<-sample(1:nrow(bankist),round(0.75*nrow(bankist)))
train<-bankist[index,]              
test<-bankist[-index,]

x<-model.matrix(y~.,bankist)
x
x<-data.frame(x)
y<-bankist$y
y
xtrain<-x[index,]
xtest<-x[-index,]
ytrain<-y[index]
ytest<-y[-index]

mod_retneur<-neuralnet(ytrain~.,data=xtrain,hidden=c(5),act.fct="logistic",linear.output = T,
                       algorithm = "sag",learningrate = 0.00001,threshold = 0.5)

mod_retneur2<-neuralnet(ytrain~.,data=xtrain,hidden=c(5),act.fct="tanh",linear.output = F,
                       algorithm = "rprop+",learningrate = 0.00001,threshold = 0.5)

softplus <- function(x) log(1 + exp(x))
mod_retneur3<-neuralnet(ytrain~.,data=xtrain,hidden=c(5),act.fct=softplus,linear.output = F,
                        algorithm = "sag",learningrate = 0.0001,threshold = 1)

# Error in if (reached.threshold < min.reached.threshold) { : 
# missing value where TRUE/FALSE needed 
# Soluzione:
# 1)Abbassare learning rate e mettere threshold=0.5
# 2)usare linear.output=F al posto di T
# abbassare il threshold dovrebbe migliorare la capacità del modello di distinguere le categorie
# assieme al learning rate, sopratutto quando c'è una forte sproporzionalità nelle categorie di y
# "sag"("stochastic average gradient" velocizza la convergenza)
# il più lento  è "backprop"
# linear.output=F può migliorare il modello rispetto a  T


plot(mod_retneur)

summary(mod_retneur)
mod_retneur$net.result
table(y)

prev.retneur<-neuralnet::compute(mod_retneur,xtest)
pred.retneur<-ifelse(prev.retneur$net.result[,1]>0.5,"no","yes")
pred.retneur<-as.factor(pred.retneur)
table(pred.retneur)

length(pred.retneur)
length(ytest)
mat.conf<-table(ytest,pred.retneur)
sum(diag(mat.conf))/sum(mat.conf)

library(pROC)
prev.retneur<-neuralnet::compute(mod_retneur,xtest)
auc.neuralnet<-auc(ytest,prev.retneur$net.result[,2])
auc.neuralnet


## RETI NEURALI USANDO NNET e CV

# Utilizzare train, dicotomizza automaticamente la matrice x
library(caret)
library(mlbench)
set.seed(2468)
control <- trainControl(method="repeatedcv", number=10, repeats=3)
model <- train(y~., data=train, method="nnet",trControl=control)
print(model)
#size è il numero di neuroni latenti, decay è "weight decay" del termine di penalizzazione (lambda)
imp<-varImp(model)
plot(imp)
plot(model)

library(nnet)
mod<-nnet(y~.,data=train,size=1,decay=0.1)
summary(mod)
str(mod)

library(devtools)
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')
wts.in<-mod$wts
struct<-mod$n
plot.nnet(mod)
plot.nnet(wts.in,struct=struct)

prev<-predict(mod,newdata=test)
pred<-predict(mod,newdata=test,type="class")
mean(test$y==pred)

library(pROC)
auc.mod<-auc(test$y,prev[,1])
auc.mod

#ALBERI

library(rpart)
library(rpart.plot)

set.seed(2468)
index<-sample(1:nrow(bank),round(0.75*nrow(bank)))
train<-bank[index,]              
test<-bank[-index,]

#MODELLO SEMPLICE
mod.alb<-rpart(y~.,data=train,method="class")
rpart.plot(mod.alb,type=5,digits=3,fallen.leaves = TRUE)

pred<-predict(mod.alb,newdata=test,type="class")
table(pred)
mat.conf<-table(test$y,pred)
mat.conf
mean(test$y==pred)

#CALCOLO AUC
library(pROC)
prev<-predict(mod.alb,newdata=test,type="prob")
auc.tree<-auc(test$y,prev[,2])
auc.tree

plot(mod.alb$variable.importance)

# GRAFICO VARIABLE IMPORTANCE
library(tidyverse)
df <- data.frame(imp = mod.alb$variable.importance)
df2 <- df %>% 
  tibble::rownames_to_column() %>% 
  dplyr::rename("variable" = rowname) %>% 
  dplyr::arrange(imp) %>%
  dplyr::mutate(variable = forcats::fct_inorder(variable))

ggplot2::ggplot(df2) +
  geom_col(aes(x = variable, y = imp),
           col = "black", show.legend = F) +
  coord_flip() +
  scale_fill_grey() +
  theme_bw()


#MODELLO 2 con CV
library(caret)
library(mlbench)
set.seed(2468)

train.control <- trainControl(method = "cv", number = 10)
model<-train(y~.,data=bank,method= "rpart",
             metric = "Accuracy", trControl = train.control)

print(model)

set.seed(2468)
train.control <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)
model<-train(y~.,data=bank,method= "rpart",
             metric = "ROC", trControl = train.control)

print(model)

#cp migliore è 0.01055662

mod.alb2<-rpart(y~.,data=train,method="class",cp=0.01055662)
rpart.plot(mod.alb,type=5,digits=3,fallen.leaves = TRUE)

pred<-predict(mod.alb2,newdata=test,type="class")
table(pred)
mat.conf<-table(test$y,pred)
mat.conf
mean(test$y==pred)

#CALCOLO AUC
prev2<-predict(mod.alb2,newdata=test,type="prob")
auc.tree<-auc(test$y,prev2[,2])
auc.tree

# GRAFICO VARIABLE IMPORTANCE
library(tidyverse)
df <- data.frame(imp = mod.alb2$variable.importance)
df2 <- df %>% 
  tibble::rownames_to_column() %>% 
  dplyr::rename("variable" = rowname) %>% 
  dplyr::arrange(imp) %>%
  dplyr::mutate(variable = forcats::fct_inorder(variable))

ggplot2::ggplot(df2) +
  geom_col(aes(x = variable, y = imp),
           col = "black", show.legend = F) +
  coord_flip() +
  scale_fill_grey() +
  theme_bw()


##Logistic

set.seed(2468)
index<-sample(1:nrow(bank),round(0.75*nrow(bank)))
trainglm<-bank[index,]              
testglm<-bank[-index,]

mod.glm<-glm(y~.,data=trainglm,family="binomial")
summary(mod.glm)

set.seed(2468)

train.control <- trainControl(method = "cv", number = 10)
model<-train(y~.,data=bank,method= "glm",family="binomial",
             metric = "Accuracy", trControl = train.control)

print(model)

set.seed(2468)
train.control <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)
model<-train(y~.,data=bank,method= "glm",family="binomial",
             metric = "ROC", trControl = train.control)

print(model)

prev<-predict(mod.glm,newdata=testglm,type="response")
pred<-ifelse(prev>0.5,"yes","no")
mean(testglm$y==pred)

auc.glm<-auc(testglm$y,prev)
auc.glm

#RANDOM FOREST

library(ggplot2)
library(cowplot)
library(randomForest)

table(bank$y)
dim(bank)
mod<-randomForest(y~.,data=train,mtry=sqrt(17))
mod
varImpPlot(mod)


oob.error.data <- data.frame(
  Trees=rep(1:nrow(mod$err.rate), times=3),
  Type=rep(c("OOB", "no", "yes"), each=nrow(mod$err.rate)),
  Error=c(mod$err.rate[,"OOB"], 
          mod$err.rate[,"no"], 
          mod$err.rate[,"yes"]))

ggplot(data=oob.error.data, aes(x=Trees, y=Error)) +
  geom_line(aes(color=Type))

pred.rf<-predict(mod,newdata=test,type="class")
prev.rf<-predict(mod,newdata=test,type="prob")

mean(test$y!=pred.rf)

library(pROC)
auc.rf<-auc(test$y,prev.rf[,2])
auc.rf

#ADABOOST
mod.adaboost<-boosting(y~.,data=train,mfinal=100,coeflearn = "Freund")
summary(mod.adaboost)

pred.ada<-predict(mod.adaboost,newdata=test)
pred.ada$error
mean(test$y!=pred.ada$class)

auc.ada<-auc(test$y,pred.ada$prob[,2])
auc.ada

errorevol(mod.adaboost,newdata=train)->evol.train
errorevol(mod.adaboost,test)->evol.test

plot.errorevol(evol.test,evol.train)

##SVM

library(e1071)

set.seed(2468)
indice<-sample(1:nrow(bank),0.7*nrow(bank))
train<-bank[indice,]
test<-bank[-indice,]

mod.svm<-svm(y~.,data=train,cost=1,scale=FALSE,kernel="radial",probability=TRUE)
summary(mod.svm)

plot(mod.svm,data=train)

pred<-predict(mod.svm,newdata=test,probability=TRUE)
table(test$y,pred)
mean(test$y!=pred)

library(pROC)
auc.svm<-auc(test$y,attr(pred,"probabilities")[,2])
auc.svm

##
library(dplyr)
conteggi<-bank %>%
  count(y) 

dati<-bank %>%
  group_by(education) %>%
     count(y)

no<-dati[which(dati$y=="no"),3]/conteggi[1,2]
yes<-dati[which(dati$y=="yes"),3]/conteggi[2,2]
prep.dati<-data.frame(no,yes)
colnames(prep.dati)<-c("no","yes")
rownames(prep.dati)<-c("primary","secondary","tertiary","unknown")
prep.dati
##

##GLM
mod.glm<-glm(y~.,data=train,family="binomial")
summary(mod.glm)

prev<-predict(mod.glm,newdata=test,type="response")
pred<-ifelse(prev>0.5,1,0)
pred[which(pred=="0")]<-"no"
pred[which(pred=="1")]<-"yes"
pred<-as.factor(pred)
mean(test$y!=pred)

library(pROC)
auc.glm<-auc(test$y,prev)
auc.glm
