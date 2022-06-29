pokedex <- read.csv("~/Downloads/pokedex.csv")
pokedex$type<-as.factor(pokedex$type)
pokedex$is_legendary<-as.factor(pokedex$is_legendary)
attach(pokedex)

#ANALYSIS
head(pokedex)                                
table(is_legendary)
str(pokedex)
table(name)

library(dplyr)

legendary_pokemon<-pokedex %>%
  count(is_legendary) %>%
  mutate(prop=n/nrow(pokedex))

x<-pokedex$height_m
y<-pokedex$weight_kg

library(ggplot2)
ggplot(pokedex, aes(x,y))+
  geom_point(aes(color=is_legendary),size=2)+
  geom_smooth(method="lm")+
  geom_text(aes(label=ifelse(height_m>7.5 | weight_kg>600,as.character(name),"")),vjust=0,hjust=0)+
  expand_limits(x=16)+
  labs(title="Pokemon leggendari per altezza e peso",
       x="Altezza",y="Peso")

pokedex[pokedex$is_legendary %in% c(1),]
legendary_pokedex<-subset(pokedex, is_legendary==1)

legend_by_type<- legendary_pokedex %>%
  group_by(type) %>%
  count(is_legendary) 
  
a<-prop.table(table(type,is_legendary),1)
b<-a[,"1"]
legend_by_type<-sort(b,decreasing = TRUE)
legend_by_type


library(tidyr)
legend_by_stats <- pokedex  %>% 
  select(is_legendary, attack, sp_attack, defense, sp_defense, hp, speed)  %>% 
  gather(key = fght_stats, value = value, -is_legendary) 


legend_by_stats_plot <- legend_by_stats %>% 
  ggplot(aes(x = is_legendary, y = value, fill = is_legendary)) +
  geom_boxplot(varwidth = TRUE) +
  facet_wrap(~fght_stats) +
  labs(title = "Pokemon fight statistics",
       x = "Legendary status") 
legend_by_stats_plot


##Alberi di classificazione
set.seed(1234)
indice<-sample(1:nrow(pokedex),0.6*nrow(pokedex))
train<-pokedex[indice,]
test<-pokedex[-indice,]

library(rpart)
library(rpart.plot)

library(caret)
library(mlbench)
set.seed(1234)
control <- trainControl(method="cv", number=10)
model <- train(is_legendary~attack + defense + height_m + 
                 hp + sp_attack + sp_defense + speed + type + weight_kg,data=train,
               method="rpart",trControl=control,na.action = na.omit)
print(model)

mod.alb<-rpart(is_legendary~attack + defense + height_m + 
                 hp + sp_attack + sp_defense + speed + type + weight_kg,data=train,
               na.action = na.omit,cp=0.06666667)
rpart.plot(mod.alb,digits = 3,5,fallen.leaves = TRUE)

pred<-predict(mod.alb,newdata=test,type="class")
head(pred)
mat.conf<-table(test$is_legendary,pred)
sum(diag(mat.conf))/sum(mat.conf)

library(pROC)
prev<-predict(mod.alb,newdata=test,type="prob")
auc.tree<-auc(test$is_legendary,prev[,"1"])
auc.tree

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

##RANDOM FOREST

mod.alb$variable.importance
library(randomForest)
mod.rf<-randomForest(is_legendary~attack + defense + height_m + 
                       hp + sp_attack + sp_defense + speed + type + weight_kg,data=train,
                     na.action = na.omit)
plot(mod.rf)

library(traineR)
set.seed(1234)
model <- train.randomForest(is_legendary~attack + defense + height_m + 
                 hp + sp_attack + sp_defense + speed + type + weight_kg,data=train,
               na.action = na.omit)

pred<-predict(mod.rf,newdata=test,type="class")
head(pred)
mat.conf<-table(test$is_legendary,pred)
sum(diag(mat.conf))/sum(mat.conf)

library(pROC)
prev<-predict(mod.rf,newdata=test,type="prob")
auc.rf<-auc(test$is_legendary,prev[,"1"])
auc.rf

importance(mod.rf)
varImpPlot(mod.rf)

##K Nearest Neighbor
library(class)
pokedex<-drop_na(pokedex)
datiknn<- pokedex %>%
  select(attack,defense,height_m,percentage_male,sp_attack,sp_defense,speed,weight_kg,is_legendary)

set.seed(1234)
indice<-sample(1:nrow(datiknn),0.6*nrow(datiknn))
train<-datiknn[indice,]
test<-datiknn[-indice,]

pr <- knn(train,test,cl=train$is_legendary,k=13)
mat.conf<-table(test$is_legendary,pr)
sum(diag(mat.conf))/sum(mat.conf)
mean(test$is_legendary!=pr)

prev<- knn(train,test,cl=train$is_legendary,k=13,prob=TRUE)
auc.knn<-auc(test$is_legendary,attributes(prev)$prob)
auc.knn
roc.knn<-roc(test$is_legendary,attributes(prev)$prob)
plot(roc.knn,print.thres = T,
     print.auc=T)

## RIDGE 

library(leaps)
library(Matrix)
library(glmnet)
detach(pokedex)

x<-model.matrix(is_legendary~attack+defense+height_m+hp+percentage_male+sp_attack+
                  sp_defense+speed+type+weight_kg+generation,pokedex)
y<-pokedex$is_legendary

set.seed(1234)
indice<-sample(1:nrow(pokedex),0.7*nrow(pokedex))
xtrain<-x[indice,]
xtest<-x[-indice,]
ytrain<-y[indice]
ytest<-y[-indice]
mod.ridge<-cv.glmnet(xtrain,ytrain,family="binomial",alpha=0)
mod.ridge
summary(mod.ridge)
lambda.sel<-mod.ridge$lambda.1se
lambdamin<-mod.ridge$lambda.min
prev<-predict(mod.ridge,s=lambda.sel,newx = xtest)
pred<-ifelse(prev>0.5,1,0)
mean(ytest==pred)
pred2<-predict(mod.ridge,s=lambda.min,newx=xtest,type="class")
mean(ytest==pred2)
plot(mod.ridge)

coeff<-predict(mod.ridge,type="coefficients",s=lambdamin)
coeff
griglia<-10^seq(10,-2,length=100)
out.ridge<-glmnet(xtrain,ytrain,lambda=griglia,alpha=0,family="binomial")
ridge.coef<-predict(out.ridge,type="coefficients",s=lambdamin)
ridge.coef
out.ridge$lambda.min

#LASSO

library(leaps)
library(Matrix)
library(glmnet)
detach(pokedex)

x<-model.matrix(is_legendary~attack+defense+height_m+hp+percentage_male+sp_attack+
                  sp_defense+speed+type+weight_kg+generation,pokedex)
y<-pokedex$is_legendary

set.seed(1234)
indice<-sample(1:nrow(pokedex),0.7*nrow(pokedex))
xtrain<-x[indice,]
ytrain<-y[indice]
xtest<-x[-indice,]
ytest<-y[-indice]
mod.lasso<-cv.glmnet(xtrain,ytrain,alpha=1,family="binomial")
mod.lasso
lambda.sel<-mod.lasso$lambda.1se
lambdamin<-mod.lasso$lambda.min
pred<-predict(mod.lasso,s=lambda.sel,type="class",newx=xtest)
tcc<-mean(ytest==pred)
tcc
pred2<-predict(mod.lasso,s=lambdamin,type="class",newx=xtest)
tcc2<-mean(ytest==pred2)
tcc2

plot(mod.lasso)
coeff<-predict(mod.lasso,type="coefficients",s=lambda.sel)
coeff
griglia<-10^seq(10,-2,length=100)
out.lasso<-glmnet(xtrain,ytrain,lambda=griglia,alpha=1,family="binomial")
lasso.coef<-predict(out.lasso,type="coefficients",s=lambda.sel)
lasso.coef


##SVM
library(kernlab)

model.ksvm = ksvm(is_legendary ~ sp_attack + defense, data = train, type="C-svc")
plot(model.ksvm, data=train)

mod.svm<-svm(is_legendary~sp_attack+defense ,data=train,
             cost=100,kernel="radial",gamma=1,probability=TRUE,na.action=na.omit)
mod.svm

plot(mod.svm,data=train,attack~defense,xlim=c(1,250))

library(e1071)
pokedex<-na.omit(pokedex)
set.seed(1234)
indice<-sample(1:nrow(pokedex),0.7*nrow(pokedex))
train<-pokedex[indice,]
test<-pokedex[-indice,]

mod.svm<-svm(is_legendary~attack+defense+ height_m + 
               hp + sp_attack + sp_defense + speed + type + weight_kg ,data=train,na.action = na.omit,
             cost=10,kernel="radial",gamma=0.01,probability=TRUE)
mod.svm

prev<-predict(mod.svm,newdata=test,probability = TRUE)
pred<-ifelse(attr(prev,"probabilities")[,2]>0.5,1,0)
mean(test$is_legendary!=pred)
library(pROC)
auc.svm<-auc(test$is_legendary,attr(prev,"probabilities")[,2])
auc.svm

library(caret)
library(mlbench)
set.seed(1234)
control <- trainControl(method="cv", number=10)
model <- train(is_legendary~attack + defense + height_m + 
                 hp + sp_attack + sp_defense + speed + type + weight_kg,data=train,
               method="svmLinear",trControl=control,na.action = na.omit)
print(model)

mod.svm.linear<-svm(is_legendary~attack+defense+ height_m + 
               hp + sp_attack + sp_defense + speed + type + weight_kg ,data=train,na.action = na.omit,
             cost=1,kernel="linear",probability=TRUE)

mod.svm.linear

prev.linear<-predict(mod.svm.linear,newdata=test,probability = TRUE)
pred.linear<-ifelse(attr(prev.linear,"probabilities")[,2]>0.5,1,0)
mean(test$is_legendary!=pred.linear)
library(pROC)
auc.svm.linear<-auc(test$is_legendary,attr(prev.linear,"probabilities")[,2])
auc.svm.linear

#ottimizzazione parametri SVM

tune.out<-tune(svm,is_legendary~attack+defense+ height_m + 
                 hp + sp_attack + sp_defense + speed + type + weight_kg,data=train,kernel="radial",ranges=list(cost=c(0.001,0.01,0.1,1,5,10,100,1000),gamma=c(0,5,1,2,3,4)))
summary(tune.out)
plot(tune.out$best.model,dati[train,])

table(true=dati[-train,"y"],pred=predict(tune.out$best.model,
                                         newdata=dati[-train,]))
mean(dati[-train,"y"]!=predict(tune.out$best.model,
                               newdata=dati[-train,]))
predict(tune.out$best.model,
        newdata=dati[-train,],class="prob")

#Rete Neurale

library(neuralnet)
library(nnet)
library(tidyr)
library(dplyr)
pokedex<-drop_na(pokedex)
sum(is.na(pokedex))

mins<-apply(pokedex[,c("attack","defense","height_m","hp","percentage_male","sp_attack","sp_defense","speed","weight_kg")],2,min)
maxs<-apply(pokedex[,c("attack","defense","height_m","hp","percentage_male","sp_attack","sp_defense","speed","weight_kg")],2,max)

pokedex_retneur<- pokedex %>%
  mutate(attack=scale(attack,center=mins[1],scale=maxs[1]-mins[1]),
         defense=scale(defense,center=mins[2],scale=maxs[2]-mins[2]),
         height_m=scale(height_m,center=mins[3],scale=maxs[3]-mins[3]),
         hp=scale(hp,center=mins[4],scale=maxs[4]-mins[4]),
         percentage_male=scale(percentage_male,center=mins[5],scale=maxs[5]-mins[5]),
         sp_attack=scale(sp_attack,center=mins[6],scale=maxs[6]-mins[6]),
         sp_defense=scale(sp_defense,center=mins[7],scale=maxs[7]-mins[7]),
         speed=scale(speed,center=mins[8],scale=maxs[8]-mins[8]),
         weight_kg=scale(weight_kg,center=mins[9],scale=maxs[9]-mins[9]),)

set.seed(1234)
indice<-sample(1:nrow(pokedex_retneur),0.7*nrow(pokedex_retneur))
train<-pokedex_retneur[indice,]
test<-pokedex_retneur[-indice,]

#PROCEDURA PER NEURAL NET
x<-model.matrix(is_legendary~attack+defense+sp_attack+sp_defense+height_m+hp+percentage_male+speed+weight_kg,pokedex_retneur)
x<-data.frame(x)
y<-pokedex_retneur$is_legendary
set.seed(1234)
indice<-sample(1:nrow(pokedex_retneur),round(0.7*nrow(pokedex_retneur)))
xtrain<-x[indice,]
xtest<-x[-indice,]
ytrain<-y[indice]
ytest<-y[-indice]

mod.retneur<-neuralnet(ytrain~attack+defense+sp_attack+sp_defense+height_m+hp+percentage_male+speed+weight_kg,data=xtrain,
                       hidden=c(2),act.fct = softplus,linear.output = TRUE)
mod.retneur
summary(mod.retneur)
plot(mod.retneur)

prev<-neuralnet::compute(mod.retneur,xtest)
str(prev)
head(prev$net.result)
pred<-ifelse(prev$net.result[,2]>0.5,1,0)
tanh.true<-mean(ytest!=pred)
tanh.false<-mean(ytest!=pred)
logistic.true<-mean(ytest!=pred)
logistic.false<-mean(ytest!=pred)

library(pROC)
auc.mod.retneur<-auc(ytest,prev$net.result[,2])
auc.mod.retneur

softplus <- function(x) log(1 + exp(x))

#NNET
library(caret)
library(mlbench)
set.seed(1234)
control <- trainControl(method="repeatedcv", number=10, repeats=3)
model <- train(is_legendary~attack+defense+sp_attack+sp_defense+height_m+hp+percentage_male+speed+weight_kg, data=train, method="nnet",trControl=control)
print(model)
#size è il numero di neuroni latenti, decay è "weight decay" del termine di penalizzazione (lambda)
imp<-varImp(model)
plot(imp)
plot(model)

library(nnet)
mod<-nnet(is_legendary~attack+defense+sp_attack+sp_defense+height_m+hp+percentage_male+speed+weight_kg,data=train,size=1,decay=0.1)
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
mean(test$is_legendary!=pred)

library(pROC)
auc.mod<-auc(test$is_legendary,prev[,1])
auc.mod

#### Analysis 

str(pokedex)
dati<-data.frame(name,attack,defense,hp,height_m,sp_attack,sp_defense,speed,weight_kg)
dati<-dati %>%
  filter(is_legendary==1)
sum(is.na(weight_kg))
dati<-drop_na(dati)
str(dati)
dati[] <- lapply(dati, function(x) {
  if(is.integer(x)) as.numeric(as.character(x)) else x
})
sapply(dati, class)
nomi<-dati$name
dati <-dati[,-1]

ris.km<-kmeans(dati
               ,3,nstart=20)
ris.km

gruppi<-ris.km$cluster
plot(x,col=ris.km$cluster+1)

datigruppo<-dati %>%
  mutate(gruppo=gruppi,nomi=nomi)

datigruppo$gruppo
datigruppo$gruppo<-factor(datigruppo$gruppo,level=c("1","2","3","no"))

ggplot(datigruppo, aes(speed,weight_kg))+
  geom_point(color=datigruppo$gruppo)+
  geom_text(label=as.character(nomi),vjust=0,hjust=0)

ggplot(pokedex, aes(speed,weight_kg,color=is_legendary))+
  geom_point()

library(FactoMineR)
datigruppo<-datigruppo[,-9]
ris.pca<-PCA(datigruppo,scale.unit=TRUE,quali.sup=9)

plot.PCA(ris.pca,axes=c(3,4),choir="var")
ris.pca
ris.pca$eig

ris.clus<-HCPC(ris.pca,description = TRUE)
ris.clus


require(Factoshiny)
res <- Factoshiny(datigruppo[,-9])

res.PCA<-PCA(datigruppo[, -9],ncp=Inf, scale.unit=FALSE,graph=FALSE)
res.HCPC<-HCPC(res.PCA,nb.clust=3,consol=FALSE,graph=FALSE,metric='manhattan')
plot.HCPC(res.HCPC,choice='tree',title='Hierarchical tree')
plot.HCPC(res.HCPC,choice='map',draw.tree=FALSE,title='Factor map')
plot.HCPC(res.HCPC,choice='3D.map',ind.names=FALSE,centers.plot=FALSE,angle=60,title='Hierarchical tree on the factor map')
gruppi.HCPC<-res.HCPC$call$X$clust
datigruppo[c(25,29,67,18,65,58),"nomi"]

datigruppo<-datigruppo  %>%
  mutate(gruppi=gruppi.HCPC)

prop.table(table(datigruppo$gruppi))

library(tidyr)
legend_by_stats_gruppo<- datigruppo %>%
  select(nomi,gruppi,attack,sp_attack,defense,sp_defense,weight_kg,height_m,hp,speed) %>%
  gather(key=gruppi_stats, value=value, -c(nomi,gruppi))


legend_by_stats_plot_gruppo <- legend_by_stats_gruppo %>% 
  ggplot(aes(x = gruppi, y = value, fill = gruppi)) +
  geom_boxplot(varwidth = TRUE) +
  facet_wrap(~gruppi_stats) +
  labs(title = "Pokemon fight statistics",
       x = "Legendary gruppi status") 
legend_by_stats_plot_gruppo

