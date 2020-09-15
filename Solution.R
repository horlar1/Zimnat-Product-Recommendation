options(warn = -1)

library(lightgbm)
library(plyr)
library(tidyverse)
library(caret)
library(SOAR)
library(data.table)
library(lubridate)
library(catboost)
library(xgboost)
#####
path.dir = getwd()
data.dir = paste0(path.dir,"/Data")

#dir.create(paste0(path.dir,"/tmp"))
save.files.dir = paste0(path.dir,"/tmp")
subm.dir = paste0(path.dir,"/subm")
source("utils.R")


####
df = fread(paste0(data.dir,"/train.csv")) %>% as.data.frame()
test = fread(paste0(data.dir,"/test.csv")) %>% as.data.frame()
samp = fread(paste0(data.dir,"/SampleSubmission.csv"))

###
train.id = df$ID
test.id = samp$`ID X PCODE`
target = df %>% select(P5DA:ECY3)
target2 = test %>% select(P5DA:ECY3)

# Store(train.id)
# Store(test.id)
# Store(target)

## baseline features
drop.cols = c('ID')
df = bind_rows(df %>% select(-tidyselect::all_of(drop.cols)),test %>% select(-tidyselect::all_of(drop.cols)))

df = df %>% mutate(
  join_date = as.Date(join_date, format = '%d/%m/%Y'),
  join_date = ymd(join_date),
  join_year = year(join_date),
  no_of_days = as.numeric(Sys.Date() - join_date),
  join_date = NULL,
  Sex_M = ifelse(sex== 'M',1,0),
  Age = year(Sys.Date()) - birth_year,
  branch_code = as.numeric(as.factor(branch_code)),
  occupation_code = as.numeric(as.factor(occupation_code)),
  occupation_category_code=as.numeric(as.factor(occupation_category_code)),
  marital_status = as.numeric(as.factor(marital_status)),
  occupation_cat_code_Mult = occupation_category_code*occupation_code,
  birth_year = NULL,
  sex =NULL
) %>% 
  select(-c(P5DA:ECY3))


#### EXTRA FEATURE ENGINEERING
### TWO WAY COUNT FEATURE
df$branchcode_occupation_cnt = my.f2cnt(df,"branch_code","occupation_code")
df$year_branch_Cnt = my.f2cnt(df,"join_year","branch_code")
df$occupation_cat_cnt = my.f2cnt(df,"occupation_category_code","occupation_code")

#####
df = df %>% group_by(occupation_code) %>% 
  mutate(mean_noofdays = mean(no_of_days,na.rm = T)) %>%
  ungroup() 

#### entropy feature
branch_occupation = calc_entropy(df,"branch_code","occupation_code","branch_occupation")
df = df %>% left_join(branch_occupation, by="branch_code")



########
##
#######
df_train = df[1:length(train.id),]
df_test = df[(length(train.id)+1):nrow(df),]

#### MORE FEATURES
###
df_train$product_sum = rowSums(target)
df_test$product_sum = rowSums(test %>% select(P5DA:ECY3) %>% as.matrix())+1
######
df = rbind(df_train,df_test)
df = df %>% group_by(branch_code,occupation_code) %>%
            mutate(mean_product_sum = mean(product_sum,na.rm = T)) %>% 
            ungroup()
#######
df_train = df[1:length(train.id),]
df_test = df[(length(train.id)+1):nrow(df),]


####### MAKE TRAIN DATA SIMILAR TO TEST DATA
### takes some time to run- 10 MINS
pp = data.frame()
tp = data.frame()
cc = c()
m = df_train
start.time = Sys.time()
for(i in 1:nrow(m)){
  tt = which(target[i,] == 1)
  m2 = matrix(unlist(rep(m[i,],length(tt))), nrow = length(tt), byrow = T)
  colnames(m2) = colnames(m)
  t2 = matrix(unlist(rep(target[i,], length(tt))),nrow = length(tt), byrow=T) 
  colnames(t2)=  colnames(target)
  t3 = colnames(t2[,tt])
  for(j in 1:nrow(t2)){
    t2[j,tt[j]] = 0
  }
  pp = rbind(pp,m2)
  tp = rbind(tp,t2)
  cc = c(cc,t3)
}
Sys.time() - start.time
#########################
df_train = pp
new_target = tp
new_label = cc


######### PRODUCT K-MEANS CLUSTERING
dr.dat = rbind(tp,target2)
sumsq = NULL
for (i in 1:9) {
  set.seed(1234)
  sumsq[i] = sum(kmeans(dr.dat,centers = i, iter.max = 1000,
    algorithm = "Forgy")$withinss)
}
plot(1:9,sumsq,type= "b")
###
set.seed(1234)
kmns = kmeans(dr.dat,3,iter.max = 1000,
  algorithm = "Forgy",trace = T)


######
df_train$RVSZ_K6QO_QBOL_sum = rowSums(new_target %>% select(RVSZ,K6QO,QBOL))
df_test$RVSZ_K6QO_QBOL_sum = rowSums(target2 %>% select(RVSZ,K6QO,QBOL))

#####
df_train$product_cumm_sum  = rowSums(matrixStats::rowCumsums(new_target %>% as.matrix()))
df_test$product_cumm_sum = rowSums(matrixStats::rowCumsums(target2 %>% select(P5DA:ECY3) %>% as.matrix()))
######
df_train$product_cluster = kmns$cluster[1:66353]
df_test$product_cluster = kmns$cluster[66354:76353]

### 
df_train = cbind(df_train,new_target)
df_test = cbind(df_test,target2)

#########################################
############# DON'T RUN
#######################################
#### TWO WAY INTERACTIONS
#############
# num.features = 
# ######
# start.time = Sys.time()
# df_train$y = label
# num.pairs = data.frame(t(combn(as.character(num.features), 2)), stringsAsFactors = F)
# num.pairs$pv = NA
# Store(num.pairs)
# for (i in 1:nrow(num.pairs)) {
#   frmla.alte=as.formula(paste('y ~', num.pairs$X1[i], '+', num.pairs$X2[i]))
#   frmla.mul=as.formula(paste('y ~', num.pairs$X1[i], '*', num.pairs$X2[i]))
#   num.pairs$pv[i] = anova(glm(frmla.alte, df_train, family = "gaussian"),
#     glm(frmla.mul, df_train, family = "gaussian"),
#     test = 'Chisq')$`Pr(>Chi)`[2]
# }
# head(num.pairs)
# gc(reset = T)
# 
# total.time = Sys.time() - start.time
# total.time
# 
# num.pairs$log = log(num.pairs$pv)
# nn = num.pairs %>% filter(log < -50)
# 
# ##########
# start.time = Sys.time()
# w = data.frame(id =c(1:66353))
# w2 = data.frame(id =c(1:10000))
# for (i in 1:nrow(nn)){
# 
#   ######
#   var1 = nn[i,1]
#   var2 = nn[i,2]
#   var = paste(var1,var2,"Mult" ,sep = "_")
#   w[,var] = as.numeric(df_train[,var1]) * as.numeric(df_train[,var2])
#   w2[,var] = as.numeric(df_test[,var1]) * as.numeric(df_test[,var2])
#   var = paste(var1,var2,"Add" ,sep = "_")
#   w[,var] = as.numeric(df_train[,var1]) + as.numeric(df_train[,var2])
#   w2[,var] = as.numeric(df_test[,var1]) + as.numeric(df_test[,var2])
# }
# rm(tmp) ;gc(reset = T)
# total.time = Sys.time() - start.time
# total.time
load(paste0(save.files.dir,"/train_2way_features.RData"))
load(paste0(save.files.dir,"/test_2way_features.RData"))


### DATASET 2
cols = c('marital_status',"branch_code","occupation_code","occupation_category_code","join_year","no_of_days","Sex_M","Age")
df_train3 = cbind(df_train %>% select(tidyselect::all_of(cols)),train_2way[,2:ncol(train_2way)],new_target)
df_test3 = cbind(df_test %>% select(tidyselect::all_of(cols)),test_2way[,2:ncol(test_2way)],target2)


########################################
###    MODELLING
########################################
label = as.numeric(as.factor(new_label)) - 1
## MODEL 1 ------- LIGHTGBM
devresult = matrix(0,nrow = nrow(df_train),ncol = 21)
predte = rep(0,210000)
cvscore = c()
int.seed = c(500)
for (i in 1) {
  cat("model training",i,"\n")
  
  set.seed(int.seed[i])
  folds = createFolds(label, k = 5)
  
  params <- list(objective = "multiclass", 
    boost="gbdt",
    metric="multi_logloss",
    boost_from_average="false",
    num_threads=8,
    learning_rate = 0.05,
    num_leaves = 20,
    max_depth=-1,
    tree_learner = "serial",
    feature_fraction = 0.8,
    bagging_freq = 1,
    bagging_fraction = 0.8,num_class=21,
    verbose = -1)
  
  
  for (this.round in 1:length(folds)) {
    cat("model training",i," ",this.round,"\n")
    valid = c(1:length(label))[unlist(folds[this.round])]
    dev = c(1:length(label))[unlist(folds[1:length(folds)!= this.round])]
    
    dtrain = lgb.Dataset(data = as.matrix(df_train[dev,]),label = label[dev])
    dvalid = lgb.Dataset(data = as.matrix(df_train[valid,]),label = label[valid])
    
    model = lgb.train(data = dtrain,
      params = params,
      nrounds = 3000,
      valids = list(val1 = dvalid),
      boosting_type = "gbdt",
      num_threads = 8,
      eval_freq =500,
      seed = 54321,
      verbose = -1,
      early_stopping_rounds = 10
    )
    
    pred = predict(model,as.matrix(df_train[valid,]))
    p2 = matrix(pred,nrow = 21,ncol = length(pred)/21) %>% t()
    devresult[valid,] = p2
    pred_test = predict(model, as.matrix(df_test[,colnames(df_train)]))
    predte = predte + pred_test
    
    cat("model cv score:", model$best_score,"\n")
    cvscore = c(cvscore, model$best_score)
    cat("model cv mean score:",mean(cvscore), "\n")
  }
}

lgb_oof = devresult
colnames(lgb_oof) = paste0("lgb1_",1:21)
pred = predte/5
lgb_pred = matrix(pred,nrow = 21,ncol = length(pred)/21) %>% t()
colnames(lgb_pred) = paste0("lgb1_",1:21)

###########################################################
## MODEL 2 ------- CATBOOST
library(catboost)
devresult = matrix(0,nrow = nrow(df_train),ncol = 21)
p = rep(0,210000)
cvscore = c()
dtest = catboost.load_pool(as.matrix(df_test[,colnames(df_train)]))
dtrain = catboost.load_pool(as.matrix(df_train),label = label)

int.seed = c(101)
i =1
for (i in 1) {
  
  set.seed(int.seed[i])
  folds = createFolds(label, k = 5)
  
  print(paste0("model training on label ", i))
  
 for(rnd in 1:length(folds)){
    valid = c(1:length(label))[unlist(folds[rnd])]
    dev = c(1:length(label))[unlist(folds[1:length(folds) != rnd])]
    
    dtrain = catboost.load_pool(as.matrix(df_train[dev,]), label=label[dev])
    dvalid = catboost.load_pool(as.matrix(df_train[valid,]),label=label[valid])
    
    params = list(
      iterations = 5000,
      learning_rate = 0.1,
      depth = 5,
      eval_metric = "MultiClass",
      loss_function = "MultiClass",
      random_seed = 54321,
      use_best_model = TRUE,
      logging_level = "Verbose",
      rsm = 0.8,
      od_type = "Iter",
      od_wait = 100,
      metric_period = 500
    )
model = catboost.train(learn_pool =dtrain,test_pool=dvalid,params=params)
    predt = catboost.predict(model,
      pool = catboost.load_pool(as.matrix(df_train[valid,])),                       prediction_type = "Probability")
    devresult[valid,] = predt
    ###
pred = catboost.predict(model, pool = dtest,prediction_type = "Probability")
p = p + pred

  }   
}
cat_oof = devresult
colnames(cat_oof) = paste0("cat_",1:21)
cat_pred = p/5
colnames(cat_pred) = paste0("cat_",1:21)

#################################################################
## MODEL 3 ------- LIGHTGBM ON DATASET 2
devresult = matrix(0,nrow = nrow(df_train3),ncol = 21)
predte = rep(0,210000)
cvscore = c()
int.seed = c(500)

for (i in 1) {
  cat("model training",i,"\n")
  
  set.seed(int.seed[i])
  folds = createFolds(label, k = 5)
  
  params <- list(objective = "multiclass", 
    boost="gbdt",
    metric="multi_logloss",
    boost_from_average="false",
    num_threads=8,
    learning_rate = 0.1,
    num_leaves = 50,
    max_depth=-1,
    tree_learner = "serial",
    feature_fraction = 0.8,
    bagging_freq = 1,
    bagging_fraction = 0.8,num_class=21,
    verbose = -1)
  
  
  for (this.round in 1:length(folds)) {
    cat("model training",i," ",this.round,"\n")
    valid = c(1:length(label))[unlist(folds[this.round])]
    dev = c(1:length(label))[unlist(folds[1:length(folds)!= this.round])]
    
    dtrain = lgb.Dataset(data = as.matrix(df_train3[dev,]),label = label[dev])
    dvalid = lgb.Dataset(data = as.matrix(df_train3[valid,]),label = label[valid])
    
    model = lgb.train(data = dtrain,
      params = params,
      nrounds = 3000,
      valids = list(val1 = dvalid),
      boosting_type = "gbdt",
      num_threads = 8,
      eval_freq =500,
      seed = 54321,
      verbose = -1,
      early_stopping_rounds = 50
    )
    
    pred = predict(model,as.matrix(df_train3[valid,]))
    p2 = matrix(pred,nrow = 21,ncol = length(pred)/21) %>% t()
    devresult[valid,] = p2
    pred_test = predict(model, as.matrix(df_test3[,colnames(df_train3)]))
    predte = predte + pred_test
    
    cat("model cv score:", model$best_score,"\n")
    cvscore = c(cvscore, model$best_score)
    cat("model cv mean score:",mean(cvscore), "\n")
  }
}
lgb_oof2 = devresult
colnames(lgb_oof2) = paste0("lgb2_",1:21)
pred = predte/5
lgb_pred2 = matrix(pred,nrow = 21,ncol = length(pred)/21) %>% t()
colnames(lgb_pred2) = paste0("lgb2_",1:21)

##############################################################
## MODEL 4 ------- XGBOOST ON DATASET 2
devresult = matrix(0,nrow = nrow(df_train3),ncol = 21)
predte = rep(0,210000)
cvscore = c()
int.seed = c(54321)

label2 = label

for (i in 1:length(int.seed)) {
  cat("model training",i,"\n")
  
  set.seed(int.seed[i])
  folds = createFolds(label, k = 5)
  
  for (this.round in 1:length(folds)) {
    cat("model training",i," ","fold ",this.round,"\n")
    valid = c(1:length(label2))[unlist(folds[this.round])]
    dev = c(1:length(label2))[unlist(folds[1:length(folds)!= this.round])]
    
    dtrain<- xgb.DMatrix(data= as.matrix(df_train3[dev,]), label= label2[dev])
    dvalid <- xgb.DMatrix(data= as.matrix(df_train3[valid,]),label=label2[valid])
    valids <- list(val = dvalid)
    
    param = list(booster = "gbtree",
      objective = "multi:softprob",
      eval_metric = "mlogloss",
      eta = 0.05,
      colsample_bytree = 0.8,
      max_depth = 5,
      min_child_weight = 1,
      nthread = 8,
      num_class = 21,
      subsample = 0.8
    )
    
    model<- xgb.train(data = dtrain,
      params= param, 
      nrounds = 1000, 
      verbose = T, 
      list(val = dvalid) ,       
      early_stopping_rounds = 10, 
      print_every_n = 500,
      maximize = F
    )
    pred = predict(model,as.matrix(df_train3[valid,]))
    p2 = matrix(pred,nrow = 21,ncol = length(pred)/21) %>% t()
    devresult[valid,] = p2
    pred_test = predict(model, as.matrix(df_test3[,colnames(df_train3)]))
    predte = predte + pred_test
    }
}

xgb_oof = devresult
colnames(xgb_oof) = paste0("xgb_",1:21)
pred = predte/5
xgb_pred = matrix(pred,nrow = 21,ncol = length(pred)/21) %>% t()
colnames(xgb_pred) = paste0("xgb_",1:21)

##############################################################
## MODEL 5 ------- XGBOOST ON DATASET 1
devresult = matrix(0,nrow = nrow(df_train3),ncol = 21)
predte = rep(0,210000)
cvscore = c()
int.seed = c(2020)

label2 = label

for (i in 1:length(int.seed)) {
  cat("model training",i,"\n")
  
  set.seed(int.seed[i])
  folds = createFolds(label, k = 5)
  
  for (this.round in 1:length(folds)) {
    cat("model training",i," ","fold ",this.round,"\n")
    valid = c(1:length(label2))[unlist(folds[this.round])]
    dev = c(1:length(label2))[unlist(folds[1:length(folds)!= this.round])]
    
    dtrain<- xgb.DMatrix(data= as.matrix(df_train[dev,]), label= label2[dev])
    dvalid <- xgb.DMatrix(data= as.matrix(df_train[valid,]),label=label2[valid])
    valids <- list(val = dvalid)
    
    param = list(booster = "gbtree",
      objective = "multi:softprob",
      eval_metric = "mlogloss",
      eta = 0.05,
      colsample_bytree = 0.8,
      max_depth = 5,
      min_child_weight = 1,
      nthread = 8,
      num_class = 21,
      subsample = 0.8
    )
    
    model<- xgb.train(data = dtrain,
      params= param, 
      nrounds = 1000, 
      verbose = T, 
      list(val = dvalid) ,       
      early_stopping_rounds = 10, 
      print_every_n = 500,
      maximize = F
    )
    pred = predict(model,as.matrix(df_train[valid,]))
    p2 = matrix(pred,nrow = 21,ncol = length(pred)/21) %>% t()
    devresult[valid,] = p2
    pred_test = predict(model, as.matrix(df_test[,colnames(df_train)]))
    predte = predte + pred_test
  }
}

xgb_oof2 = devresult
colnames(xgb_oof2) = paste0("xgb2_",1:21)
pred = predte/5
xgb_pred2 = matrix(pred,nrow = 21,ncol = length(pred)/21) %>% t()
colnames(xgb_pred2) = paste0("xgb2_",1:21)


## FINAL MODEL
final_data = cbind(df_train,lgb_oof,cat_oof,lgb_oof2,xgb_oof,xgb_oof2)
final_test = cbind(df_test,lgb_pred,cat_pred,lgb_pred2,xgb_pred,xgb_pred2)

dtrain = xgb.DMatrix(as.matrix(final_data), label=label)
dtest = xgb.DMatrix(as.matrix(final_test,colnames(final_data)))


params = list(booster = "gbtree",
  objective = "multi:softprob",
  eval_metric = "mlogloss",
  eta = 0.05,
  colsample_bytree = 0.8,
  max_depth = 3,
  min_child_weight = 1,
  nthread = 8,
  num_class = 21,
  subsample = 0.8
)

#############
set.seed(1235)
mod.xgb = xgb.train(data = dtrain,params = params,nrounds = 179)
######
pred= predict(mod.xgb,dtest)
pred = matrix(pred,nrow = 21,ncol = length(pred)/21) %>% t()
colnames(pred) = levels(as.factor(new_label))
#######
sub = cbind(ID = test$ID,pred) %>% as.data.frame()
final_sub = gather(sub,product,label,`66FJ`:SOP4) %>% 
  mutate(ID_PCODE = paste0(ID,' ','X'," ",product),
    Label = as.numeric(label)) %>% select(ID_PCODE,Label)

samp$ID_PCODE = samp$`ID X PCODE`
final_sub = final_sub %>% left_join(samp, by = "ID_PCODE")
final_sub$pred = ifelse(final_sub$Label.y == 1,1, final_sub$Label.x)
final_sub = final_sub %>% select(`ID X PCODE`,pred)

##### FINAL OUTPUT
fwrite(final_sub,file = paste0(subm.dir,"/final.csv"),row.names = F)

### DATA OUTPUT
final_data = cbind(df_train,lgb_oof,cat_oof,lgb_oof2,xgb_oof)
final_test = cbind(df_test,lgb_pred,cat_pred,lgb_pred2,xgb_pred)

fwrite(final_data, file = paste0(data.dir,"/final_data.csv"),row.names = F)
fwrite(final_test, file = paste0(data.dir,"/final_test.csv"),row.names = F)
