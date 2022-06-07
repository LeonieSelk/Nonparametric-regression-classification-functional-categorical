
#This script gives examples for simulated data (training data and new data to evaluate the estimator)
#that correspond to scenarios MixC and MixR

source(file="Functions.R")


##########################################################
#Parameters:

n<-100 #number of training observations
xN<-50 #number of new x-values 
tpn<-300 #number of evaluation-points for functional observations (same for training data and new data)

pr_fun<-2 #number of relevant functional covariates (same for training data and new data)
pr_cat<-2 #number of relevant categorial covariates (same for training data and new data)
p_r<-pr_cat+pr_fun #number of relevant covariates (same for training data and new data)
pi_fun<-6 #number of irrelevant functional covariates (same for training data and new data)
pi_cat<-6 #number of irrelevant categorical covariates (same for training data and new data)
p_i<-pi_cat+pi_fun #number of irrelevant covariates (same for training data and new data)
p<-p_r+p_i
p_fun<-pr_fun+pi_fun
p_cat<-pr_cat+pi_cat

#for scenario MixC
factor<-0.3 #factor for shift of functional observations
Gtilde<-2 #number of different values of C_ij (random part in shift of functional observations)
G<-((Gtilde-1)*pr_fun+1)*(pr_cat+1) #number of classes for Y

#for scenario MixR
hn_fun<-n^{-1/(p+4)} #rule-of-thumb bandwidth for functional covariates
hn_cat<-(p+4)/log(n) #rule-of-thumb bandwidth for categorical covariates adjusted for Picard kernel and Ber(0.5)-RVs 
hn_con<-n^{-1/(p+4)} #rule-of-thumb bandwidth for continuous covariates






####################################################
#function to generate functional observations:

simfx1 <- function(n, p, tps, bx=5, mx=2*pi){ 
  fx <- list()
  for (j in 1:p)  {
    tmax <- max(tps[[j]])
    fx[[j]] <- matrix(0, nrow=n, ncol=length(tps[[j]]))
    for (i in 1:n){
      bij <- runif(5, 0, bx)
      mij <- runif(5, 0, mx)
      tfx <- function(tp){
        (sum(bij*sin(tp*(5-bij)*(2*pi/tmax)) - mij) + 15)/100}
      fx[[j]][i,] <- sapply(tps[[j]],tfx)
    }}
  fx <- lapply(fx, scale)
  for (j in 1:p){fx[[j]] <- fx[[j]]/10}
  return(fx)}





####################################################################
####################################################################
#Classification / Scenario MixC:
####################################################################


##################################################
#Generating observations:


###################
#Training data:


#functional observations:

tps <- list() 
if(p_fun>0){for(j in 1:p_fun){tps[[j]]<- 1:tpn}}

if(p_fun>0){Xfun <- simfx1(n = n, p = p_fun, tps = tps)} #list (length p_fun) of nx(tpn) matrices

Ymatrix<-matrix(0,nrow=n,ncol=pr_fun)

Xfunrel<-list()
if(pr_fun>0){
    for(j in 1:pr_fun){
      Cj<-sample(0:(Gtilde-1),n,replace=TRUE)
      Xfun[[j]]<-Xfun[[j]]+factor*Cj
      Xfunrel[[j]]<-Xfun[[j]]; Ymatrix[,j]<-Cj}
    if(p_fun>pr_fun){
      for(j in (pr_fun+1):p_fun){
        Cj<-sample(0:(Gtilde-1),n,replace=TRUE)
        Xfun[[j]]<-Xfun[[j]]+factor*Cj}} 
  }

X<-list()
if(p_fun>0){for(j in 1:p_fun){X[[j]]<-Xfun[[j]]}} #list for all covariates (fun, cat), length p



#categorical observations:

if(p_cat>0){
  Xcat<-t(as.matrix(rbinom(n,size=1,prob=0.5)))
  if(p_cat>1){
    for(j in 2:p_cat){
      Xcat<-rbind(Xcat,rbinom(n,size=1,prob=0.5))}} #(p_cat)xn matrix
  for(j in 1:p_cat){X[[p_fun+j]]<-Xcat[j,]}
}

if(pr_cat>0){Xcatrel<-Xcat[1:pr_cat,]}




#classification:

if(pr_fun>0){Y_fun<-rowSums(Ymatrix)+1}else{Y_fun<-1}

if(pr_cat>0){
    if(pr_cat>1){Y_cat<-colSums(Xcatrel)+1
      }else{Y_cat<-Xcatrel+1}
  }else{Y_cat<-1}

Y<-(pr_cat+1)*(Y_fun-1)+Y_cat

  



########################################
#new data to evaluate the estimator:


#functional observations:

tps <- list() 
if(p_fun>0){for(j in 1:p_fun){tps[[j]]<- 1:tpn}}

if(p_fun>0){xfun <- simfx1(n = xN, p = p_fun, tps = tps)} #list (length p_fun) of (xN)x(tpn) matrices

ymatrix<-matrix(0,nrow=xN,ncol=pr_fun)

xfunrel<-list()
if(pr_fun>0){
  for(j in 1:pr_fun){
    Cj<-sample(0:(Gtilde-1),xN,replace=TRUE)
    xfun[[j]]<-xfun[[j]]+factor*Cj 
    xfunrel[[j]]<-xfun[[j]]; ymatrix[,j]<-Cj}
  if(p_fun>pr_fun){
    for(j in (pr_fun+1):p_fun){
      Cj<-sample(0:(Gtilde-1),xN,replace=TRUE)
      xfun[[j]]<-xfun[[j]]+factor*Cj}} 
}

x<-list()
if(p_fun>0){for(j in 1:p_fun){x[[j]]<-xfun[[j]]}} #list for all covariates (fun, cat), length p



#categorical observations:

if(p_cat>0){
  xcat<-t(as.matrix(rbinom(xN,size=1,prob=0.5)))
  if(p_cat>1){
    for(j in 2:p_cat){
      xcat<-rbind(xcat,rbinom(xN,size=1,prob=0.5))}} #(p_cat)x(xN) matrix
  for(j in 1:p_cat){x[[p_fun+j]]<-xcat[j,]}
}

if(pr_cat>0){xcatrel<-xcat[1:pr_cat,]}



#classification:

if(pr_fun>0){y_fun<-rowSums(ymatrix)+1}else{y_fun<-1}

if(pr_cat>0){
  if(pr_cat>1){y_cat<-colSums(xcatrel)+1
     }else{y_cat<-xcatrel+1}
  }else{y_cat<-1}

y<-(pr_cat+1)*(y_fun-1)+y_cat




#####################################################################
#apply weight optimization and estimator:

w<-opt_weights_c(X,p_fun,p_cat,0,Y,G)

est_res<-predict_c(x,w,X,p_fun,p_cat,0,Y,G)


#visualization of results:

barplot(c(w)/sum(w), ylab="Weights", main="Sparse classification model, mixed covariates",ylim=c(0,1.2))
if(pr_fun>0 && pi_fun>0){
  abline(v=(1.2*pr_fun+0.1),col="gray64")}
if(pr_cat>0 && pi_cat>0){
  abline(v=(1.2*(p_fun+pr_cat)+0.1),col="gray64")}
if(p_fun>0 && p_cat>0){
  abline(v=(1.2*p_fun+0.1))}
if(pr_fun>0){
  axis(1,at=(1.2*pr_fun/2),las=2,labels="Relevant\npredictor(s)",tick=FALSE,line=-0.8)}
if(pi_fun>0){
  axis(1,at=(1.2*(pr_fun+pi_fun/2)),las=2,labels="Noise",tick=FALSE,line=-0.8)}
if(pr_cat>0){
  axis(1,at=(1.2*(p_fun+pr_cat/2)),las=2,labels="Relevant\npredictor(s)",tick=FALSE,line=-0.8)}
if(pi_cat>0){
  axis(1,at=(1.2*(p_fun+pr_cat+pi_cat/2)),las=2,labels="Noise",tick=FALSE,line=-0.8)}
if(p_fun>0){
  if(pr_fun>0 && pi_fun>0){
    axis(3,at=(1.2*pr_fun+0.1),labels="Functional",tick=FALSE, line=-1)
  }else{
    axis(3,at=(1.2*p_fun/2),labels="Functional",tick=FALSE, line=-1)
    }}
if(p_cat>0){
  if(pr_cat>0 && pi_cat>0){
    axis(3,at=(1.2*(p_fun+pr_cat)+0.1),labels="Categorical",tick=FALSE, line=-1)
  }else{
    axis(3,at=(1.2*(p_fun+p_cat/2)),labels="Categorical",tick=FALSE, line=-1)
  }}



miss_class<-sum(y!=est_res$Ghat)/xN
print(paste("Missclassification rate:", miss_class))


Iy<-c()
for(i in 1:xN){for(g in 1:G){Iy<-append(Iy,as.integer(y[i]==g))}}
boxplot((Iy-est_res$Phat)^2,ylab="Squared estimation error", xlab="", main="Sparse classification model, \nmixed covariates")












#########################################################################
#########################################################################
#Regression / Scenario MixR:
#########################################################################


#################################################
#Generating observations:


####################
#Training data:


#functional observations:

tps <- list() 
if(p_fun>0){for(j in 1:p_fun){tps[[j]]<- 1:tpn}}

if(p_fun>0){Xfun <- simfx1(n = n, p = p_fun, tps = tps)} #list (length p_fun) of nx(tpn) matrices

Xfunrel<-list() 
if(pr_fun>0){for(j in 1:pr_fun){Xfunrel[[j]]<-Xfun[[j]]}}

X<-list()
if(p_fun>0){for(j in 1:p_fun){X[[j]]<-Xfun[[j]]}} #list for all covariates (fun, cat), length p



#categorical observations:

if(p_cat>0){Xcat<-t(as.matrix(rbinom(n,size=1,prob=0.5)))

if(p_cat>1){for(j in 2:p_cat){Xcat<-rbind(Xcat,rbinom(n,size=1,prob=0.5))}} #(p_cat)xn matrix

for(j in 1:p_cat){X[[p_fun+j]]<-Xcat[j,]}}

if(pr_cat>0){Xcatrel<-Xcat[1:pr_cat,]}



#building regression:

eps<-rnorm(n)

f_fun<-function(x){
  y<-matrix(0,nrow=dim(x[[1]])[1],ncol=pr_fun)
  for(j in 1:pr_fun){
    y[,j]<-x[[j]]%*%(5*dgamma((1:tpn)/10, 3, 1/3))}
  return(rowSums(y))}

f_cat<-function(x){
  if(pr_cat>1){y<-2*colSums(x)}else{y<-2*x}
  return(y)}

Y<-eps 

if(pr_fun>0){Y<-Y+f_fun(Xfunrel)}

if(pr_cat>0){Y<-Y+f_cat(Xcatrel)}






##############################################
#new data to evaluate the estimator:


#functional observations:

tps <- list() 
if(p_fun>0){for(j in 1:p_fun){tps[[j]]<- 1:tpn}}

if(p_fun>0){xfun <- simfx1(n = xN, p = p_fun, tps = tps)} #list (length p_fun) of (xN)x(tpn) matrices

xfunrel<-list()
if(pr_fun>0){for(j in 1:pr_fun){xfunrel[[j]]<-xfun[[j]]}}

x<-list()
if(p_fun>0){for(j in 1:p_fun){x[[j]]<-xfun[[j]]}} #list for all covariates (fun, cat), length p



#categorical observations:

if(p_cat>0){xcat<-t(as.matrix(rbinom(xN,size=1,prob=0.5)))

if(p_cat>1){for(j in 2:p_cat){xcat<-rbind(xcat,rbinom(xN,size=1,prob=0.5))}} #(p_cat)x(xN) matrix

for(j in 1:p_cat){x[[p_fun+j]]<-xcat[j,]}}

if(pr_cat>0){xcatrel<-xcat[1:pr_cat,]}




#building regression:

eps<-rnorm(xN)

f_fun<-function(x){
  y<-matrix(0,nrow=dim(x[[1]])[1],ncol=pr_fun)
  for(j in 1:pr_fun){
    y[,j]<-x[[j]]%*%(5*dgamma((1:tpn)/10, 3, 1/3))}
  return(rowSums(y))}

f_cat<-function(x){
  if(pr_cat>1){y<-2*colSums(x)}else{y<-2*x}
  return(y)}

y<-eps

if(pr_fun>0){y<-y+f_fun(xfunrel)}

if(pr_cat>0){y<-y+f_cat(xcatrel)}








#########################################################################
#apply weight optimization and estimator:

w<-opt_weights_r(X,p_fun,p_cat,0,Y,hn_fun,hn_cat,hn_con)

est_res<-predict_r(x,w,X,p_fun,p_cat,0,Y,hn_fun,hn_cat,hn_con)


#visualization of results:

barplot(w/sum(w), ylab="Weights", main="Sparse regression model, mixed covariates",ylim=c(0,1.2))
if(pr_fun>0 && pi_fun>0){
  abline(v=(1.2*pr_fun+0.1),col="gray64")}
if(pr_cat>0 && pi_cat>0){
  abline(v=(1.2*(p_fun+pr_cat)+0.1),col="gray64")}
if(p_fun>0 && p_cat>0){
  abline(v=(1.2*p_fun+0.1))}
if(pr_fun>0){
  axis(1,at=(1.2*pr_fun/2),las=2,labels="Relevant\npredictor(s)",tick=FALSE,line=-0.8)}
if(pi_fun>0){
  axis(1,at=(1.2*(pr_fun+pi_fun/2)),las=2,labels="Noise",tick=FALSE,line=-0.8)}
if(pr_cat>0){
  axis(1,at=(1.2*(p_fun+pr_cat/2)),las=2,labels="Relevant\npredictor(s)",tick=FALSE,line=-0.8)}
if(pi_cat>0){
  axis(1,at=(1.2*(p_fun+pr_cat+pi_cat/2)),las=2,labels="Noise",tick=FALSE,line=-0.8)}
if(p_fun>0){
  if(pr_fun>0 && pi_fun>0){
    axis(3,at=(1.2*pr_fun+0.1),labels="Functional",tick=FALSE, line=-1)
  }else{
    axis(3,at=(1.2*p_fun/2),labels="Functional",tick=FALSE, line=-1)
  }}
if(p_cat>0){
  if(pr_cat>0 && pi_cat>0){
    axis(3,at=(1.2*(p_fun+pr_cat)+0.1),labels="Categorical",tick=FALSE, line=-1)
  }else{
    axis(3,at=(1.2*(p_fun+p_cat/2)),labels="Categorical",tick=FALSE, line=-1)
  }}



boxplot((y-est_res)^2/(max(y)-min(y))^2,ylab="Squared estimation error", xlab="", main="Sparse regression model, \nmixed covariates")

