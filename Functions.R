
library(matrixStats)


#This script contains the functions to determine the optimal weights 
#and estimate the posterior probability / regression function



######################################################
#Classification
######################################################

#opt-weights_c is a function that returns the minimizing weights 
#and expects the following input:
# X: list with covariates (training data), where
#     X[[1]],...,X[[p_fun]] contain the functional covariates as matrices of size (number of observations)x(number of evaluation points)
#     X[[p_fun+1]],...,X[[p_fun+p_cat]] contain the categorical covariates 
#     X[[p_fun+p_cat+1]],...,X[[p_fun+p_cat+p_con]] contain the continuous covariates 
# p_fun: number of functional covariates 
# p_cat: number of categorical covariates 
# p_con: number of continuous covariates 
# Y: vector with classification results (training data)
# G: number of classes for Y 



opt_weights_c<-function(X,p_fun,p_cat,p_con,Y,G){

  
#Kernel estimation:
# kern: kernel function (one-sided Picard-kernel))
# d: distances: normed L^2-distance for functional covariates, normed euclidean distance for categorical and continuous covariates
# dcv: pre-calculation of distances for all combinations of covariates (preparation for leave-one-out estimation)
# phatm: leave-one-out estimator
  
kern<-function(x){t(exp(-x))}

d<-function(x,Z){
  if(p_fun>0){y<-colSums((c(x[[1]])-t(Z[[1]]))^2)/sum(colVars(Z[[1]]))}
  if(p_fun>1){
    for(i in 2:p_fun){y<-rbind(y,colSums((c(x[[i]])-t(Z[[i]]))^2)/sum(colVars(Z[[i]])))}}
  if(p_cat>0){
    if(p_fun>0){y<-rbind(y,(x[[p_fun+1]]-Z[[p_fun+1]])^2/var(Z[[p_fun+1]]))
      }else{y<-1/hn_cat^2*(x[[1]]-Z[[1]])^2/var(Z[[1]])}
    }
  if(p_cat>1){
    for(i in 2:p_cat){y<-rbind(y,(x[[p_fun+i]]-Z[[p_fun+i]])^2/var(Z[[p_fun+i]]))}}
  if(p_con>0){
    if(p_fun>0||p_cat>0){y<-rbind(y,(x[[p_fun+p_cat+1]]-Z[[p_fun+p_cat+1]])^2/var(Z[[p_fun+p_cat+1]]))
      }else{y<-1/hn_con^2*(x[[1]]-Z[[1]])^2/var(Z[[1]])}
    }
  if(p_con>1){
    for(i in 2:p_con){y<-rbind(y,(x[[p_fun+p_cat+i]]-Z[[p_fun+p_cat+i]])^2/var(Z[[p_fun+p_cat+i]]))}}
  return(sqrt(y))} 

dcv<-list() 
for(i in 1:n){
  xdcv<-list();Xdcv<-list()
  if(p_fun>0){
    for(j in 1:p_fun){xdcv[[j]]<-X[[j]][i,];Xdcv[[j]]<-X[[j]][-i,]}}
  if(p_cat>0){
    for (j in (p_fun+1):(p_fun+p_cat)){xdcv[[j]]<-X[[j]][i];Xdcv[[j]]<-X[[j]][-i]}}
  if(p_con>0){
    for (j in (p_fun+p_cat+1):(p_fun+p_cat+p_con)){xdcv[[j]]<-X[[j]][i];Xdcv[[j]]<-X[[j]][-i]}}
  dcv[[i]]<-d(xdcv,Xdcv)} 

phatm<-function(g,i,w){
  if(sum(kern(t(w^2)%*%dcv[[i]]))!=0){
      return((t(as.integer(Y[-i]==g))%*%kern(t(w^2)%*%dcv[[i]]))/sum(kern(t(w^2)%*%dcv[[i]])))
    }else{return(0)}}
  


#Loss-function:
# Q: quadratic loss (Brier Score)

Q<-function(w){
  y<-0
  for (i in 1:n){y<-y+sum((as.integer(Y[i]==(1:G))-sapply((1:G),function(x){phatm(x,i,w)}))^2)}
  return(y)}




#Pre-Estimation of weights:

pre_w<-matrix(0,nrow=1,ncol=p)

for(j in 1:p){
  
#Loss-function for Pre-Estimation:
  
pre_phatm<-function(g,i,w){
  if(sum(kern(t(w^2)%*%dcv[[i]][j,]))!=0){
      return((t(as.integer(Y[-i]==g))%*%kern(t(w^2)%*%dcv[[i]][j,]))/sum(kern(t(w^2)%*%dcv[[i]][j,])))
    }else{return(0)}}

pre_Q<-function(w){
  y<-0
  for (i in 1:n){y<-y+sum((as.integer(Y[i]==(1:G))-sapply((1:G),function(x){pre_phatm(x,i,w)}))^2)}
  return(y)}


#Optimization for Pre-Estimation:

pre_w[j]<-optimize(pre_Q,c(0,2*1/(n^{-1/5})))$minimum

}




#Optimization: determine minimizing weights:

wopt<-optim(pre_w,Q)$par



return(wopt^2)}





############################################################

#predict_c is a function that returns predicted classes and posterior probabilities 
#and expects the following input:
#
# w: weights for the kernel estimator (e.g. the return of opt_weights_c)
# x: list with covariates (new data), where
#     x[[1]],...,x[[p_fun]] contain the functional covariates as matrices of size (number of observations)x(number of evaluation points)
#     x[[p_fun+1]],...,x[[p_fun+p_cat]] contain the categorical covariates 
#     x[[p_fun+p_cat+1]],...,x[[p_fun+p_cat+p_con]] contain the continuous covariates 
# X: list with covariates (training data), where
#     X[[1]],...,X[[p_fun]] contain the functional covariates as matrices of size (number of observations)x(number of evaluation points)
#     X[[p_fun+1]],...,X[[p_fun+p_cat]] contain the categorical covariates 
#     X[[p_fun+p_cat+1]],...,X[[p_fun+p_cat+p_con]] contain the continuous covariates 
# p_fun: number of functional covariates 
# p_cat: number of categorical covariates 
# p_con: number of continuous covariates 
# Y: vector with classification results (training data)
# G: number of classes for Y
#
#The output contains
# Phat: estimated posterior probabilities as a vector 
#       where Phat[1],...,Phat[G] contain the probabilities for the first observation of the new data,
#       Phat[G+1],...,Phat[G+G] the probability for the second observation of the new data, etc
# Ghat: predicted classes as a vector of length (number of new data observations)



predict_c<-function(x,w,X,p_fun,p_cat,p_con,Y,G){
  

#Kernel estimation:  
# kern: kernel function (one-sided Picard-kernel))
# d: distances: normed L^2-distance for functional covariates, normed euclidean distance for categorical and continuous covariates
# phat: estimator for posterior probability
  
kern<-function(z){t(exp(-z))}

d<-function(z,Z){
  if(p_fun>0){y<-colSums((c(z[[1]])-t(Z[[1]]))^2)/sum(colVars(Z[[1]]))}
  if(p_fun>1){
    for(i in 2:p_fun){y<-rbind(y,colSums((c(z[[i]])-t(Z[[i]]))^2)/sum(colVars(Z[[i]])))}}
  if(p_cat>0){
    if(p_fun>0){y<-rbind(y,(z[[p_fun+1]]-Z[[p_fun+1]])^2/var(Z[[p_fun+1]]))
      }else{y<-1/hn_cat^2*(z[[1]]-Z[[1]])^2/var(Z[[1]])}}
  if(p_cat>1){
    for(i in 2:p_cat){y<-rbind(y,(z[[p_fun+i]]-Z[[p_fun+i]])^2/var(Z[[p_fun+i]]))}}
  if(p_con>0){
    if(p_fun>0||p_cat>0){y<-rbind(y,(z[[p_fun+p_cat+1]]-Z[[p_fun+p_cat+1]])^2/var(Z[[p_fun+p_cat+1]]))
      }else{y<-1/hn_con^2*(z[[1]]-Z[[1]])^2/var(Z[[1]])}}
  if(p_con>1){
    for(i in 2:p_con){y<-rbind(y,(z[[p_fun+p_cat+i]]-Z[[p_fun+p_cat+i]])^2/var(Z[[p_fun+p_cat+i]]))}}
  return(sqrt(y))} 

phat<-function(z,g,v){
  if(sum(kern(v%*%d(z,X)))!=0){
      return((t(as.integer(Y==g))%*%kern(v%*%d(z,X)))/sum(kern(v%*%d(z,X))))
    }else{return(0)}}



#Evaluation of phat with new data:
# Phat: estimated posterior probabilities
# Ghat: predicted classes

Phat<-c(); Ghat<-c()

for(i in 1:(dim(x[[1]])[1])){
  xval<-list()
  if(p_fun>0){for(j in 1:p_fun){xval[[j]]<-x[[j]][i,]}}
  if(p_cat>0){for(j in 1:p_cat){xval[[p_fun+j]]<-x[[p_fun+j]][i]}}
  if(p_con>0){for(j in 1:p_con){xval[[p_fun+p_cat+j]]<-x[[p_fun+p_cat+j]][i]}}
  for(g in 1:G){
    Phat<-append(Phat,phat(xval,g,w))
  }
  Ghat<-append(Ghat,which.max(Phat[((G*(i-1)+1):(G*(i-1)+G))]))
}



return(list("Phat"=Phat,"Ghat"=Ghat))}










######################################################
#Regression
######################################################

#opt-weights_r is a function that returns the minimizing weights 
#and expects the following input:
# X: list with covariates (training data), where
#     X[[1]],...,X[[p_fun]] contain the functional covariates as matrices of size (number of observations)x(number of evaluation points)
#     X[[p_fun+1]],...,X[[p_fun+p_cat]] contain the categorical covariates 
#     X[[p_fun+p_cat+1]],...,X[[p_fun+p_cat+p_con]] contain the continuous covariates 
# p_fun: number of functional covariates 
# p_cat: number of categorical covariates 
# p_con: number of continuous covariates 
# Y: vector with classification results (training data)
# hn_fun: rule-of-thumb bandwidth for functional covariates
# hn_cat: rule-of-thumb bandwidth for categorical covariates
# hn_con: rule-of-thumb bandwidth for continuous covariates



opt_weights_r<-function(X,p_fun,p_cat,p_con,Y,hn_fun,hn_cat,hn_con){
  

#Kernel estimation:
# kern: kernel function (one-sided Picard-kernel))
# d: distances: normed L^2-distance for functional covariates, normed euclidean distance for categorical and continuous covariates
# dcv: pre-calculation of distances for all combinations of covariates (preparation for leave-one-out estimation)
# yhatm: leave-one-out estimator
  
kern<-function(x){t(exp(-x))}

d<-function(x,Z){
  if(p_fun>0){y<-1/hn_fun^2*colSums((c(x[[1]])-t(Z[[1]]))^2)/sum(colVars(Z[[1]]))}
  if(p_fun>1){
    for(i in 2:p_fun){y<-rbind(y,1/hn_fun^2*colSums((c(x[[i]])-t(Z[[i]]))^2)/sum(colVars(Z[[i]])))}}
  if(p_cat>0){
    if(p_fun>0){y<-rbind(y,1/hn_cat^2*(x[[p_fun+1]]-Z[[p_fun+1]])^2/var(Z[[p_fun+1]]))
      }else{y<-1/hn_cat^2*(x[[1]]-Z[[1]])^2/var(Z[[1]])}}
  if(p_cat>1){
    for(i in 2:p_cat){y<-rbind(y,1/hn_cat^2*(x[[p_fun+i]]-Z[[p_fun+i]])^2/var(Z[[p_fun+i]]))}}
  if(p_con>0){
    if(p_fun>0||p_cat>0){y<-rbind(y,1/hn_con^2*(x[[p_fun+p_cat+1]]-Z[[p_fun+p_cat+1]])^2/var(Z[[p_fun+p_cat+1]]))
      }else{y<-1/hn_con^2*(x[[1]]-Z[[1]])^2/var(Z[[1]])}}
  if(p_con>1){
    for(i in 2:p_con){y<-rbind(y,1/hn_con^2*(x[[p_fun+p_cat+i]]-Z[[p_fun+p_cat+i]])^2/var(Z[[p_fun+p_cat+i]]))}}
  return(sqrt(y))}

dcv<-list()
for(i in 1:n){
  xdcv<-list();Xdcv<-list()
  if(p_fun>0){
    for(j in 1:p_fun){xdcv[[j]]<-X[[j]][i,];Xdcv[[j]]<-X[[j]][-i,]}}
  if(p_cat>0){
    for (j in (p_fun+1):(p_fun+p_cat)){xdcv[[j]]<-X[[j]][i];Xdcv[[j]]<-X[[j]][-i]}}
  if(p_con>0){
    for (j in (p_fun+p_cat+1):(p_fun+p_cat+p_con)){xdcv[[j]]<-X[[j]][i];Xdcv[[j]]<-X[[j]][-i]}}
  dcv[[i]]<-d(xdcv,Xdcv)} 

yhatm<-function(i,w){
  if(sum(kern(t(w^2)%*%dcv[[i]]))!=0){
      return((t(Y[-i])%*%kern(t(w^2)%*%dcv[[i]]))/sum(kern(t(w^2)%*%dcv[[i]])))
    }else{return(0)}}
  


#Loss-function:
# Q: quadratic loss (Brier Score)

Q<-function(w){
  y<-0
  for (i in 1:n){y<-y+(Y[i]-yhatm(i,w))^2}
  return(y)}  



#Optimization: determine minimizing weights

wopt<-optim(rep.int(1,p),Q)$par

  


return(wopt^2)}








######################################################

#predict_r is a function that returns predicted values of the regression function 
#and expects the following input:
# w: weights for the kernel estimator (e.g. the return of opt_weights_r)
# x: list with covariates (new data), where
#     x[[1]],...,x[[p_fun]] contain the functional covariates as matrices of size (number of observations)x(number of evaluation points)
#     x[[p_fun+1]],...,x[[p_fun+p_cat]] contain the categorical covariates 
#     x[[p_fun+p_cat+1]],...,x[[p_fun+p_cat+p_con]] contain the continuous covariates 
# X: list with covariates (training data), where
#     X[[1]],...,X[[p_fun]] contain the functional covariates as matrices of size (number of observations)x(number of evaluation points)
#     X[[p_fun+1]],...,X[[p_fun+p_cat]] contain the categorical covariates 
#     X[[p_fun+p_cat+1]],...,X[[p_fun+p_cat+p_con]] contain the continuous covariates 
# p_fun: number of functional covariates 
# p_cat: number of categorical covariates 
# p_con: number of continuous covariates 
# Y: vector with classification results (training data)
# hn_fun: rule-of-thumb bandwidth for functional covariates
# hn_cat: rule-of-thumb bandwidth for categorical covariates
# hn_con: rule-of-thumb bandwidth for continuous covariates




predict_r<-function(x,w,X,p_fun,p_cat,p_con,Y,hn_fun,hn_cat,hn_con){
  
  
#Kernel estimation:  
# kern: kernel function (one-sided Picard-kernel))
# d: distances: normed L^2-distance for functional covariates, normed euclidean distance for categorical and continuous covariates
# yhat: regression estimator
  
kern<-function(z){t(exp(-z))}

d<-function(z,Z){
  if(p_fun>0){y<-1/hn_fun^2*colSums((c(z[[1]])-t(Z[[1]]))^2)/sum(colVars(Z[[1]]))}
  if(p_fun>1){
    for(i in 2:p_fun){y<-rbind(y,1/hn_fun^2*colSums((c(z[[i]])-t(Z[[i]]))^2)/sum(colVars(Z[[i]])))}}
  if(p_cat>0){
    if(p_fun>0){y<-rbind(y,1/hn_cat^2*(z[[p_fun+1]]-Z[[p_fun+1]])^2/var(Z[[p_fun+1]]))
      }else{y<-1/hn_cat^2*(z[[1]]-Z[[1]])^2/var(Z[[1]])}}
  if(p_cat>1){
    for(i in 2:p_cat){y<-rbind(y,1/hn_cat^2*(z[[p_fun+i]]-Z[[p_fun+i]])^2/var(Z[[p_fun+i]]))}}
  if(p_con>0){
    if(p_fun>0||p_cat>0){y<-rbind(y,1/hn_con^2*(z[[p_fun+p_cat+1]]-Z[[p_fun+p_cat+1]])^2/var(Z[[p_fun+p_cat+1]]))
      }else{y<-1/hn_con^2*(z[[1]]-Z[[1]])^2/var(Z[[1]])}}
  if(p_con>1){
    for(i in 2:p_con){y<-rbind(y,1/hn_con^2*(z[[p_fun+p_cat+i]]-Z[[p_fun+p_cat+i]])^2/var(Z[[p_fun+p_cat+i]]))}}
  return(sqrt(y))}

yhat<-function(z,v){
  if(sum(kern(v%*%d(z,X)))!=0){
      return((t(Y)%*%kern(v%*%d(z,X)))/sum(kern(v%*%d(z,X))))
    }else{return(0)}}

  


#Evaluation of yhat with new data
# Yhat: predicted values of the regression function

Yhat<-c()
for(i in 1:(dim(x[[1]])[1])){
  xval<-list()
  if(p_fun>0){
    for(j in 1:p_fun){xval[[j]]<-x[[j]][i,]}}
  if(p_cat>0){
    for(j in 1:p_cat){xval[[p_fun+j]]<-x[[p_fun+j]][i]}}
  if(p_con>0){
    for(j in 1:p_con){xval[[p_fun+p_cat+j]]<-x[[p_fun+p_cat+j]][i]}}
  Yhat<-append(Yhat,yhat(xval,w))
}
  

return(Yhat)}

