#=====================
#------- 1(a) -------- 
#=====================
beta0 = -16
beta1 = 1.4
beta2 = 0.3
x1 = 5
x2 = 36
y = (exp(beta0 + beta1 * x1 + beta2 * x2))/(1 + exp(beta0 + beta1 * x1 + beta2 * x2))
y


#=====================
#------- 1(b) -------- 
#=====================
y = 0.5
x2 = 18
(log(y / (1 - y)) - beta0 - beta2 * x2)/beta1


#=====================
#------- 2(a) -------- 
#=====================
#fit the training datas from BankTrain.csv with logistic model 
#Banktrain = read.csv("/Users/mac/Desktop/Computer Science/Data Mining/Assignment/A2/BankTrain.csv")

Banktrain = read.csv("E:/ÎÄµµ/UC/318/Assignment/A2/BankTrain.csv")
glm.fit = glm(y ~ x1 + x3 , data = Banktrain, family =  binomial)
summary(glm.fit)


#=====================
#------- 2(b) -------- 
#=====================
library(boot)
set.seed(2)
#create a new function to fit the model with a single row
boot.fn = function(data, index){
  return (coef(glm(y ~ x1 + x3, data = Banktrain, family = binomial, subset = index)))
}

#estimate the standard errors for coefficients using the bootstrap
boot(data = Banktrain, statistic = boot.fn, R = 1000)

#compare to the standard errors calculating by the model
summary(glm(y ~ x1 + x3, data = Banktrain, family = binomial))


#=====================
#------- 2(c) -------- 
#=====================
library(lattice)
# parameters of boundary when threshold = 0.5
slope = -coef(glm.fit)[2] / coef(glm.fit)[3]
intercept = -coef(glm.fit)[1] / coef(glm.fit)[3]

xyplot(x3 ~ x1, data = Banktrain, groups = y, pch = c(20, 4), col = c("green", "red"), main = 'Training data and Decision boundary',
       key=list(space='top', 
                points = list(pch = c(20, 4), col = c('green', 'red')),
                text = list(lab = c('genuine banknote', 'forged banknote'))),
       panel = function(...){
         panel.xyplot(...)
         panel.abline(intercept, slope)})


#=====================
#------- 2(d) -------- 
#=====================
#predict probabilities with testing datas from BankTest.csv
Banktest = read.csv("E:/ÎÄµµ/UC/318/Assignment/A2/BankTest.csv")

glm.probs = predict(glm.fit, Banktest, type = "response")

#create a vector with the results of predictions from logistic model
glm.pred = rep("0", 412)
glm.pred[glm.probs > .5] = "1"

#createa a matrix to classify how the predictions perform
table(glm.pred, Banktest$y)
mean(glm.pred == Banktest$y) 


#=====================
#------- 2(e) -------- 
#=====================
#predict probabilities with testing datas from BankTest.csv

glm.probs.train = predict(glm.fit, Banktrain, type = "response")

#create a vector with the results of predictions from logistic model
threshold = seq(0, 1, length = 100)
error.rate = list()

for(x in threshold){
  glm.pred.train = rep("0", 960)
  glm.pred.train[glm.probs.train > x] = "1"
  error.traing = mean(glm.pred.train != Banktrain$y)
  error.rate = c(error.rate, list(error.traing)) 
}
min(unlist(error.rate))
match(c(min(unlist(error.rate))), error.rate)

plot(threshold, error.rate, pch = 20, col = 'red')

glm.probs.test = predict(glm.fit, Banktest, type = "response")
glm.pred.test = rep("0", 412)
glm.pred.test[glm.probs.test > .42] = "1"

#createa a matrix to classify how the predictions perform
table(glm.pred.test, Banktest$y)
mean(glm.pred.test == Banktest$y)



#=====================
#------- 3(a) -------- 
#=====================
#LDA model
library(MASS)
lda.fit = lda(y ~ x1 + x3, data = Banktrain)

#predict classification of training data and training error
lda.class.train = predict(lda.fit, Banktrain)$class
mean(lda.class.train != Banktrain$y)
                                                                                                          
#predict classification of test data and test error
lda.class.test = predict(lda.fit, Banktest)$class
mean(lda.class.test != Banktest$y)


#=====================
#------- 3(b) -------- 
#=====================
#ODA model
qda.fit = qda(y ~ x1 + x3, data = Banktrain)

#predict classification of training data and training error
qda.class.train = predict(qda.fit, Banktrain)$class
mean(qda.class.train != Banktrain$y)

#predict classification of test data and test error
qda.class.test = predict(qda.fit, Banktest)$class
mean(qda.class.test != Banktest$y)


#=====================
#------- 3(c) -------- 
#=====================
#Compute training  error and test error of logistic model

#train error
glm.probs.train1 = predict(glm.fit, Banktrain, type = "response")
glm.pred.train1 = rep("0", 960)
glm.pred.train1[glm.probs.train1 > .5] = "1"
mean(glm.pred.train1 != Banktrain$y)

#test error
glm.probs.test1 = predict(glm.fit, Banktest, type = "response")
glm.pred.test1 = rep("0", 412)
glm.pred.test1[glm.probs.test1 > .5] = "1"
mean(glm.pred.test1 != Banktest$y)

sd(Banktrain$x1)^2
sd(Banktrain$x3)^2


#=====================
#-------   4   -------
#=====================
fx = function(x, pi0, pi1){return (pi0 * dnorm(x, 0, 2) - pi1 * dnorm(x, 2, 2))}
root = uniroot(fx, c(-5, 5), pi0 = 0.4, pi1 = 0.6, tol = 0.0001)
boundary = root$root
boundary
pi0 = 0.4
pi1 = 1 - pi0
fx0 = function(x) pi0 * dnorm(x, 0, 2)
fx1 = function(x) pi1 * dnorm(x, 2, 2)
1 - (integrate(fx0, -Inf, boundary)$value + integrate(fx1, boundary, Inf)$value)
