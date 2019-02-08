## STAT 318 Assignment 3

Student Name : Xiao Meng

Student ID : 88211337

------

#### 1.

##### (a)

As known,

![1(a)](E:\文档\UC\318\Assignment\A3\1(a).png)

The best split should have maximal reduction in impurity
$$
\begin{align}
\Delta (I) &= I(m) - \frac{N_{2m}}{N_m}I(2m) - \frac{N_{2m+1}}{N_m}I(2m+1)\\
\end{align}
$$
For split with $$X_1$$, we can get the reduction in impurity, 
$$
\begin{align}
\Delta (I) &=\frac12\times(1-\frac12) + \frac12(1-\frac12) - \frac34\times(\frac13\times\frac23 + \frac23\times\frac13) - \frac14\times0= \frac12-\frac13 = \frac16
\end{align}
$$
For split with $$X_2$$, we can get the reduction in impurity, 
$$
\begin{align}
\Delta (I) &=\frac12 - \frac12\times2\times\frac25\times\frac35 -\frac12\times2\times\frac35\times\frac25 = \frac1{50}
\end{align}
$$
For split with $$X_3$$, we can get the reduction in impurity, 
$$
\begin{align}
\Delta (I) &=\frac12 - \frac12\times2\times\frac12\times\frac12 -\frac12\times2\times\frac12\times\frac12 = 0
\end{align}
$$
Therefore, the best split with maximal $$\Delta (I)$$ is $$X_1$$.

---

##### (b)

















---

##### (c)

As (b) shown, only the left daughter node is impure, to find the best split of left daughter node, calculate the impure reduction of $$X_2, X_3$$.

For split with $$X_2$$, we can get the reduction in impurity, 
$$
\begin{align}
\Delta (I) &=2\times\frac13\times\frac23 - \frac25\times0 -\frac35\times2\times\frac49\times\frac59 = \frac4{27}
\end{align}
$$
For split with $$X_3$$, we can get the reduction in impurity, 
$$
\begin{align}
\Delta (I) &=2\times\frac13\times\frac23 - \frac23\times2\times\frac12\times\frac12 -\frac13\times0 = \frac19
\end{align}
$$
Therefore the best split of the left daughter node with the maximal $$\Delta (I)$$ is $$X_2$$ 

















---

##### (d)

As the tree in (c) shown, the right daughter node of node $$X_2 < 0.5$$ should be classifed as High. So there are 20 observations to be misclassified.

---

##### (e)

After splitting with $$X_3$$, for its left daughter node we can get the reduction impurity:

For $$X_1$$:
$$
\begin{align}
\Delta (I) &=2\times\frac12\times\frac12 - 2\times\frac12\times\frac12 = 0
\end{align}
$$

For $$X_2​$$ :
$$
\begin{align}
\Delta (I) &=2\times\frac12\times\frac12 = \frac12
\end{align}
$$
With the maximal $$\Delta(I)$$,  the best split for the left daughter node is $$X_2$$.

For the right daughter node we can get:

For $$X_1$$ : 
$$
\begin{align}
\Delta (I) &=2\times\frac12\times\frac12 = \frac12
\end{align}
$$
For $$X_2$$ :
$$
\begin{align}
\Delta (I) &=2\times\frac12\times\frac12 - 2\times\frac12\times2\times\frac45\times\frac15 = \frac9{50}
\end{align}
$$
With the maximal $$\Delta(I)$$,  the best split for the left daughter node is $$X_1$$.























---

##### (f)

From (e) we can get a tree with no misclassifed while we got 20 misclassied in tree of (c), which means tree in (e) is more accurancy than tree in (c). Then, we can understand the greedy natural in CART is that each iteration of split just choose a locally optimise just like choosing the first best split $$X_1$$ in (c), rather than globally which could make a better tree in some future steps.

---

---

#### 2

##### (a)

Fit and plot a regression tree to the training set.

![2(a)](E:\文档\UC\318\Assignment\A3\2(a).png)

​                                                       **Figure 2.1 : regression tree with the traing set**

For this regression tree, the training MSE is 3.0411(4dp) and the testing MSE is 6.1262(4dp).

Based on the regression tree we can see, there are 20 terminal nodes and the residual mean deviance is 3.275. For all 9 variables, there are only 6 variables to be used to create the regression tree. The Price is the most significant factor, besides Age and Advertising also played important roles for sale.

---

##### (b)

![2(b)](E:\文档\UC\318\Assignment\A3\2(b).png)

​                                                                  **Figure 2.2 : Error rate of each size of tree**



![2(b)2](E:\文档\UC\318\Assignment\A3\2(b)2.png)

​						                  **Figure 2.3 : regression tree with 9 nodes**

From figure 2.2 we can see, compare with 20 terminal nodes, the error rate of many nodes below 20 is lower and regreesion tree is simpier. So pruning improve the tree's performance. When the number of nodes is 9, the regression tree has the lowest cross-validation error rate. Also, this regression tree with 9 nodes is not too complex and all simpier trees with less nodes have higher error rate. Therefore, I chosed 9 nodes to do the pruning.

After pruning to 9 nodes, the test MSE 5.6903(4dp) which is lower than 6.1262(4dp) of regression tree with 20 nodes in (a). That means pruning improve the test MSE. 

---

##### (c)

When fit a bagged regression tree with all predictor in this problem, which means parameter mtry = 9, and fit a random forrest with mtry = p/3 = 3, we can get the test and training MSEs shown as table below: 

|                        | test MSE | training MSE |
| :--------------------- | :------: | :----------: |
| bagged regression tree |  4.5547  |    0.8447    |
| random forest          |  4.9128  |    1.045     |

From this table, we can know both test and traning MSEs of bagged regression tree are lower, which means bagged regression tree performed better than random forest in this problem. While one feature of random forest is known  as decorrelating, therefore decorrelating trees does not have an effective strategy for this problem.

---

##### (d)

Fit the boosted regression tree with the training data for tree depths from 1 to 5, shrinkage parameters in (0.1, 0.01, 0.001) and number of trees in (1000, 2500, 5000), the best tree with minimal test MSEs has tree depth of 1, shrinkage parameter of 0.01 and 1000 trees. The traning  and test MSE for the best tree is 3.5571 and 4.1067.

---

##### (e) 

With the test MSE of each regression tree, the boosted regression tree with lowest test MSE performed best in this problem.

As the importance shown, the most important predictors are Price, CompPrice, Advertising, and age.

![2(e)](E:\文档\UC\318\Assignment\A3\2(e).png)

---

---

#### 3.

##### (a)

![3(a)k=5](E:\文档\UC\318\Assignment\A3\3(a)k=5.png)

![3(a)k=52](E:\文档\UC\318\Assignment\A3\3(a)k=52.png)

![3(a)k=53](E:\文档\UC\318\Assignment\A3\3(a)k=53.png)

After performing k-means clustering, Hierachical clustering with complete linkage and single kage with setting cluster as 5, there are some differences among these three cluster methods and the actual cluster labels. 

Shown as the table blow, for the result of the k-means clustering with appropriate trials, compare with the actual cluster labels, Cluster 2 and 3 are exactly the same as the actual cluster lable while there are only 2 misclassifed in Cluster1. But for Cluster 4 and 5 in k-means clustering which is Cluster 5 and 4 for its actual cluster, are totally misclassified. The error rate of k-means clustering of this run is 40.267%.



![1539231586818](C:\Users\santo\AppData\Roaming\Typora\typora-user-images\1539231586818.png)

In Hierachical Clustering with Complete Linkage for this problem, besides Cluster 1 is almost assigned the same as the actual one, the other clusters are all misclassified. The whole Cluster 2, 3, 5 are assigned to Cluster 3, 4, 2, as well as part of Cluster 4 is allocated to Cluster 3 and the others is to Cluster 5 in hierarchical clustering with complete linkage. The error rate is 80.13%.

![1539231602241](C:\Users\santo\AppData\Roaming\Typora\typora-user-images\1539231602241.png)

In Hierachical Clustering with Single Linkage for this problem, besides Cluster 1 and 4 is almost assigned the same as the actual one. The whole Cluster 2, 3, 5 are assigned to Cluster 3, 1, 2 in hierarchical clustering with complete linkage. The error rate is 69.33%.

![1539231616801](C:\Users\santo\AppData\Roaming\Typora\typora-user-images\1539231616801.png)

After performing these three motheds clustering we can get, with the minimal error rate of clusters, K-means could give the best performance with appropriate trials. While Hierachical Clustering with Single Linkage performed better than Complete Linkage.





---

##### (b)

![3(b)k = 31](E:\文档\UC\318\Assignment\A3\3(b)k = 31.png)

![3(b)k = 32](E:\文档\UC\318\Assignment\A3\3(b)k = 32.png)

![3(b)k = 33](E:\文档\UC\318\Assignment\A3\3(b)k = 33.png)

Using 3 clusters to perform hese three methods of clustering to data2, we can see the result from pictures above. 

Compare with the actual cluster labels, the k-mean clustering after appropriate trials for this data set has assigned cluster 1 correctly. The actual cluster 2 has been allocated to each cluster by k-mean clustering as table shown below. Then almost all the cluster 3 is assigned to cluster1. The error rate is 52.67%.

![1539239120046](C:\Users\santo\AppData\Roaming\Typora\typora-user-images\1539239120046.png)

Hierachical Clustering with Complete Linkage for this data set has a similar performance to k-means clustering. It also assigned correctly in cluster 1, partial assigned cluster 2 into every cluster and almost all the cluster 3 is assigned to cluster1. The error rate is 40.29%.

![1539239137780](C:\Users\santo\AppData\Roaming\Typora\typora-user-images\1539239137780.png)

In this data set, Hierachical Clustering with Single Linkage shows its obvious feature that trailing clusters which single observations are fused one at a time. Single linkage assigned only one observation to cluster 2 and 3, and assigned all the other observations to cluster 1, which is a significant departure from the actual cluster labels and gives a high error rate of 91.63%.

![1539239149309](C:\Users\santo\AppData\Roaming\Typora\typora-user-images\1539239149309.png)

Compared with these three methods in this data set, we can get Hierachical clustering with complete linkage performed the best and meanwhile k-mean clustering performed a little worse. With a trailing clusters in algorithms of single linkage, single linkage method played the worst in this data set.

---

---

### Appendix

```R
#======================
#2 fit regression tree
#======================

#(a)Fit a regression tree to the training set.
library(tree)
cartrain = read.csv("E:/文档/UC/318/Assignment/A3/carTrain.csv")
cartest = read.csv("E:/文档/UC/318/Assignment/A3/carTest.csv")
tree.carseats = tree(Sales~., cartrain)
summary(tree.carseats)

#plot the tree
plot(tree.carseats)
text(tree.carseats, pretty = 0)

#MSE of training and test data
pred.train = predict(tree.carseats, cartrain)
plot(pred.train, cartrain$Sales)
abline(0, 1)
mean((pred.train - cartrain$Sales)^2)

pred.test = predict(tree.carseats, cartest)
plot(pred.test, cartest$Sales)
abline(0, 1)
mean((pred.test - cartest$Sales)^2)

#(b) Pruning
#Find the best depth of tree
cv.carseats = cv.tree(tree.carseats)
plot(cv.carseats$size, cv.carseats$dev, type = "b", xlab = "Tree Size", ylab = "Mean Squared Error")

#Pruning with the best depth
tree.carseats.prune = prune.tree(tree.carseats, best = 9)
plot(tree.carseats.prune)
text(tree.carseats.prune, pretty = 0, cex = 0.9)
pred.test.prune = predict(tree.carseats.prune, cartest)
mean((pred.test.prune - cartest$Sales) ^ 2)

#(c) Fit a bagged regression tree and a random forest
#fit a bagged regression tree(Using random forest with the number of all the #predict variables : 9)
library(randomForest)
bag.carseats = randomForest(Sales~., cartrain, mtry = 9, importance = TRUE)
bag.carseats

#test and training MSE for bagged
pred.train.bag = predict(bag.carseats, cartrain)
mean((pred.train.bag - cartrain$Sales)^2)
pred.test.bag = predict(bag.carseats, cartest)
mean((pred.test.bag - cartest$Sales)^2)

#fit a random forest with p/3 = 3 variables
rf.carseats = randomForest(Sales~., cartrain, mtry = 3, importance = TRUE)
rf.carseats

#test and traning MSEs for random forest
pred.train.rf = predict(rf.carseats, cartrain)
mean((pred.train.rf - cartrain$Sales)^2)
pred.test.rf = predict(rf.carseats, cartest)
mean((pred.test.rf - cartest$Sales)^2)

#(d)Fit a boosted regression tree with training data and calculate the test and traning MSEs
#fit a boosted regression tree with different depth, number of trees and shrinkages to find the best one(with smallest test MSE)
library(gbm)
mse.boost = c()
min_mse = 50
for (n in c(1000, 2500, 5000)){
     for (d in seq(1:5)){
         for (s in c(0.1, 0.01, 0.001)){
             boost.carseats = gbm(Sales~., cartrain, distribution = "gaussian", n.trees = n, interaction.depth = d, shrinkage = s, verbose = F)
             pred.test.boost = predict(boost.carseats, cartest, n.trees = n)
             mse = mean((pred.test.boost - cartest$Sales)^2)
             mse.boost = c(mse.boost, mse)
             if (mse < min_mse){
                min_mse = mse
                min_par = c(s, d, n)
             }
         }
     }
}
options(scipen = 2)
min_mse
min_par

#calculate the test and training MSEs with depth = 1, number of tree = 1000, shrinkage = 0.01
boost.carseats.best = gbm(Sales~., cartrain, distribution = "gaussian", n.tree = 1000, interaction.depth = 1, shrinkage = 0.01, verbose = F)
pred.train.boost.best = predict(boost.carseats.best, cartrain, n.trees = 1000)
mean((pred.train.boost.best - cartrain$Sales)^2)
pred.test.boost.best = predict(boost.carseats.best, cartest, n.trees = 1000)
mean((pred.test.boost.best - cartest$Sales)^2)

#(e)the most important predictors in this problem
#With the smallest test MSE, boost regression tree model performed the best.
#Find the most important predictors in boost regression tree model
summary(boost.carseats.best)

#======================
#3 Cluster
#======================
#(a)k-means with k = 5 plot with different colours and show the total within-cluster sum of squares
data1 = read.csv("E:/文档/UC/318/Assignment/A3/A3data1.csv")
km1.out = kmeans(data1[1:2], 5, nstart = 50)
plot(data1[1:2], col = (km1.out$cluster + 1), main = "K-Means Clustering Result with K = 5", pch = 20, cex = 1)
km1.out$tot.withinss

#(b)hierarchical clustering with k = 5 plot with different colours with complete linkage
hc1.complete = hclust(dist(data1[1:2]), method = "complete")
cluster1.complete = cutree(hc1.complete, 5)
plot(data1[1:2], col = (cluster1.complete + 1), main = "Hierachical Clustering of Complete Linkage Result with K = 5", pch = 20)

#(b)hierarchical clustering with k = 5 plot with different colours with single linkage
hc1.single = hclust(dist(data1[1:2]), method = "single")
cluster1.single = cutree(hc1.single, 5)
plot(data1[1:2], col = (cluster1.single + 1), main = "Hierachical Clustering of Complete Linkage Result with K = 5", pch = 20)

#(C)compare the clustering methods of complete and single with the actual cluster 
km1.clusters = km1.out$cluster
table(km1.clusters, data1$Cluster)
mean(km1.clusters != data1$Cluster)

table(cluster1.complete, data1$Cluster)
mean(cluster1.complete != data1$Cluster)

table(cluster1.single, data1$Cluster)
mean(cluster1.single != data1$Cluster)

#Repeat (a) - (c) with K = 3
data2 = read.csv("E:/文档/UC/318/Assignment/A3/A3data2.csv")
km2.out = kmeans(data2[1:2], 3, nstart = 50)
plot(data2[1:2], col = (km2.out$cluster + 1), main = "K-Means Clustering Result with K = 3", pch = 20, cex = 1)

hc2.complete = hclust(dist(data2[1:2]), method = "complete")
cluster2.complete = cutree(hc2.complete, 3)
plot(data2[1:2], col = (cluster2.complete + 1), main = "Hierachical Clustering of Complete Linkage Result with K = 3", pch = 20)

hc2.single = hclust(dist(data2[1:2]), method = "single")
cluster2.single = cutree(hc2.single, 3)
plot(data2[1:2], col = (cluster2.single + 1), main = "Hierachical Clustering of Complete Linkage Result with K = 5", pch = 20)

km2.clusters = km2.out$cluster
table(km2.clusters, data2$Cluster)
mean(km2.out$cluster != data2$Cluster)

table(cluster2.complete, data2$Cluster)
mean(cluster2.complete != data2$Cluster)

table(cluster2.single, data2$Cluster)
mean(cluster2.single != data2$Cluster)


```

