# Notebook1
Mes codes pour le language Python

#Question préliminaire

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import perceptron_source
from perceptron_source import rand_gauss, rand_bi_gauss, rand_clown, rand_checkers, grid_2d, plot_2d,frontiere, mse_loss, gradient, plot_gradient, poly2, collist, symlist, gr_mse_loss, hinge_loss, gr_hinge_loss


############################################################################
########            Data Generation: example                        ########
############################################################################
#Section 1 : introduction
#Question 1
n=100
mu=[1,1]
sigma=[1,1]
rand_gauss(n,mu,sigma)

n1=20
n2=20
mu1=[1,1]
mu2=[-1,-1]
sigma1=[0.9,0.9]
sigma2=[0.9,0.9]
data1=rand_bi_gauss(n1,n2,mu1,mu2,sigma1,sigma2)
data11=rand_bi_gauss(n1,n2,mu1,mu2,sigma1,sigma2)

std1=1
std2=5
n1=50
n2=50
data2=rand_clown(n1,n2,std1,std2)


std=0.1
data3=rand_checkers(n1,n2,std)
dataX=data1[:,:2]
dataXX=data11[:,:2]
dataY=data1[:,2]

#Question 2
#save data1
import pickle

with open('data1.pickle', 'wb') as output:
    pickle.dump(data1, output)
#load data1
    
with open('data1.pickle', 'rb') as data:
    dataset = pickle.load(data)
############################################################################
########            Displaying labeled data                         ########
############################################################################
#Question 3
plt.close("all")

plt.figure(1, figsize=(15,5))
plt.subplot(131)
plt.title('First data set')
plot_2d(data1[:,:2],data1[:,2],w=None)

plt.subplot(132)
plt.title('Second data set')
plot_2d(data2[:,:2],data2[:,2],w=None)

plt.subplot(133)
plt.title('Third data set')
plot_2d(data3[:,:2],data3[:,2],w=None)
plt.show()


############################################################################
########                Logistic regression example                          ########
############################################################################
#On utilise dataX et dataY définis en Section 1 Question 1
from sklearn import linear_model
#Question 1
my_log=linear_model.LogisticRegression()
#Question 2
my_log.fit(dataX,dataY)
my_log.coef_
my_log.intercept_

#Question 4
#Pour la prediction il faut d'abord disposer d'un jeu de données tests
#On choisit dataXX généré en Section 1 Question 1

dataYY=my_log.predict(dataXX)
############################################################################
########                Perceptron example                          ########
############################################################################
#Section 3
#Cost function
#Voir le source pour l'implementation
#Faire des graphes �  w fixe pour le loss

#Apprentissage du perceptron en pratique
#Question 1 : voir cours
#Question 2 pour le MSE
# MSE Loss
epsilon=0.01
niter=75

dataX=data1[:,:2]
dataY=data1[:,2]

#w_ini: intial guess for the hyperplan
w_ini=np.zeros([niter,dataX.shape[1]+1])
std_ini=1
for i in list(range(dataX.shape[1]+1)):
	w_ini[-1,-i+1]=std_ini*np.random.randn(1,1)
	print (w_ini[-1,-i+1])

lfun=mse_loss
gr_lfun=gr_mse_loss


plt.figure(7)
wh,costh=gradient(dataX,dataY,epsilon,niter,w_ini,lfun,gr_lfun,stoch=False)
plot_gradient(dataX,dataY,wh,costh,lfun)
plt.suptitle('MSE and batch')
plt.show()

#Question 3 pour le MSE
epsilon=0.001
plt.figure(8)
plt.suptitle('MSE and stochastic')
wh_sto,costh_sto=gradient(dataX,dataY,epsilon,niter,w_ini,lfun,gr_lfun,
						stoch=True)
plot_gradient(dataX,dataY,wh_sto,costh_sto,lfun)
plt.show()




# Question 2 pour le Hinge Loss
epsilon=0.01
niter=30

dataX=data1[:,:2]
dataY=data1[:,2]

w_ini=np.zeros([niter,dataX.shape[1]+1])
std_ini=10
for i in list(range(dataX.shape[1]+1)):
	w_ini[-1,-i+1]=std_ini*np.random.randn(1,1)


lfun=hinge_loss
gr_lfun=gr_hinge_loss
wh,costh=gradient(dataX,dataY,epsilon,niter,w_ini,lfun,gr_lfun,stoch=False)

plt.figure(9)
plt.suptitle('Hinge and batch')
plot_gradient(dataX,dataY,wh,costh,lfun)
plt.show()

#Question 3 pour le Hinge loss
plt.figure(10)
plt.suptitle('Hinge and stochastic')
wh_sto,costh_sto=gradient(dataX,dataY,epsilon,niter,w_ini,lfun,gr_lfun,
						stoch=True)
plot_gradient(dataX,dataY,wh_sto,costh_sto,lfun)
plt.show()


# Create a figure with all the boundary displayed with a
# brighter display for the newest one
epsilon=1
niter=30
plt.figure(11)
wh_sto,costh_sto=gradient(dataX,dataY,epsilon,niter,w_ini,lfun,gr_lfun,
						stoch=True)
indexess=np.linspace(0,1,niter)
for i in range(niter):
	plot_2d(dataX,dataY,wh_sto[i,:],indexess[i])






