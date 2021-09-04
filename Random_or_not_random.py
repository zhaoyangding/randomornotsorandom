#!/usr/bin/env python
# coding: utf-8

# In[1]:


# generate a vector of random numbers which obeys the given distribution.
#
# n: length of the vector
# mu: mean value
# sigma: standard deviation.
# dist: choices for the distribution, you need to implement at least normal 
#       distribution and uniform distribution.
#
# For normal distribution, you can use ``numpy.random.normal`` to generate.
# For uniform distribution, the interval to sample will be [mu - sigma/sqrt(3), mu + sigma/sqrt(3)].

import numpy as np
import math
def generate_random_numbers(n, mu, sigma, dist="normal"): 
    if dist == "normal":
        return np.random.normal(mu, sigma, n)
    elif dist == "uniform":
    #write you code here
        return np.random.normal(mu - sigma/math.sqrt(3), mu + sigma/math.sqrt(3),n)
        pass
    else:
        raise Exception("The distribution {unknown_dist} is not implemented".format(unknown_dist=dist))
        
        
# test your code:
y_test = generate_random_numbers(5, 0, 0.1, "normal")
print(y_test)


# In[2]:


y1 = generate_random_numbers(105, 0.5, 1.0, "normal")
print(y1)
y2 = generate_random_numbers(105, 0.5, 1.0, "uniform")
print(y2)


# In[3]:


# IGD, the ordering is permitted to have replacement. 
def IGD_wr_task1(y):
    n = len(y)
    ordering = np.random.choice(n, n, replace=True)
    # implement the algorithm's iteration of IGD. Your result should return the the final xk
    # at the last iteration and also the history of objective function at each xk.
    x0=0
    f=[]
    x=[x0]
    for k in range(n):
        rk=1/(k+1)
        x1=x0-rk*(x0-y[ordering[k]])
        x0=x1
        f.append(1/2*sum((x1-y)**2))
        x.append(x1)
    return f,x1,x

    pass
    
# IGD, the ordering is not permitted to have replacement.
def IGD_wo_task1(y):
    n = len(y)
    ordering = np.random.choice(n, n, replace=False)
    # implement the algorithm's iteration of IGD. Your result should return the the final xk
    # at the last iteration and also the history of objective function at each xk.
    x0=0
    f=[]
    x=[x0]
    for k in range(n):
        rk=1/(k+1)
        x1=x0-rk*(x0-y[ordering[k]])
        x0=x1
        f.append(1/2*sum((x1-y)**2))
        x.append(x1)
    return f,x1,x
    pass


# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns
f_norm_1,xk_norm_1,x_norm_1=IGD_wr_task1(y1)

f_norm_2,xk_norm_2,x_norm_2=IGD_wo_task1(y1)
for history in [f_norm_1,f_norm_2]:
    plt.plot(history)
f_uniform_1,xk_uniform_1,x_uniform_1=IGD_wr_task1(y2)

f_uniform_2,xk_uniform_2,x_uniform_2=IGD_wo_task1(y2)

##plot the history

for history in [f_uniform_1,f_uniform_2]:
    plt.plot(history)


# In[5]:


#Conclusion
# From the graph, it clearly show that method without replacement generate a better result
#since IGD_wo_task1(y) converge to the true solution (mean value of y) more steady

#Compare a few values of xk
for x in [x_norm_1,x_norm_2]:
    plt.plot(x)
for x in [x_uniform_1,x_uniform_2]:
    plt.plot(x)


# In[35]:


# IGD, the ordering is permitted to have replacement. 
#
#
def IGD_wr_task2(y):
    n = len(y)
    ordering = np.random.choice(n, n, replace=True)
    # implement the algorithm's iteration of IGD. Your result should return the the final xk
    # at the last iteration and also the history of objective function at each xk.
    s0=0
    g=[]
    s=[s0]
    for k in range(n):
        a=np.random.uniform(1,2,n)
        b=min(1/a)
        rk=0.95*b
        s1=s0-rk*a[ordering[k]]*(s0-y)
        s0=s1
        s.append(s1)
        g.append(1/2*sum(a*(s1-y)**2))
    return g,s1,s
    pass
# IGD, the ordering is not permitted to have replacement.
#
#
def IGD_wo_task2(y):
    n = len(y)
    ordering = np.random.choice(n, n, replace=False)
    # implement the algorithm's iteration of IGD. Your result should return the the final xk
    # at the last iteration and also the history of objective function at each xk.
    s0=0
    g=[]
    s=[s0]
    for k in range(n):
        a=np.random.uniform(1,2,n)
        b=min(1/a)
        rk=0.95*b
        s1=s0-rk*a[ordering[k]]*(s0-y)
        s0=s1
        s.append(s1)
        g.append(1/2*sum(a*(s1-y)**2))
    return g,s1,s
    pass


# In[38]:


g_norm_1,sk_norm_1,s_norm_1=IGD_wr_task2(y1)

g_norm_2,sk_norm_2,s_norm_2=IGD_wo_task2(y1)
for history in [g_norm_1,g_norm_2]:
    plt.plot(history)
g_uniform_1,sk_uniform_1,s_uniform_1=IGD_wr_task2(y2)

g_uniform_2,sk_uniform_2,s_uniform_2=IGD_wo_task2(y2)

##plot the history

for history in [g_uniform_1,g_uniform_2]:
    plt.plot(history)


# In[20]:


#Conclusion
# From the graph, it clearly show that method without replacement generate a better result
#since IGD_wo_task1(y) converge to the true solution (mean value of y) more steady


# In[27]:


# generation of exact solution and data y and matrix A.

def generate_problem_task3(m, n, rho):
    A = np.random.normal(0., 1.0, (m, n))
    x = np.random.random(n) # uniform in (0,1)
    w = np.random.normal(0., rho, m)
    y = A@x + w
    return A, x, y


# In[24]:


# We generate the problem with 200x100 matrix. rho as 0.01.
#
A, xstar, y = generate_problem_task3(200, 100, 0.01)


# In[39]:


# In these two functions, we could only focus on the first n steps and try to make comparisons on these data only.
# In practice, it requires more iterations to converge, due to the matrix might not be easy to deal with.
# You can put the ordering loop into a naive loop: namely, we simply perform the IGD code several rounds.
#
#
#
# IGD, the ordering is permitted to have replacement. 
#
#
def IGD_wr_task3(y, A):
    n = len(y)
    ordering = np.random.choice(n, n, replace=True)
    # implement the algorithm's iteration of IGD. Your result should return the the final xk
    # at the last iteration and also the history of objective function at each xk.
    t0=0
    h=[]
    t=[t0]
    for k in range(n):
        rk=0.001
        t1=t0-rk*A[ordering[k]]*(A[ordering[k]]@t0-y[ordering[k]])
        t0=t1
        t.append(t1)
        h.append(sum(A@t1-y)**2)
    return h,t1,x
    pass

# IGD, the ordering is not permitted to have replacement.
#
#
def IGD_wo_task3(y, A):
    n = len(y)
    ordering = np.random.choice(n, n, replace=False)
    # implement the algorithm's iteration of IGD. Your result should return the the final xk
    # at the last iteration and also the history of objective function at each xk.
    t0=0
    h=[]
    t=[t0]
    for k in range(n):
        rk=0.001
        t1=t0-rk*A[ordering[k]]*(A[ordering[k]]@t0-y[ordering[k]])
        t0=t1
        t.append(t1)
        h.append(sum(A@t1-y)**2)
    return h,t1,x
    pass


# In[42]:


h_norm_1,tk_norm_1,t_norm_1=IGD_wr_task3(y1,A)

for history in [h_norm_1]:
    plt.plot(history)

h_uniform_1,tk_uniform_1,t_uniform_1=IGD_wr_task3(y2,A)

for history in [h_uniform_2]:
    plt.plot(history)


# In[41]:


#Conclusion
# From the graph, it clearly show that method without replacement generate a better result
#since IGD_wo_task1(y) converge to the true solution (mean value of y) more steady

