
# coding: utf-8

# # 自动微分autograd
# 
# 神经网络中反向传播算法是用微分的链式法则求微分. 
# 
# 这一部分貌似pytorch自家的tutorial讲得更清楚些: 
# http://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

# In[93]:


import numpy as np
import torch
from torch.autograd import Variable


# # Variable
# 一个变量里包含三个部分
# ![](http://pytorch.org/tutorials/_images/Variable.png)
# 我大概理解是把Tensor放进Variable里面就可以生成. 

# In[53]:


x = Variable(torch.ones(2, 3), requires_grad=True)
print("打印x{0}".format(x))
print("x中的data, 好像和直接打印x差别不大,{0}".format(x.data))

y=x**2

print("一个Variable要通过其他过程的生成才有grad_fn, 否则就是个类似常量的东西")
print("所以x的grad_fn={0}".format(x.grad_fn))
print("但y=x**2, 所以y的grad_fn={0}".format(y.grad_fn))


# # 微分
# 如果一个Vairable是通过其他Variable生成的, 就可以进行求导了. 我大学一年级上高数的时候怎么没碰上这种好事. (不过好像当时对Mathematica还比较熟练). 
# 
# 只需要先运行y.backward(gradient=xxx), 注意如果不指定的时候y.backward()相当于y.backward(torch.Tensor([1.0])), 但如果y是个矩阵或者向量, 就会报错

# In[113]:


x = Variable(torch.ones(2, 3)*0.5, requires_grad=True)
y = 7 * x**3 + 3
print("x={0}\ny={1}".format(x,y))
y.backward(gradient=torch.ones(2,3)*1.1)
print(x.grad)
print(7*3*(0.5**2)*1.1)


# 我还不是很理解
# ```python 
# y.backward(gradient=torch.ones(2,3)*1.1)
# ```
# 中gradient的定义
# 从计算的结果上来看是
# 求出
# $$
# \frac{dy}{dx}|_{x=x.data} \times gradient
# $$
# 对于
# $$
# y=7 x ^3+3
# \\
# \frac{dy}{dx}=21 x^2, 代入x=0.5
# \\
# 21\times (0.5)^2
# 然后再乘以gradient = 1.1
# \\
# 21\times (0.5)^2 \times 1.1 = 5.775
# $$

# 在http://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html 例题里面用的参数不太好, 其中的Xi和gradient都是取的1, 容易混淆. 
# 注意如果z.backward()里面的gradient不省略的话, 维度应当和z本身一致. 

# In[114]:


x1 = Variable(torch.ones(2, 3)*np.sqrt(2), requires_grad=True)
x2 = Variable(torch.ones(3, 2)*np.sqrt(3), requires_grad=True)
y1 = torch.sqrt(x1)
z = torch.mm(x1**2,x2**2)
print(z)
z.backward(gradient=torch.ones(2,2)*np.sqrt(5))
print(x1.grad)
print(2*np.sqrt(2)*
      (np.sqrt(3)**2+np.sqrt(3)**2)
        *np.sqrt(5))


# In[ ]:




