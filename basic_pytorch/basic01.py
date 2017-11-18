
# coding: utf-8

# # Pytorch 基础
# 
# * pytorch和tensorflow一样也是做机器学习的重要框架, 不像tensorflow做出来的变量先是placeholder, 只把位置holder住, 没有往里feed数据之前很难进行调试. pytorch里面的tensor是有值的, 所以应该比较容易debug. 
# 
# * pytorch的另一个好处是貌似比较容易迁移到GPU的运算上, pytorch据说和numpy差不多, 所以如果慢慢熟悉了pytorch, 在程序里用pytorch替代numpy, 以后追求运算速度时只主要相对简单的步骤就可以用上CUDA了. 
# 
# 正因为上述原因, 所以我打算用pytorch把deeplearning.ai的课程作业挑一些重写一遍. 
# 
# 首先, 按照[Yoshua Bengio实验室MILA开放面向初学者的PyTorch教程](https://github.com/mila-udem/welcome_tutorials) ([机器之心翻译](https://www.jiqizhixin.com/articles/2017-11-16-8)) 熟悉一遍pytorch的基本命令. 必要时也要参考[pytorch中文文档](https://pytorch-cn.readthedocs.io/zh/latest/)的说明. 
# 
# 安装就不详细说了, 还是用[cocalc](https://cocalc.com/app)或者[MS azure notebook](https://notebooks.azure.com)的在线版吧, 其中cocalc从善如流已经装好了pytorch, azure notebook里面没有, 但可以参考[这个帖子](https://github.com/Microsoft/AzureNotebooks/issues/201#issuecomment-338466615)的解决方案, 加入一段自动运行的设置, 每次启动之前会安装一遍pytorch. 

# # 张量

# In[2]:


import numpy as np
import torch 


# ## 初始化张量
# 参考https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch/#creation-ops
# 

# In[56]:


print("N(0,1)的正态分布随机数{0}".format(torch.randn(2,3)))
print("[0,1]的均匀分布随机数{0}".format(torch.rand(2,3)))
print(torch.zeros(2,3))
print(torch.ones(2, 3))
print(torch.arange(1, 3))


# ## 转换张量
# 用torch.Tensor可以将List, np array等转换成tensor

# In[42]:


a=[[1,2],[3,4]]
x= torch.Tensor(a)
b=np.linspace(1,12,12).reshape(3,4)
y=torch.Tensor(b)
print(x)
print(y)
print(y[:,1])


# ## 获取维度
# 用.size()获得tensor的维数, 相当于.shape()

# In[43]:


y=torch.linspace(1,12,12).view(3,4)
print(y.size()[0])


# ## reshape
# torch用.view()来做reshape

# In[44]:


y=torch.linspace(1,12,12).view(3,4)
print("原始形态\n{0}".format(y))
print('矮胖\n{0}'.format(y.view(-1,6)))
print("压扁\n{0}".format(y.view(1,-1)))


# ## 连接
# torch.cat,按照不同轴来连接张量

# In[65]:


y=torch.arange(0,6).view(2,3)
print('原张量{0}'.format(y))
print('按照0轴方向连接{0}'.format(torch.cat([y,y],0)))
print('按照1轴方向连接{0}'.format(torch.cat([y,y],1)))


# ## 转置
# * torch.t() 简单的二维转置
# * torch.transpose()
# 
# 也可以写成y.t()
# 注意是函数, 要有括号

# In[75]:


y=torch.arange(0,6).view(2,3)
print(y)
print(torch.t(y))
print(torch.transpose(y,0,1))


# ## 切片索引
# 跟常规numpy的相同, 复杂的逻辑切片用torch.masked_select

# In[136]:


y=torch.arange(0,6)
x=(y>=3) & (y<=4)
print(y[x])


# # 常用数学运算

# ## 随机数
# 

# In[137]:


torch.manual_seed(1)
torch.rand(1,3)


# ## 乘法
# dot乘法, 用一个诡异的torch.mm, 不是torch.dot!!!

# In[100]:


x=torch.arange(0,6).view(2,3)
y=torch.arange(6,0,-1).view(3,2)
print(x)
print(y)
print(torch.mm(x,y))
print(x.mm(y))


# ## 乘方开方 torch.sqrt()

# In[45]:


y=torch.linspace(1,12,12).view(3,4)
torch.sqrt(y**2)


# ## 求和, 求积
# * torch.sum(input,dim)
# * torch.prod(input,dim)

# In[112]:


y=torch.linspace(1,10,10).view(2,5)
print(y)
print("按0轴求和{0}".format(torch.sum(y,0)))
print("按1轴求和{0}".format(torch.sum(y,1)))

print("按0轴求乘积{0}".format(torch.prod(y,0)))
print("按1轴求乘积{0}".format(torch.prod(y,1)))


# ## 范数
# torch.norm(input, p, dim, out=None) → Tensor

# In[110]:


y=torch.arange(0,6).view(2,-1)
print(y)
print("1阶范数, 按0轴取{0}".format(y.norm(p=1,dim=0)))
print("2阶范数, 按1轴取{0}".format(y.norm(p=2,dim=1)))

