# 利用Pytorch重做deeplearning.ai作业

* pytorch和tensorflow一样也是做机器学习的重要框架, 不像tensorflow做出来的变量先是placeholder, 只把位置holder住, 没有往里feed数据之前很难进行调试. pytorch里面的tensor是有值的, 所以应该比较容易debug.

* pytorch的另一个好处是貌似比较容易迁移到GPU的运算上, pytorch据说和numpy差不多, 所以如果慢慢熟悉了pytorch, 在程序里用pytorch替代numpy, 以后追求运算速度时只主要相对简单的步骤就可以用上CUDA了.

正因为上述原因, 所以我打算用pytorch把deeplearning.ai的课程作业挑一些重写一遍.

首先, 按照[Yoshua Bengio实验室MILA开放面向初学者的PyTorch教程](https://github.com/mila-udem/welcome_tutorials) ([机器之心翻译](https://www.jiqizhixin.com/articles/2017-11-16-8)) 熟悉一遍pytorch的基本命令. 必要时也要参考[pytorch中文文档](https://pytorch-cn.readthedocs.io/zh/latest/)的说明.

安装就不详细说了, 还是用[cocalc](https://cocalc.com/app)或者[MS azure notebook](https://notebooks.azure.com)的在线版吧, 其中cocalc从善如流已经装好了pytorch, azure notebook里面没有, 但可以参考[这个帖子](https://github.com/Microsoft/AzureNotebooks/issues/201#issuecomment-338466615)的解决方案, 加入一段自动运行的设置, 每次启动之前会安装一遍pytorch.

----

# 目录
----

# 熟悉pytorch
* [与Numpy相似的基本操作](/basic_pytorch/basic01.html)
