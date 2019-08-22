# on-lstm-val
# 运行说明
## 文件说明
* 1 on-lstm源码：https://github.com/bojone/on-lstm
* 2 将 new_hier_model.py放到上述源码的同一根目录下

## 依赖环境
* 1 rouge安装参考：https://blog.csdn.net/Hay54/article/details/78744912
````
Tips：
如果是用Conda虚拟环境，找到path to your virtual env/lib/python2.7/site-packeges/pyrouge
将rouge安装完成后的Rouge-1.5.5的文件夹拷贝到上述路径下
`````
* 2 其他需安装
````
安装glob2, sentencepiece.运行命令：
pip install glob2
pip install sentencepiece
````
## Dataset
* Dataset来源于：CNN/DM，源数据：https://drive.google.com/open?id=1BM9wvnyXx9JvgW2um0Fk9bgQRrx03Tol
* 由于源数据是pytorch环境可用，因此对源数据进行了轻微处理
* 下载data.zip(https://pan.baidu.com/s/1-x_Y8AkvxKRKSCn3QUqLbQ)，解压后放置到和代码同一级的目录即可

## 训练次数说明
1. 由于机器GPU所限制，设置了batch_size为100，若GPU资源丰富，可以将batch_size修改为1000或者10000
2. 若修改了batch_size，则new_hier_model.py中fit_genenrator函数中的一个参数项：step_per_epoch应对应减小，设置为1000
