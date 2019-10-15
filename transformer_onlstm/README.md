# 环境配置：
python2.7+Keras 2.2.4 + Tensorflow 1.8
opennmt-tf
glob2


# 所用到的文件
* 1 transformer_demo.py : 实现两级Encoder,sentence-level和Document-level的编码及降维
* 2 transformer_onlstm.py： 主执行文件
* 3 on_lstm_keras.py： 模型文件

# 现存的问题：
* 1 由于降维的方式写的比较早，这块可以变更为其他方式
* 2 利用tensorflow的eager function 无法和on_lstm_keras做一个很好的融合，所以这块是最大的问题。
