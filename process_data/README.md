### Preprocess Dataset
#### Step1 准备
* 创建两个文件夹：json_data, train_data
* 将所有json放到json_data中

#### Step2 调节参数
* 在main函数中，有一个参数test_num, 用于调整test dataset的pdd个数。现默认为1
* 下载spm.cnndm.model，并和main.py放置在同一级目录下。 这是一个提前pre-trained的vocalbulary，直接可以使用。


#### Step3 运行
* python main.py
