# 人脸年龄估算
本次实验是参照Rothe文献，代码部分参照了[yu4u](https://github.com/yu4u/age-gender-estimation)的部分代码。

---
## 使用模型

---
>* ResNet50
>* VGG16

## 环境配置

---
> * 我的电脑配置是：
>    * 显卡：GTX965
>    * 内存：16G
>    * cuda: 10.0
>    * tensoeflow 2.0
>    * keras: 2.3

## 训练流程

>首先在ImageNet获得预训练`->`使用处理好的IMDB数据进行微调网络`->`  
使用划分好的appa-real数据集对网络进行微调网络，并训练出20个网络`->`  
取20个网络的均值作为最终预测值`->`使用softmax期望值作为模型的输出结果。

## 数据集的处理

---
### [IMDB数据集](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) `(`下载链接`)`

---
>点击imdb数据集，网站提供裁剪好的人脸头像数据集；然后再使用create_db.py  
对数据集进行筛选图像。主要是剔除多张人脸，以及显示不清楚的图像。

### [appa-real数据集](http://chalearnlap.cvc.uab.es/dataset/26/description/)

---
>appa-real数据集它不仅含有真实年龄，而且含有表观年龄标签；我们使用他对网  
络进行微调，从而减少真实年龄与表观年龄的误差。然后我对里面不清晰的图像进行  
了筛选，记录在ignore.txt中。

## 展示部分

---
>分别有两个文件，一个是demo.py;另一个是使用pyqt5做的界面。

<img src="https://github.com/nablejohne/age_estimate/tree/master/ui/demo.jpg" width="480px">


## 参考文献
[1] R. Rothe, R. Timofte, and L. V. Gool, "DEX: Deep EXpectation of apparent age from a single image,"  

in Proc. of ICCV, 2015.

[2] R. Rothe, R. Timofte, and L. V. Gool, "Deep expectation of real and apparent age from a single image  

without facial landmarks," in IJCV, 2016.

[3]Agustsson E , Timofte R , Escalera S , et al. Apparent and Real Age Estimation in Still Images with  

Deep Residual Regressors on Appa-Real Database[C]// IEEE International Conference on Automatic Face &   

Gesture Recognition (FG). IEEE, 2017.



  