# 使用旋转任务自监督学习并在图像分类中测试

## 简介

旋转任务是一种自监督学习的代理任务。与监督学习不同，它不要求图片有原始标签，而是将原始图片旋转0，90，180，270度，并生成对应的四种角度标签让模型学习。模型需要预测图片的旋转角度，也就是完成一个4分类问题。

<center>
    <img style="width:60%" src= ".\images\image_rotation_example.jpg">
</center>



本实验尝试在 VOC 和 CIFAR-10 中预训练Resnet-18，并将其在 CIFAR-10 中测试其在传统图片分类效果。

## 内容目录

```
images
logs                          # 存放Tensorboard记录
models

Dataloader.py                 # 加载数据集并旋转图片
Linear_Classfication.py       # 测试分类任务的效果
Rotation_test.py              # 图片旋转测试
Train_RotNet.py               # 训练RotNet
```

## 使用说明

1、打开Train_RotNet.py 修改数据集路径。

2、调整超参数进行旋转预训练。

3、在 Linear_Classfication.py 中导入预训练好的模型，进行图片分类测试。

4、使用 Tensorboard 打开 logs 文件夹查看模型训练情况