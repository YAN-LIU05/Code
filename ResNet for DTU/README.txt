文件夹结构如下，该代码实现了利用ResNet模型对Drone inspection images of wind turbine数据集中的图片进行分类。

/root
├── annotations
│   ├── test1024-s.json
│   ├── train1024-s.json
│   ├── val1024-s.json
│   └── test-HR.json
│
├── Nordtank 2017
├── Nordtank 2018
├── Nordtank_dealed
│   ├── DJI_0004_0_0.JPG
│   ├── ……
│
├── deal.py
├── data.py
├── model_resnet18.py
├── model_resnet50.py
├── model_torch_resnet.py
├── train.py 
├── requirments.txt
└── README.txt

Nordtank 2017 和 Nordtank 2018 取自 Drone inspection images of wind turbine 数据集，网址为：https://data.mendeley.com/datasets/hd96prn3nc/1
annotations 中的标注取自：https://github.com/imadgohar/DTU-annotations

运行：

python deal.py  # 处理分割数据集
python train.py  # 训练模型并检验

这里说明，在训练是要输入1、2、3来选择调用的模型：

choice == '1'，使用torchvision的ResNet模型
choice == '2'，使用自定义ResNet-18模型
choice == '3'，使用自定义ResNet-50模型
