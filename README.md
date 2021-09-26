# Semantic-Aware-Video-Text-Detection

![image-20210923153053176](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20210923153053176.png)

## Introduction

This is a PyTorch implemntation of the CVPR 2021 paper [Semantic-Aware-Video-Text-Detection](https://openaccess.thecvf.com/content/CVPR2021/papers/Feng_Semantic-Aware_Video_Text_Detection_CVPR_2021_paper.pdf).

## Installation

The code is based on the [mmdetection(2.11.0)](https://github.com/open-mmlab/mmdetection/tree/v2.11.0) framework.

#### Requirements:

- Python3.6+
- PyTorch 1.3+ and torchvision that matches the Pytorch installation.
- CUDA 9.2+
- GCC 5+
- MMCV

```
# install the mmcv
pip install mmcv-full==1.3.9
# clone our model
git clone https://github.com/zjb-1/Semantic-Aware-Video-Text-Detection.git
# install the cocoapi
cd Semantic-Aware-Video-Text-Detection/cocoapi/PythonAPI
python setup.py build_ext install
# install our model
cd ../../
pip install -r requirements.txt
pip install -v -e .
```

## Models

If you need a pre-trained model or a trained model, you can contact me.

## Datasets

- The video datasets format is as follows:

  ```
  dataset
  ├─Video1
  │    ├─1.jpg
  │    ├─1.txt
  │    ├─2.jpg
  │    ├─2.txt
  │    └─...
  ├─Video2
  │    ├─1.jpg
  │    ├─1.txt
  │    ├─2.jpg
  │    ├─2.txt
  │    └─...
  ├─ ...    
  ```

  The txt file format is as follows(Coordinate points arranged clockwise, text, id):

  ​       x1,y1,x2,y2,x3,y3,x4,y4	text	id

- Then, you need to run the train_label_gen.py / test_label_gen.py to generate the label file.(Remember to modify the file path in the file).

## Training

Before training, you need to modify the profile(mask_track_rcnn_r50_fpn.py) and shell file(train.sh).

```
# training
bash train.sh
```

## Evaluation

Before evaluation, you need to modify the test shell file(test.sh).

```
# test
bash test.sh
```

You will get visual results.