# Session-7 Assignment Task
1. change the code such that it uses GPU
2. change the architecture to C1C2C3C40 (basically 3 MPs)
3. total RF must be more than 44
4. one of the layers must use Depthwise Separable Convolution
5. one of the layers must use Dilated Convolution
6. use GAP (compulsory):- add FC after GAP to target #of classes (optional)
7. achieve 80% accuracy, as many epochs as you want. Total Params to be less than 1M. 
8. upload to Github

### File Description

[velasoka.py](https://github.com/velasoka-repo/EVA4/blob/master/Session-7/velasoka.py "velasoka.py") has all the required code to run model
- Its a helper/util file to simplify or reduce boiler plate code.

[EVA4-S7.ipynb](https://github.com/velasoka-repo/EVA4/blob/master/Session-7/EVA4_S7.ipynb "EVA4-S7.ipynb") has the assignment code.



------------


#### [EVA4-S7.ipynb](https://github.com/velasoka-repo/EVA4/blob/master/Session-7/EVA4_S7.ipynb "EVA4-S7.ipynb")  has 3 models

1. **Simple Convolution Model** - it has simple code to setup basic building block of Neural Network
Model uses C1, C2, C3, C4 & O
> Conv2d

> BatchNorm2d

> ReLU

> MaxPool2d

> AvgPool2d


------------


2. **Dilated Convolution Model** - it has addition with `dilated` rate of 2 convolution on top of `Simple Convolution Model`
Model uses C1, C2, C3, C4 & O
> Conv2d

> BatchNorm2d

> ReLU

> MaxPool2d

> AvgPool2d

> `dilation=2`


------------


3. **Dilated & Depthwise Separable Convolution Model** - it has addition of `dilated`, `Depthwise` along with `point wise` convolution on top of `Simple Convolution Model`
Model uses C1, C2, C3, C4 & O
> Conv2d

> BatchNorm2d

> ReLU

> MaxPool2d

> AvgPool2d

> `dilation=2`

> groups=`<in_channel must be equally divisible by groups size>`


------------

# Assignment Requirement

Total params: `348,106`
Trainable params: `348,106`

### Number of Epochs

**EPOCH: 4**

Training Batch=12499, loss=0.39225, Correct Prediction=41309/50000, `Train Accuracy=82.61800`: 100%|██████████| 12500/12500 [01:59<00:00, 104.30it/s]

Test Batch=2499, Correct Validation=8084/10000, `Test Accuracy=80.84000`: 100%|██████████| 2500/2500 [00:14<00:00, 168.20it/s]


[Google Colab File](https://colab.research.google.com/github/velasoka-repo/EVA4/blob/master/Session-7/EVA4_S7.ipynb)
