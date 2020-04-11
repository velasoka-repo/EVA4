# Session-11 Assignment

### File Description

[velasoka.py](https://github.com/velasoka-repo/EVA4/blob/master/Session-11/velasoka.py "velasoka.py") has all the required code to run model
- Its a helper/util file to simplify or reduce boiler plate code.

[nn.py](https://github.com/velasoka-repo/EVA4/blob/master/Session-11/model/nn.py "nn.py") has ResNet code. 

[augment.py](https://github.com/velasoka-repo/EVA4/blob/master/Session-11/utils/augment.py "augment.py") has augmentation code for Left Rotate, RandomCrop, Cutout

[data.py](https://github.com/velasoka-repo/EVA4/blob/master/Session-11/utils/data.py "data.py") has CIFAR10 dataloader

[runner.py](https://github.com/velasoka-repo/EVA4/blob/master/Session-11/utils/runner.py "runner.py") has model runner code to train & Test Data

[torch_util.py](https://github.com/velasoka-repo/EVA4/blob/master/Session-11/utils/torch_util.py "torch_util.py") has wrapper function for optimizer, OneCyclePolicy etc

[transform.py](https://github.com/velasoka-repo/EVA4/blob/master/Session-11/utils/transform.py "transform.py") has transformation like to_tensor, normalize etc

[visualize.py](https://github.com/velasoka-repo/EVA4/blob/master/Session-11/utils/visualize.py "visualize.py") has matplotlib util function

[EVA4-S11.ipynb](https://github.com/velasoka-repo/EVA4/blob/master/Session-11/EVA4_S11.ipynb "EVA4-S11.ipynb") has the assignment code.



#### Train Accuracy
[Train-Accuracy.png](https://github.com/velasoka-repo/EVA4/blob/master/Session-11/images/Train-Accuracy.png "Train-Accuracy.png")


------------

#### TriangleLR curve 
[triangler.png](https://github.com/velasoka-repo/EVA4/blob/master/Session-11/images/triangler.png "triangler.png")


------------


#### [EVA4-S11.ipynb](https://github.com/velasoka-repo/EVA4/blob/master/Session-11/EVA4_S11.ipynb "EVA4-S11.ipynb")  has following code

1. **RandomCrop, Cutout Image Augmentation methods**
2. **LR Finder & its Graph**
4. **Model Train & Test Validation**
5. **OneCycleLR code**

------------

# Assignment Requirement
Total params: `6,575,242`
Trainable params: `6,575,242`


### Number of Epochs (It reaches 87.63% Test Accuracy)

**EPOCH: 24**

Batch: 97, loss: 0.00, Train Accuracy: 100.00: 100%|██████████| 98/98 [00:24<00:00,  4.01it/s]
Batch: 99, Test Accuracy: 87.63: 100%|██████████| 100/100 [00:02<00:00, 37.42it/s]

[Google Colab File](https://colab.research.google.com/github/velasoka-repo/EVA4/blob/master/Session-11/EVA4_S11.ipynb)
