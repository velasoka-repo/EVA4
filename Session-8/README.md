# Session-8 Assignment Task
1. Go through this repository: https://github.com/kuangliu/pytorch-cifar (Links to an external site.)
2. Extract the ResNet18 model from this repository and add it to your API/repo. 
3. Use your data loader, model loading, train, and test code to train ResNet18 on Cifar10
4. Your Target is 85% accuracy. No limit on the number of epochs. Use default ResNet18 code (so params are fixed). 

### File Description

[velasoka.py](https://github.com/velasoka-repo/EVA4/blob/master/Session-8/velasoka.py "velasoka.py") has all the required code to run model
- Its a helper/util file to simplify or reduce boiler plate code.

[EVA4-S8.ipynb](https://github.com/velasoka-repo/EVA4/blob/master/Session-8/EVA4_S8.ipynb "EVA4-S8.ipynb") has the assignment code.



------------


#### [EVA4-S8.ipynb](https://github.com/velasoka-repo/EVA4/blob/master/Session-8/EVA4_S8.ipynb "EVA4-S8.ipynb")  has 2 models

1. **ResNet18 with Fully Connected Layer** - its just replica of shared [github code](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py)

------------


2. **ResNet18 without Fully Connected Layer(Linear)** - existing shared [github code](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py) is modified to use `Conv2d` instead of `Linear`

------------
3. **ResNet18 without Fully Connected Layer(Linear) and with Dropout** - existing shared [github code](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py) is modified to use `Conv2d` instead of `Linear` and added `Dropout2d`

------------

# Assignment Requirement

Total params: `11,173,962`
Trainable params: `11,173,962`

### Number of Epochs (only one time it reached 85% Test Accuracy, on next epoch it reduced to 84%)

**EPOCH: 21**

Training Batch=12499, loss=0.00074, Correct Prediction=49958/50000, `Train Accuracy=99.91600`: 100%|██████████| 12500/12500 [03:35<00:00, 57.88it/s]

Test Batch=2499, Correct Validation=8514/10000, `Test Accuracy=85.14000`: 100%|██████████| 2500/2500 [00:25<00:00, 96.78it/s] 
  0%|          | 0/12500 [00:00<?, ?it/s]



[Google Colab File](https://colab.research.google.com/github/velasoka-repo/EVA4/blob/master/Session-8/EVA4_S8.ipynb)
