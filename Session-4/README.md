# Session-4 Assignment Task
1.   99.4% validation accuracy 
2.   Less than 20k Parameters
3.   Less than 20 Epochs
4.   No fully connected layer
5.   LEARN HOW TO INTEGRATE THINGS LIKE DROPOUT, BATCHNORM, ETC

**Hint:**
> To learn how to add different things we covered in this session, you can refer to this code: https://www.kaggle.com/enwei26/mnist-digits-pytorch-cnn-99 (Links to an external site.) DONT COPY ARCHITECTURE, JUST LEARN HOW TO INTEGRATE THINGS LIKE DROPOUT, BATCHNORM, ETC.


### File Description

[EVA4-S4.ipynb](https://github.com/velasoka-repo/EVA4/blob/master/Session-9/EVA4_S9.ipynb "EVA4-S9.ipynb") has the assignment code.


### Model Summary

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 28, 28]             100
       BatchNorm2d-2           [-1, 10, 28, 28]              20
         Dropout2d-3           [-1, 10, 28, 28]               0
            Conv2d-4           [-1, 10, 28, 28]             910
       BatchNorm2d-5           [-1, 10, 28, 28]              20
         Dropout2d-6           [-1, 10, 28, 28]               0
         MaxPool2d-7           [-1, 10, 14, 14]               0
            Conv2d-8           [-1, 20, 14, 14]           1,820
       BatchNorm2d-9           [-1, 20, 14, 14]              40
        Dropout2d-10           [-1, 20, 14, 14]               0
           Conv2d-11           [-1, 20, 14, 14]           3,620
      BatchNorm2d-12           [-1, 20, 14, 14]              40
        Dropout2d-13           [-1, 20, 14, 14]               0
        MaxPool2d-14             [-1, 20, 7, 7]               0
        Dropout2d-15             [-1, 20, 7, 7]               0
           Conv2d-16             [-1, 20, 5, 5]           3,620
      BatchNorm2d-17             [-1, 20, 5, 5]              40
           Conv2d-18             [-1, 30, 3, 3]           5,430
      BatchNorm2d-19             [-1, 30, 3, 3]              60
           Conv2d-20             [-1, 10, 1, 1]           2,710
================================================================
Total params: 18,430
Trainable params: 18,430
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.58
Params size (MB): 0.07
Estimated Total Size (MB): 0.65
----------------------------------------------------------------


#### [EVA4-S4.ipynb](https://github.com/velasoka-repo/EVA4/blob/master/Session-4/EVA4_S4.ipynb "EVA4-S4.ipynb")  has following code

1. **Importing Required Packages**
2. **ToTensor & Normalize Transformation**
3. **Train & Test Dataset, Dataloader**
4. **Model with Batch Normalization & Dropout**
5. **Model Summary**
6. **Train & Test Function**
7. **Result of Model**

------------

# Assignment Requirement

Total params: `18,430`
Trainable params: `18,430`

### Number of Epochs (it reached 99.44% Test Accuracy)


**EPOCH: 20**

loss=0.00066 batch_id=468 `Train Accuracy=99.45: 100%|██████████| 469/469` [00:16<00:00, 29.18it/s]

Test set: Average loss: 0.0178, `Test Accuracy: 9944/10000 (99.44%)`


[Google Colab File](https://colab.research.google.com/github/velasoka-repo/EVA4/blob/master/Session-4/EVA4_S4.ipynb)
