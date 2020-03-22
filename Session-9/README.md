# Session-9 Assignment Task
1. Move your last code's transformations to Albumentations. Apply ToTensor, HorizontalFlip, Normalize (at min) + More (for additional points)
2. Please make sure that your test_transforms are simple and only using ToTensor and Normalize
3. Implement GradCam function as a module. 
4. Your final code (notebook file) must use imported functions to implement transformations and GradCam functionality
5. Target Accuracy is 87%

### File Description

[velasoka.py](https://github.com/velasoka-repo/EVA4/blob/master/Session-9/velasoka.py "velasoka.py") has all the required code to run model
- Its a helper/util file to simplify or reduce boiler plate code.

[velasoka_albumentations.py](https://github.com/velasoka-repo/EVA4/blob/master/Session-9/velasoka_albumentations.py "velasoka_albumentations.py") has all the required albumentation code

[velasoka_gradcam.py](https://github.com/velasoka-repo/EVA4/blob/master/Session-9/velasoka_gradcam.py "velasoka_gradcam.py") has all the required gradcam code

[velasoka_model.py](https://github.com/velasoka-repo/EVA4/blob/master/Session-9/velasoka_model.py "velasoka_model.py") has all the required ResNet18 code

[EVA4-S9.ipynb](https://github.com/velasoka-repo/EVA4/blob/master/Session-9/EVA4_S9.ipynb "EVA4-S9.ipynb") has the assignment code.



------------


#### [EVA4-S9.ipynb](https://github.com/velasoka-repo/EVA4/blob/master/Session-9/EVA4_S9.ipynb "EVA4-S9.ipynb")  has following code

1. **Loading models, albumentations from python files**
2. **Model summary of ResNet18 & GradCamHook**
3. **Albumentations Running Epochs**
4. **GradCam Visualize for CIFAR10 Test Dataset (10 right & wrong image prediction)**

------------

# Assignment Requirement

Total params: `11,173,962`
Trainable params: `11,173,962`

### Number of Epochs (only one time it reached 85% Test Accuracy, on next epoch it reduced to 84%)

**EPOCH: 25**

Training Batch=12499, loss=0.00054, Correct Prediction=49991/50000, `Train Accuracy=99.98200`: 100%|██████████| 12500/12500 [10:35<00:00, 19.67it/s]

Test Batch=2499, Correct Validation=8493/10000, `Test Accuracy=84.93000`: 100%|██████████| 2500/2500 [00:52<00:00, 48.08it/s]

[Google Colab File](https://colab.research.google.com/github/velasoka-repo/EVA4/blob/master/Session-9/EVA4_S9.ipynb)

# Correctly Predicted Image (GradCam Visualize)

![1](https://github.com/velasoka-repo/EVA4/blob/master/Session-9/images/right-prediction/right-prediction1.png)
![2](https://github.com/velasoka-repo/EVA4/blob/master/Session-9/images/right-prediction/right-prediction2.png)
![3](https://github.com/velasoka-repo/EVA4/blob/master/Session-9/images/right-prediction/right-prediction3.png)
![4](https://github.com/velasoka-repo/EVA4/blob/master/Session-9/images/right-prediction/right-prediction4.png)
![5](https://github.com/velasoka-repo/EVA4/blob/master/Session-9/images/right-prediction/right-prediction5.png)
![6](https://github.com/velasoka-repo/EVA4/blob/master/Session-9/images/right-prediction/right-prediction6.png)
![7](https://github.com/velasoka-repo/EVA4/blob/master/Session-9/images/right-prediction/right-prediction7.png)
![8](https://github.com/velasoka-repo/EVA4/blob/master/Session-9/images/right-prediction/right-prediction8.png)
![9](https://github.com/velasoka-repo/EVA4/blob/master/Session-9/images/right-prediction/right-prediction9.png)
![10](https://github.com/velasoka-repo/EVA4/blob/master/Session-9/images/right-prediction/right-prediction10.png)

# Wrongly Predicted Image (GradCam Visualize)

![1](https://github.com/velasoka-repo/EVA4/blob/master/Session-9/images/wrong-prediction/wrong-prediction1.png)
![2](https://github.com/velasoka-repo/EVA4/blob/master/Session-9/images/wrong-prediction/wrong-prediction2.png)
![3](https://github.com/velasoka-repo/EVA4/blob/master/Session-9/images/wrong-prediction/wrong-prediction3.png)
![4](https://github.com/velasoka-repo/EVA4/blob/master/Session-9/images/wrong-prediction/wrong-prediction4.png)
![5](https://github.com/velasoka-repo/EVA4/blob/master/Session-9/images/wrong-prediction/wrong-prediction5.png)
![6](https://github.com/velasoka-repo/EVA4/blob/master/Session-9/images/wrong-prediction/wrong-prediction6.png)
![7](https://github.com/velasoka-repo/EVA4/blob/master/Session-9/images/wrong-prediction/wrong-prediction7.png)
![8](https://github.com/velasoka-repo/EVA4/blob/master/Session-9/images/wrong-prediction/wrong-prediction8.png)
![9](https://github.com/velasoka-repo/EVA4/blob/master/Session-9/images/wrong-prediction/wrong-prediction9.png)
![10](https://github.com/velasoka-repo/EVA4/blob/master/Session-9/images/wrong-prediction/wrong-prediction10.png)






