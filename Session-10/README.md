# Session-10 Assignment Task
1. Pick your last code
2. Make sure  to Add CutOut to your code. It should come from your transformations (albumentations)
 - Use this repo: https://github.com/davidtvs/pytorch-lr-finder (Links to an external site.) 
3. Move LR Finder code to your modules
4. Implement LR Finder (for SGD, not for ADAM)
 - Implement ReduceLROnPlatea: https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau (Links to an external site.)
5. Find best LR to train your model
6. Use SDG with Momentum
7. Train for 50 Epochs. 
8. Show Training and Test Accuracy curves
9. Target 88% Accuracy.
10. Run GradCAM on the any 25 misclassified images. Make sure you mention what is the prediction and what was the ground truth label.


### File Description

[velasoka.py](https://github.com/velasoka-repo/EVA4/blob/master/Session-10/velasoka.py "velasoka.py") has all the required code to run model
- Its a helper/util file to simplify or reduce boiler plate code.

[velasoka_albumentations.py](https://github.com/velasoka-repo/EVA4/blob/master/Session-10/velasoka_albumentations.py "velasoka_albumentations.py") has albumentation code. It uses `albumentation` library

[velasoka_gradcam.py](https://github.com/velasoka-repo/EVA4/blob/master/Session-10/velasoka_gradcam.py "velasoka_gradcam.py") has gradcam code. It adds heatmap to the image

[velasoka_model.py](https://github.com/velasoka-repo/EVA4/blob/master/Session-10/velasoka_model.py "velasoka_model.py") has ResNet18 code

[velasoka_lrfinder.py](https://github.com/velasoka-repo/EVA4/blob/master/Session-10/velasoka_lrfinder.py "velasoka_lrfinder.py") has LR Finder code. It uses `torch-lr-finder` model

[velasoka_notebook.py](https://github.com/velasoka-repo/EVA4/blob/master/Session-10/velasoka_notebook.py "velasoka_notebook.py") has wrapper function for show gradcam, running model etc

[EVA4-S10.ipynb](https://github.com/velasoka-repo/EVA4/blob/master/Session-10/EVA4_S10.ipynb "EVA4-S10.ipynb") has the assignment code.



------------


#### [EVA4-S10.ipynb](https://github.com/velasoka-repo/EVA4/blob/master/Session-10/EVA4_S10.ipynb "EVA4-S10.ipynb")  has following code

1. **Sample Augmentation Images**
2. **LR Finder & its Graph**
3. **GradCam heatmap Visualize for CIFAR10 Test Dataset (30 right & wrong image prediction)**
4. **Model Training with ReduceLROnPlateau & Augmentation Cutout**
5. **Extra Code to observe the Test Accuracy for various Augmentation**

------------

# Assignment Requirement

Total params: `11,173,962`
Trainable params: `11,173,962`

### Number of Epochs (It reaches 87.33% Test Accuracy)

**EPOCH: 50**

Training Batch=12499, loss=0.00007, Correct Prediction=50000/50000, `Train Accuracy=100.00000`: 100%|██████████| 12500/12500 [03:47<00:00, 55.03it/s]

Test Batch=2499, Correct Validation=8733/10000, `Test Accuracy=87.33000`: 100%|██████████| 2500/2500 [00:22<00:00, 112.85it/s]

[Google Colab File](https://colab.research.google.com/github/velasoka-repo/EVA4/blob/master/Session-10/EVA4_S10.ipynb)

# 30 Wrongly Predicted Image (GradCam Visualize)

![1](https://github.com/velasoka-repo/EVA4/blob/master/Session-10/images/wrong-prediction.png)

# 30 Correctly Predicted Image (GradCam Visualize)

![1](https://github.com/velasoka-repo/EVA4/blob/master/Session-10/images/right-prediction.png)

# Various Augmentation Tried to see How Augmentation changes the Image (like flip, blur, scale etc)
> 
![1](https://github.com/velasoka-repo/EVA4/blob/master/Session-10/images/augmentation-sample.png)
