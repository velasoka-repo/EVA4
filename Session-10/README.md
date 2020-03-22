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
3. **GradCam heatmap Visualize for CIFAR10 Test Dataset (10 right & wrong image prediction)**
4. **Model Training with ReduceLROnPlateau & Augmentation Cutout**
5. **Extra Code to observe the Test Accuracy as per Augmentation**

------------

# Assignment Requirement

Total params: `11,173,962`
Trainable params: `11,173,962`

### Number of Epochs (only one time it reached 85% Test Accuracy, on next epoch it reduced to 84%)

**EPOCH: 25**

Training Batch=12499, loss=0.00054, Correct Prediction=49991/50000, `Train Accuracy=99.98200`: 100%|██████████| 12500/12500 [10:35<00:00, 19.67it/s]

Test Batch=2499, Correct Validation=8493/10000, `Test Accuracy=84.93000`: 100%|██████████| 2500/2500 [00:52<00:00, 48.08it/s]

[Google Colab File](https://colab.research.google.com/github/velasoka-repo/EVA4/blob/master/Session-10/EVA4_S9.ipynb)

# 30 Wrongly Predicted Image (GradCam Visualize)

![1](https://github.com/velasoka-repo/EVA4/blob/master/Session-10/images/wrong-prediction.png)

# 30 Correctly Predicted Image (GradCam Visualize)

![1](https://github.com/velasoka-repo/EVA4/blob/master/Session-10/images/right-prediction.png)

# Albumentation Sample Image Augmentation 
> 
![1](https://github.com/velasoka-repo/EVA4/blob/master/Session-10/images/augmentation-sample.png)
