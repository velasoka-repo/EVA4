# Session-12 Assignment
## Assignment A:
* Download this TINY IMAGENET (Links to an external site.) dataset. 
* Train ResNet18 on this dataset (70/30 split) for 50 Epochs. Target 50%+ Validation Accuracy. 
 
## Assignment B:
* Download 50 images of dogs. 
* Use [this](http://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html) to annotate bounding boxes around the dogs.
* Download JSON file. 
* Describe the contents of this JSON file in FULL details (you don't need to describe all 10 instances, anyone would work). 
 
### File Description

[velasoka.py](https://github.com/velasoka-repo/EVA4/blob/master/Session-12/velasoka.py "velasoka.py") has all the required code to run model
- Its a helper/util file to simplify or reduce boiler plate code.

[nn2.py](https://github.com/velasoka-repo/EVA4/blob/master/Session-12/model/nn2.py "nn.py") has ResNet code. 

[data.py](https://github.com/velasoka-repo/EVA4/blob/master/Session-12/utils/data.py "data.py") has TinyImageNet dataloader

* TinyImageNet train data is splitted into 70/30 to train & Test Accuracy

[runner.py](https://github.com/velasoka-repo/EVA4/blob/master/Session-12/utils/runner.py "runner.py") has model runner code to train & Test Data

[torch_util.py](https://github.com/velasoka-repo/EVA4/blob/master/Session-12/utils/torch_util.py "torch_util.py") has wrapper function for optimizer, OneCyclePolicy etc

[transform.py](https://github.com/velasoka-repo/EVA4/blob/master/Session-12/utils/transform.py "transform.py") has transformation like to_tensor, normalize etc

[visualize.py](https://github.com/velasoka-repo/EVA4/blob/master/Session-12/utils/visualize.py "visualize.py") has matplotlib util function

[EVA4-S12.ipynb](https://github.com/velasoka-repo/EVA4/blob/master/Session-12/EVA4_S12.ipynb "EVA4-S12.ipynb") has the assignment code.

[dog_via_project_18Apr2020_10h11m-formatted.json](https://github.com/velasoka-repo/EVA4/blob/master/Session-12/dog_via_project_18Apr2020_10h11m-formatted.json "dog_via_project_18Apr2020_10h11m-formatted.json") has 50 dog image bounding box

## Bounding Box Attribute Description
[attribute-description.json](https://github.com/velasoka-repo/EVA4/blob/master/Session-12/attribute-description.json "attribute-description.json") has bounding box attribute description


#### Train Accuracy
![train_accuracy.png](https://github.com/velasoka-repo/EVA4/blob/master/Session-12/images/train_accuracy.png "train_accuracy.png")

#### Test Accuracy
![test_accuracy.png](https://github.com/velasoka-repo/EVA4/blob/master/Session-12/images/test_accuracy.png "test_accuracy.png")

------------


#### [EVA4-S12.ipynb](https://github.com/velasoka-repo/EVA4/blob/master/Session-12/EVA4_S12.ipynb "EVA4-S12.ipynb")  has following code

1. **RandomHorizontalFlip, RandomCrop Image Augmentation methods**
2. **Model Train & Test Validation**

------------

# Assignment Requirement
Total params: `11,279,112`
Trainable params: `11,279,112`


### Number of Epochs (It reaches 29.25% Test Accuracy)

**EPOCH: 47**

Batch: 139, loss: 1.62, Train Accuracy: 62.64 [43849/70000]: 100%|██████████| 140/140 [00:21<00:00,  6.39it/s]
Batch: 59, Test Accuracy: 29.25: 100%|██████████| 60/60 [00:08<00:00,  7.32it/s]

[Google Colab File](https://colab.research.google.com/github/velasoka-repo/EVA4/blob/master/Session-12/EVA4_S12.ipynb)
