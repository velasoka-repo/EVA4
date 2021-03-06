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

```
{
  "_via_img_metadata": {
    "n02085620_7.jpg8497": {          #image filename with size (filename=n02085620_7.jpg, size=8497)
      "filename": "n02085620_7.jpg",
      "size": 8497,
      "regions": [
        {
          "shape_attributes": {
            "name": "rect",     #used `rect`angle to draw bounding box
            "x": 71,            #x-distance from left
            "y": 3,             #y-distance from top
            "width": 113,       #width of the rectangle from x,y point (horizontal side towards right)
            "height": 190       #height of the rectangle from x,y point (vertical side towards down)
          },
          "region_attributes": {
            "name": "dog",        #highlighted image name (used bounding box to highlight `dog` regions)(only one name value `dog`)
            "color": "rgb",       #is it a RGB | grayscale image? (default value `RGB`)
            "quality": "good",    #quality of the image (default value `good`)
            "dog_only": "false"   #does image contains other than a dog (default value `false`)
          }
        }
      ],
      "file_attributes": {
        
      }
    }
  }
}
```

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

**EPOCH: 45**

Batch: 139, loss: 0.03, `Train Accuracy: 99.33` [69531/70000]: 100%|██████████| 140/140 [03:09<00:00,  1.35s/it]
Batch: 59, `Test Accuracy: 31.15`: 100%|██████████| 60/60 [00:07<00:00,  7.81it/s]


[Google Colab File](https://colab.research.google.com/github/velasoka-repo/EVA4/blob/master/Session-12/EVA4_S12.ipynb)
