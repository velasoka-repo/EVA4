# Session-5 Assignment Task
1. Your new target is:
99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
2. Less than or equal to 15 Epochs
3. Less than 10000 Parameters
4. Do this in minimum 5 steps
5. Each File must have "target, result, analysis" TEXT block (either at the start or the end)
6. Explain your 5 steps using these target, results, and analysis with links to your GitHub files
7. Keep Receptive field calculations handy for each of your models. 

- - -

### Task-1) Basic CNN Setup [EVA4_S5_1.ipynb](https://github.com/velasoka-repo/EVA4/blob/master/Session-5/EVA4_S5_1.ipynb "EVA4-S5_1.ipynb")
#### Target:
1. Basic Transforms (ToTensor, Normalize)
2. Basic Data Loader (Train & Test)
3. Basic CNN Model & reducing parameter without fancy transforms
4. Basic Training  & Test Loop

#### Results:
1. Parameters: `1.6M`
2. Best Training Accuracy: `99.95%`
3. Best Test Accuracy: `99.26%`
4. Epochs: `15`

#### Analysis:
1. Simple Model with lots of parameter
2. Train Accuracy is increasing for each epoch
3. Test Accuracy is not stable (Accuracy is going UP & DOWN)
4. Seeing overfitting (Train: `99.95%`, Test: `99.08%`)

Task 1 [Colab Link](https://colab.research.google.com/github/velasoka-repo/EVA4/blob/master/Session-5/EVA4_S5_1.ipynb)

- - -


### Task-2) Basic CNN and reduce Hyper Parameter [EVA4_S5_2.ipynb](https://github.com/velasoka-repo/EVA4/blob/master/Session-5/EVA4_S5_2.ipynb "EVA4-S5_2.ipynb")
#### Target:
1. Basic Transforms (ToTensor, Normalize)
2. Basic Data Loader (Train & Test)
3. **Improving Basic CNN Model & trying to reducing parameter < 10K**
4. Adding Batch Normalization to increase Test Accuracy
5. Adding Dropout to reduce gap between Train & Test Accuracy 
6. Basic Training  & Test Loop

#### Results:
1. Parameters: `9.95K`
2. Best Training Accuracy: `99.04%`
3. Best Test Accuracy: `99.26%`
4. Epochs: `15`

#### *Analysis: BatchNorm after ReLU & Dropout (Best Model)
1. Achieved Target Parameters `9.95K` which is less than `10K`
2. Train Accuracy is increasing for each epoch
3. Test Accuracy is not stable (Accuracy is slightly going UP & DOWN)
4. Seeing bit of overfitting (Train: `99.04%`, Test: `99.20%`) 
5. Batch Normalized increased Test Accuracy a bit
6. Dropout reduces gap between Train & Test Accuracy (Reduced Overfitting)

#### Analysis: BatchNorm after ReLU
1. Very close to 10K parameter
2. Train Accuracy is increasing for each epoch
3. Test Accuracy is not stable (Accuracy is slightly going UP & DOWN)
4. Seeing overfitting (Train: `99.98%`, Test: `99.17%`)
5. Batch Normalized increased Test Accuracy a bit

#### Analysis: BatchNorm before ReLU
1. Very close to 10K parameter
2. Train Accuracy is increasing for each epoch
3. Test Accuracy is not stable (Accuracy is slightly going UP & DOWN)
4. Seeing overfitting (Train: `99.78%`, Test: `99.19%`) 
5. Batch Normalized increased Test Accuracy a bit

Task 2 [Colab Link](https://colab.research.google.com/github/velasoka-repo/EVA4/blob/master/Session-5/EVA4_S5_2.ipynb)

- - -

### Task-3) CNN with GAP [EVA4_S5_3.ipynb](https://github.com/velasoka-repo/EVA4/blob/master/Session-5/EVA4_S5_3.ipynb "EVA4-S5_3.ipynb")
#### Target:
1. Basic Transforms (ToTensor, Normalize)
2. Basic Data Loader (Train & Test)
3. Improved Basic CNN Model
4. Basic Training  & Test Loop
5. **Add GAP Layer** (It reduces Parameter)
6. **Improve Test Accuracy** (by increase model capacity)

#### *Results: (GAP Layer without Dropout)
1. Parameters: `9,866`
2. Best Training Accuracy: `99.71%` 
3. Best Test Accuracy: `99.43%`
4. Epochs: 15

#### *Analysis: (GAP Layer without Dropout)
1. Increasing model capacity is increasing Accuracy. Parameters `9,866`
2. Train Accuracy is increasing for each epoch
3. Test Accuracy closely stable (Accuracy is slightly going UP & DOWN 0.2%)
4. Seeing bit of Overfitting(Train: `99.71%`, Test: `99.43%`)


#### Results: (GAP Layer, 1 Dropout)
1. Parameters: `9,866`
2. Best Training Accuracy: `99.45%` 
3. Best Test Accuracy: `99.39%`
4. Epochs: 15

#### Analysis: (GAP Layer, 1 Dropout)
1. Increasing model capacity is increasing Accuracy. Parameters `9,866`
2. Train Accuracy is increasing for each epoch
3. Test Accuracy less stable (Accuracy is slightly going UP & DOWN 0.2%)
4. No Overfitting(Train: `99.45%`, Test: `99.39%`)

Task 3 [Colab Link](https://colab.research.google.com/github/velasoka-repo/EVA4/blob/master/Session-5/EVA4_S5_3.ipynb)

- - -

### Task-4) CNN with Data Augmentation [EVA4_S5_4.ipynb](https://github.com/velasoka-repo/EVA4/blob/master/Session-5/EVA4_S5_4.ipynb "EVA4-S5_4.ipynb")

#### Target:
1. Add **Data Augmentation**
2. Improve Test Accuracy

#### *Results: (Net without dropout)
1. Parameters: `9,866`
2. Best Training Accuracy: `99.23%` 
3. Best Test Accuracy: `99.44%`
4. Epochs: 15

#### *Analysis: (Net without dropout)
1. Using Augmentation we can achieve `99.44%` accuracy
2. Train Accuracy is increasing for each epoch
3. Seems to be good model(Train: `99.23%`, Test: `99.29%`) but Accuracy is not stable (Accuracy is slightly going UP & DOWN 0.5%)


#### Results: (Net2 with 0.1 dropout)
1. Parameters: `9,866`
2. Best Training Accuracy: `99.01%` 
3. Best Test Accuracy: `99.42%`
4. Epochs: 15

#### Analysis: (Net2 with 0.1 dropout)
1. Using Augmentation we can achieve `99.42%` accuracy
2. Train Accuracy is increasing for each epoch
3. Test Accuracy is increasing stable (No DOWN in accuracy)
4. Seeing bit of underfitting(Train: `99.01%`, Test: `99.42%`)

Task 4 [Colab Link](https://colab.research.google.com/github/velasoka-repo/EVA4/blob/master/Session-5/EVA4_S5_4.ipynb)

- - -

### Task-5) CNN with Data Augmentation [EVA4_S5_5.ipynb](https://github.com/velasoka-repo/EVA4/blob/master/Session-5/EVA4_S5_5.ipynb "EVA4-S5_5.ipynb")

#### Target:
1. Add **Learning Rate Scheduler**
2. Maintain Consistent Test Accuracy

#### *Results: (`Net` without dropout)
1. Parameters: `9,866`
2. Best Training Accuracy: `99.27%` 
3. Best Test Accuracy: `99.55%`
4. Epochs: 15

#### *Analysis: (`Net` without dropout)
1. Using LR scheduler, can maintain somewhat consistent accuracy `99.5x%` 
2. Train Accuracy is increasing for each epoch
3. Seeing bit of underfitting(Train: `99.27%`, Test: `99.54%`)
4. Achieved required Test Accuracy 

#### Results: (`Net2` with 0.1 dropout)
1. Parameters: `9,866`
2. Best Training Accuracy: `99.02%` 
3. Best Test Accuracy: `99.47%`
4. Epochs: 15

#### Analysis: (`Net2` with 0.1 dropout)
1. Using Augmentation we can achieve `99.47%` accuracy
2. Train Accuracy is increasing for each epoch
3. Test Accuracy is increasing stable (No DOWN in accuracy)
4. Seeing bit of underfitting(Train: `99.01%`, Test: `99.47%`)
5. Achieved required Test Accuracy 

Task 5 [Colab Link](https://colab.research.google.com/github/velasoka-repo/EVA4/blob/master/Session-5/EVA4_S5_5.ipynb)

- - -

### Final Test Accuracy

![Test Accuracy](https://github.com/velasoka-repo/EVA4/blob/master/Session-5/images/test-accuracy.png "Test Accuracy")


`*` - Desired Model Result & Analysis
