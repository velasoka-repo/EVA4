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
1. Parameters: 1.6M
2. Best Training Accuracy: `99.95%`
3. Best Test Accuracy: `99.26%`
4. Epochs: 15

#### Analysis:
1. Simple Model with lots of parameter
2. Train Accuracy is increasing for each epoch
3. Test Accuracy is not stable (Accuracy is going UP & DOWN)
4. Seeing overfitting (Train: `99.95%`, Test: `99.08%`)


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
1. Parameters: 9.95K
2. Best Training Accuracy: 99.04% 
3. Best Test Accuracy: 99.26%
4. Epochs: 15

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
