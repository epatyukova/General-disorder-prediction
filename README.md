# General-disorder-prediction
At the most basic level all compounds can be divided into two categories: ordered and disordered, judging by the presence of the partial occupancies in CIF. Here we try to find the best machine learning model which would predict disorder from the composition.

Python environment file is attached (sorry, I probably should have created specific environment for this project, but I did not)

The code is implemented in pytorch lightning, logging is to wandb (so do not forget to specify you key)

Trained models can be found on my group's external hard drive (beware formated for MAC, does not work in Windows)


**List of models:**

**Random Forrest + Magpie**
 Balanced Accuracy = 0.87, Recall = 0.9, Precision = 0.9, ROC AUC = 0.94, MCC (Matthews correlation coefficient) = 0.74
 
**3-NN + scaled Magpie**
Balanced Accuracy = 0.79, Recall = 0.83, Precision = 0.83, ROC AUC = 0.86, MCC (Matthews correlation coefficient) = 0.58

**Roost + Matscholar**
Balanced Accuracy = 0.84, Recall = 0.83, Precision = 0.89, ROC AUC = 0.91, MCC (Matthews correlation coefficient) = 0.67

**CrabNet + Mat2vec**
Balanced Accuracy = 0.90, Recall = 0.88, Precision = 0.94, ROC AUC = 0.95, MCC (Matthews correlation coefficient) = 0.79

**Ensemble-10Roost + Matscholar**
Balanced Accuracy = 0.86, Recall = 0.84, Precision = 0.92, ROC AUC = 0.93, MCC (Matthews correlation coefficient) = 0.71

**Ensemble-10CrabNet + Matscholar**
Balanced Accuracy = 0.91, Recall = 0.89, Precision = 0.95, ROC AUC = 0.97, MCC (Matthews correlation coefficient) = 0.82

**Blending (Logistic regression on output of all classifiers)**
Balanced Accuracy = 0.90, Recall = 0.91, Precision = 0.93, ROC AUC = 0.96, MCC (Matthews correlation coefficient) = 0.80

**Transfer model: CrabNet trained on formation energy transfered to predict disorder**

No gains compared to simple CrabNet

**Multi-property model: CrabNet encoder with two projection heads predicting disorder and formation energy.**

Trained on disorder data + MP (entry can have only one of those two values to be included in teh dataset). No gains compared to simple CrabNet
