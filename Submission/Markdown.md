# Team Data Science

Link to the tutorial:  https://data-flair.training/blogs/python-machine-learning-project-detecting-parkinson-disease/
## Objective: 
Data Prediction Models on Parkinson’s Disease Data using Machine Learning Algorithms in Python.<br>

![image.png](attachment:image.png)

### Our Approach:

Initially, we used the XGBoost algorithm for model preparation, afterwards we compared the efficiency of XGBoost with other algorithms to check for the highest accuracy. This approach to handling these problems will provide greater clarity about the data, it’s features and the best fit algorithm. 


```python
import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import seaborn as sn
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
```

### About the Data:
Parkinson’s disease is a progressive disorder of the central nervous system affecting movement and inducing tremors and stiffness. It has 5 stages. This is chronic and has no cure yet. It is a neurodegenerative disorder affecting dopamine-producing neurons in the brain.<br>
We took the UCI ML Parkinsons dataset for this. The dataset has 24 columns and 195 records and is only 39.7 KB.<br>
Data Set Characteristics: Multivariate<br>
Number of Instances: 197<br>
Area: Life<br>
Attribute Characteristics: Real<br>
Number of Attributes: 23<br>
Date Donated: 2008-06-26<br>
Associated Tasks: Classification<br>
Missing Values? N/A<br>

### Data Set Information:

This dataset is composed of a range of biomedical voice measurements from 31 people, 23 with Parkinson's disease (PD). Each column in the table is a particular voice measure, and each row corresponds to one of 195 voice recordings from these individuals ("name" column). The main aim of the data is to discriminate healthy people from those with PD, according to the "status" column which is set to 0 for healthy and 1 for PD.

The data is in ASCII CSV format. The rows of the CSV file contain an instance corresponding to one voice recording. There are around six recordings per patient, the name of the patient is identified in the first column.


### Attribute Information:

Matrix column entries (attributes):<br>
name - ASCII subject name and recording number<br>
MDVP:Fo(Hz) - Average vocal fundamental frequency.<br>
MDVP:Fhi(Hz) - Maximum vocal fundamental frequency.<br>
MDVP:Flo(Hz) - Minimum vocal fundamental frequency.<br>
MDVP:Jitter(%),MDVP:Jitter(Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP - Several measures of variation in fundamental frequency.<br>
MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA - Several measures of variation in amplitude.<br>
NHR,HNR - Two measures of ratio of noise to tonal components in the voice status - Health status of the subject (one) - Parkinson's, (zero) - healthy.<br>
RPDE,D2 - Two nonlinear dynamical complexity measures.<br>
DFA - Signal fractal scaling exponent.<br>
spread1,spread2,PPE - Three nonlinear measures of fundamental frequency variation.<br>


```python
df=pd.read_csv('../Downloads/parkinsons.data')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>MDVP:Fo(Hz)</th>
      <th>MDVP:Fhi(Hz)</th>
      <th>MDVP:Flo(Hz)</th>
      <th>MDVP:Jitter(%)</th>
      <th>MDVP:Jitter(Abs)</th>
      <th>MDVP:RAP</th>
      <th>MDVP:PPQ</th>
      <th>Jitter:DDP</th>
      <th>MDVP:Shimmer</th>
      <th>...</th>
      <th>Shimmer:DDA</th>
      <th>NHR</th>
      <th>HNR</th>
      <th>status</th>
      <th>RPDE</th>
      <th>DFA</th>
      <th>spread1</th>
      <th>spread2</th>
      <th>D2</th>
      <th>PPE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>phon_R01_S01_1</td>
      <td>119.992</td>
      <td>157.302</td>
      <td>74.997</td>
      <td>0.00784</td>
      <td>0.00007</td>
      <td>0.00370</td>
      <td>0.00554</td>
      <td>0.01109</td>
      <td>0.04374</td>
      <td>...</td>
      <td>0.06545</td>
      <td>0.02211</td>
      <td>21.033</td>
      <td>1</td>
      <td>0.414783</td>
      <td>0.815285</td>
      <td>-4.813031</td>
      <td>0.266482</td>
      <td>2.301442</td>
      <td>0.284654</td>
    </tr>
    <tr>
      <th>1</th>
      <td>phon_R01_S01_2</td>
      <td>122.400</td>
      <td>148.650</td>
      <td>113.819</td>
      <td>0.00968</td>
      <td>0.00008</td>
      <td>0.00465</td>
      <td>0.00696</td>
      <td>0.01394</td>
      <td>0.06134</td>
      <td>...</td>
      <td>0.09403</td>
      <td>0.01929</td>
      <td>19.085</td>
      <td>1</td>
      <td>0.458359</td>
      <td>0.819521</td>
      <td>-4.075192</td>
      <td>0.335590</td>
      <td>2.486855</td>
      <td>0.368674</td>
    </tr>
    <tr>
      <th>2</th>
      <td>phon_R01_S01_3</td>
      <td>116.682</td>
      <td>131.111</td>
      <td>111.555</td>
      <td>0.01050</td>
      <td>0.00009</td>
      <td>0.00544</td>
      <td>0.00781</td>
      <td>0.01633</td>
      <td>0.05233</td>
      <td>...</td>
      <td>0.08270</td>
      <td>0.01309</td>
      <td>20.651</td>
      <td>1</td>
      <td>0.429895</td>
      <td>0.825288</td>
      <td>-4.443179</td>
      <td>0.311173</td>
      <td>2.342259</td>
      <td>0.332634</td>
    </tr>
    <tr>
      <th>3</th>
      <td>phon_R01_S01_4</td>
      <td>116.676</td>
      <td>137.871</td>
      <td>111.366</td>
      <td>0.00997</td>
      <td>0.00009</td>
      <td>0.00502</td>
      <td>0.00698</td>
      <td>0.01505</td>
      <td>0.05492</td>
      <td>...</td>
      <td>0.08771</td>
      <td>0.01353</td>
      <td>20.644</td>
      <td>1</td>
      <td>0.434969</td>
      <td>0.819235</td>
      <td>-4.117501</td>
      <td>0.334147</td>
      <td>2.405554</td>
      <td>0.368975</td>
    </tr>
    <tr>
      <th>4</th>
      <td>phon_R01_S01_5</td>
      <td>116.014</td>
      <td>141.781</td>
      <td>110.655</td>
      <td>0.01284</td>
      <td>0.00011</td>
      <td>0.00655</td>
      <td>0.00908</td>
      <td>0.01966</td>
      <td>0.06425</td>
      <td>...</td>
      <td>0.10470</td>
      <td>0.01767</td>
      <td>19.649</td>
      <td>1</td>
      <td>0.417356</td>
      <td>0.823484</td>
      <td>-3.747787</td>
      <td>0.234513</td>
      <td>2.332180</td>
      <td>0.410335</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



### Data Preprocessing:
Data is loaded, features and label variables are separated out and whole data is TRANSFORMED. 
Splitting the data into train and test sets is done by the function present in sci-kit learn library. The train-test split procedure is used to estimate the performance of machine learning algorithms when they are used to make predictions on data not used to train the model. 


```python
features = df.loc[:,df.columns!='status'].values[:,1:]
labels = df.loc[:,'status'].values
scaler = MinMaxScaler((-1,1))
x = scaler.fit_transform(features)
y = labels
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state = 7)
```

## XGBoost Classifier:
XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. In prediction problems involving unstructured data (images, text, etc.) artificial neural networks tend to outperform all other algorithms or frameworks.


```python
model = XGBClassifier()
model.fit(x_train,y_train)
```

    [07:47:04] WARNING: ../src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.





    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                  importance_type='gain', interaction_constraints='',
                  learning_rate=0.300000012, max_delta_step=0, max_depth=6,
                  min_child_weight=1, missing=nan, monotone_constraints='()',
                  n_estimators=100, n_jobs=4, num_parallel_tree=1, random_state=0,
                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
                  tree_method='exact', validate_parameters=1, verbosity=None)




```python
y_pred = model.predict(x_test)
print("Accuracy %: ", accuracy_score(y_test, y_pred)*100)
```

    Accuracy %:  94.87179487179486



```python
cm = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None, normalize=None)
diseased_freq_avg = df[df["status"] == 1]["MDVP:Fo(Hz)"].values
healthy_freq_avg = df[df["status"] == 0]["MDVP:Fo(Hz)"].values

fig, axis = plt.subplots(1,2, figsize=(20, 7))

# Confusion Matix Heatmap
sn.heatmap(cm, annot=True, fmt='d', cmap="Greens", ax=axis[0])
axis[0].set(xlabel='Predicted', ylabel='Actual')

# Histogram plot
sn.distplot(diseased_freq_avg, hist=True, label="Parkinson's Disease Cases")
sn.distplot(healthy_freq_avg, hist=True, label="Healthy Cases")
plt.title("Average vocal fundamental frequency MDVP:Fo(Hz) Distribution plot")
```




    Text(0.5, 1.0, 'Average vocal fundamental frequency MDVP:Fo(Hz) Distribution plot')




    
![png](Markdown_files/Markdown_9_1.png)
    


# Analysis of Data Using Other Algorithms:

### Function used by algorithms to run and plot graphs.


```python
def Model(model, x_train,x_test,y_train,y_test, graph) :
    model.fit(x_train,y_train)
    prediction = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    print("Accuracy %: ", accuracy_score(y_test, prediction)*100)

    cm = confusion_matrix(y_test, prediction)
    
    print(classification_report(y_test, prediction))
  
    F,T,thresholds = roc_curve(y_test, probabilities[:,1])
    
    if graph:
        fig, axis = plt.subplots(1,2, figsize=(20, 7))
        
        # HeatMAP
        sn.heatmap(cm, fmt='', annot = True, cmap="gray_r", ax=axis[0])
        axis[0].set(xlabel='Predicted', ylabel='Actual')
        
        # Lineplot
        plt.title('Receiver Operating Characteristic')
        sn.lineplot(F, T, ax=axis[1])
        axis[1].set(xlabel = 'False Positive Rate', ylabel = 'True Positive Rate')
        plt.show()
```


```python
df_train = df.copy().drop(columns=["name"])
columns, primary = df_train.columns.tolist(), ["status"]
columns.remove(primary[0])
df_train = df_train[columns + primary]

std = StandardScaler()
scaled = pd.DataFrame(std.fit_transform(df_train[columns]), columns=columns)
df_train = pd.concat([scaled, df["status"]], axis=1)
```


```python
X, Y = df_train[columns], df_train[primary]
```


```python
train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size = 0.3, random_state = 10)
train_y, test_y = train_y["status"], test_y["status"]
```

## Logistic Regression:
Logistic regression is a supervised learning classification algorithm used to predict the probability of a target variable. The nature of target or dependent variable is dichotomous, which means there would be only two possible classes.


```python
lr = LogisticRegression()
Model(lr, train_X, test_X, train_y, test_y, graph = True)
```

    Accuracy %:  89.83050847457628
                  precision    recall  f1-score   support
    
               0       0.92      0.69      0.79        16
               1       0.89      0.98      0.93        43
    
        accuracy                           0.90        59
       macro avg       0.91      0.83      0.86        59
    weighted avg       0.90      0.90      0.89        59
    



    
![png](Markdown_files/Markdown_17_1.png)
    


## Support Vector Machine:
Support vector machines (SVMs) are powerful yet flexible supervised machine learning algorithms which are used both for classification and regression. An SVM model is basically a representation of different classes in a hyperplane in multidimensional space. The hyperplane will be generated in an iterative manner by SVM so that the error can be minimized. The goal of SVM is to divide the datasets into classes to find a maximum marginal hyperplane (MMH).


```python
svm = SVC(probability=True)
Model(svm, train_X, test_X, train_y, test_y, graph = True)
```

    Accuracy %:  84.7457627118644
                  precision    recall  f1-score   support
    
               0       1.00      0.44      0.61        16
               1       0.83      1.00      0.91        43
    
        accuracy                           0.85        59
       macro avg       0.91      0.72      0.76        59
    weighted avg       0.87      0.85      0.82        59
    



    
![png](Markdown_files/Markdown_19_1.png)
    


## Gaussian Naive Bayes:
Naïve Bayes algorithm is a classification technique based on applying Bayes’ theorem with a strong assumption that all the predictors are independent of each other. In simple words, the assumption is that the presence of a feature in a class is independent of the presence of any other feature in the same class.


```python
gnb = GaussianNB()
Model(gnb, train_X, test_X, train_y, test_y, graph = True)
```

    Accuracy %:  77.96610169491525
                  precision    recall  f1-score   support
    
               0       0.55      1.00      0.71        16
               1       1.00      0.70      0.82        43
    
        accuracy                           0.78        59
       macro avg       0.78      0.85      0.77        59
    weighted avg       0.88      0.78      0.79        59
    



    
![png](Markdown_files/Markdown_21_1.png)
    


## K-Neighbour Classifier:
K-nearest neighbors (KNN) algorithm is a type of supervised ML algorithm which can be used for both classification as well as regression predictive problems. However, it is mainly used for classification of predictive problems in industry.


```python
kn = neighbors.KNeighborsClassifier()
Model(kn, train_X, test_X, train_y, test_y, graph = True)
```

    Accuracy %:  86.4406779661017
                  precision    recall  f1-score   support
    
               0       0.90      0.56      0.69        16
               1       0.86      0.98      0.91        43
    
        accuracy                           0.86        59
       macro avg       0.88      0.77      0.80        59
    weighted avg       0.87      0.86      0.85        59
    



    
![png](Markdown_files/Markdown_23_1.png)
    


## Random Forest Classifier:
Random Forest is a popular machine learning algorithm that belongs to the supervised learning  technique. It can be used for both Classification and Regression problems in ML. It is based     on the concept of ensemble learning, which is a process of combining multiple classifiers to solve a complex problem and to improve the performance of the model.


```python
rfc = RandomForestClassifier()
Model(rfc, train_X, test_X, train_y, test_y, graph = True)
```

    Accuracy %:  93.22033898305084
                  precision    recall  f1-score   support
    
               0       0.93      0.81      0.87        16
               1       0.93      0.98      0.95        43
    
        accuracy                           0.93        59
       macro avg       0.93      0.89      0.91        59
    weighted avg       0.93      0.93      0.93        59
    



    
![png](Markdown_files/Markdown_25_1.png)
    


## Decision Tree Classifier:
Decision tree analysis is a predictive modelling tool that can be applied across many areas. Decision trees can be constructed by an algorithmic approach that can split the dataset in different ways based on different conditions. Decisions trees are the most powerful algorithms that falls under the category of supervised algorithms. They can be used for both classification and regression tasks.



```python
dt = DecisionTreeClassifier()
Model(dt, train_X, test_X, train_y, test_y, graph = True)
```

    Accuracy %:  84.7457627118644
                  precision    recall  f1-score   support
    
               0       0.73      0.69      0.71        16
               1       0.89      0.91      0.90        43
    
        accuracy                           0.85        59
       macro avg       0.81      0.80      0.80        59
    weighted avg       0.84      0.85      0.85        59
    



    
![png](Markdown_files/Markdown_27_1.png)
    


## Bagging Classifier:
A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it.


```python
bc = BaggingClassifier()
Model(bc, train_X, test_X, train_y, test_y, graph = True)
```

    Accuracy %:  89.83050847457628
                  precision    recall  f1-score   support
    
               0       0.92      0.69      0.79        16
               1       0.89      0.98      0.93        43
    
        accuracy                           0.90        59
       macro avg       0.91      0.83      0.86        59
    weighted avg       0.90      0.90      0.89        59
    



    
![png](Markdown_files/Markdown_29_1.png)
    


# Plots:


```python
status_value_counts = df['status'].value_counts()
print("Number of Parkinson's Disease patients: {} ({:.2f}%)".format(status_value_counts[1], status_value_counts[1] / df.shape[0] * 100))
print("Number of Healthy patients: {} ({:.2f}%)".format(status_value_counts[0], status_value_counts[0] / df.shape[0] * 100))
```

    Number of Parkinson's Disease patients: 147 (75.38%)
    Number of Healthy patients: 48 (24.62%)


## Plot Representing Healthy Patient VS Parkinson's Disease patients:


```python
sn.countplot(df['status'].values)
plt.xlabel("Status value")
plt.ylabel("Number of cases")
plt.show()
```


    
![png](Markdown_files/Markdown_33_0.png)
    


## Bar Plot for Visualizing Important Features:


```python
feature_imp = pd.Series(rfc.feature_importances_,index=X.columns).sort_values(ascending=False)
sn.barplot(x=feature_imp, y=feature_imp.index)

plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()
```


    
![png](Markdown_files/Markdown_35_0.png)
    


## Average Vocal Fundamental Frequency MDVP:Fo(Hz) Boxplot:


```python
diseased_freq_avg = df[df["status"] == 1]["MDVP:Fo(Hz)"].values
healthy_freq_avg = df[df["status"] == 0]["MDVP:Fo(Hz)"].values

plt.boxplot([diseased_freq_avg, healthy_freq_avg])
plt.title("Average vocal fundamental frequency MDVP:Fo(Hz) Box plot")
plt.xticks([1, 2], ["Parkinson's Disease Cases", "Healthy Cases"])
plt.show()
```


    
![png](Markdown_files/Markdown_37_0.png)
    


## Comparing Accuracy of All Models:


|S.no | MODEL | ACCURACY |
|---:|:-------------|:------|
| 1 | XGBoost | 94.87179487179486 |
| 2 | Logistic Regression | 89.83050847457628 |
| 3 | Support Vector Machine | 84.7457627118644 |
| 4 | Guassian Naive Bayes | 77.96610169491525 |
| 5 | K Neighbour | 86.4406779661017 |
| 6 | Random Forest | 93.22033898305084 |
| 7 | Decision Tree | 84.7457627118644 |
| 8 | Bagging Algorithm | 89.83050847457628 |

### According to the accuracy table and graphs, XGBoost gives the most valuable results as it has the highest accuracy ( 94.87179487179486% ). From the graph and confusion matrix, the observation is further supported. Thus, we can build a reliable model for prediction using XGBoost.


## Contribution:

|S.no | Name | SlackUsername | Contribution | Partners |
|---:|:-------------|:-----------|:------|:------|
| 1 | Sanniya Middha  | @Sanniya01       | Logistic Regression   | Rachna Behl (@Rachna)     |
| 2 | Sophie Fang | @Sophie | Random Forest, Heat map, Feature importance map, Analysis of plot differences |   |
| 3 | Shalini Gupta | @Miss_IndoriDelight | Logistic Regression,K-Nearest Neighbors(Including Scatter Plot) and Support Vector | Bhavya Saini(@Bhavyasind) |
| 4 | Shruti Poojary | @ShrutiP | Random Forest (including FacetGrid plots) |   |
| 5 | Bhushan Wagh | @XR2 | Logistic Regression, K-Nearest Neighbors, Gaussian Naïve Bayes, Support Vector, Stacking, Decision Tree, Bagging, Random Forest (Including Scatter Plot) Histograms, Box, Bar, PairPlot, HeatMap |   |
| 6 | Ikechukwu Okoye | @Ikechukwu | Support Vector and Naive Bayes Classifier with necessary accuracy metrics and plots | @Mercii |
| 7 | Arinola | @Arinola | Wrote out the team's project protocol for the advertisement submission, Logistic linear regression | @Dibyendu1153533 |
| 8 | David Guevara-Apaza | @yoodavoo | EDA, pie chart, correlation and  heatmap, histograms, XGB, Bagging Algorithm, feature importance, ROC | Chukwuemelie Aginah @Chukwu_emeliela |
| 9 | Prathamesh Bobale | @Pratham99 | Scatter plot, logistic regression |   |
| 10 | Dibyendu Biswas | @Dibyendu11153533 | Workflow Advertisement designing, Logistic linear regression | @ShrutiP |
| 11 | Anirudh | @-anirudh1009- | Gaussian Naive Bayes, K-Nearest Neighbour |   |
| 12 | Aginah Chukwuemelie | @Chukwu_emeliela | EDA, pie chart, correlation and  heatmap, histograms, Bagging Algorithm, feature importance, ROC | David Guevara-Apaza |
| 13 | Foluso Ogunfile | @fogunfile | Logistic regression, nearest centroid classifier |   |
| 14 | Bhavya Saini | @Bhavyasind | Logistic & K-Nearest Neighbors : Scatter Plot, Write up for Markdown | Shalini Gupta |
| 15 | Vinay Joshi | @vinyjoshi | Github repo, All algorithms, Code for Markdown. |   |
| 16 | Team |   | All, XGBoost, Tutorials search | All |



## Check the Github Repo for detailed analysis and everyone's contribution to the work.
