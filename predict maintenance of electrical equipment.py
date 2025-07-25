# Predictive Maintenance for Electrical Equipment: Machine Learning Classifier Evalution
#importing dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import os
from sklearn.linear_model import PassiveAggressiveClassifier
#uploading dataset
df = pd.read_csv(r'dataset\data.csv')
df.head()

#data analysis
df.info()
df.describe()
#data Correlection
df.corr()
#Checking NULL values
df.isnull().sum()
df = df.drop(['id'], axis = 1)
df
Labels = ['bearings','wpump','radiator','exvalve','acmotor']
for i in Labels:
    df[i] = LabelEncoder().fit_transform(df[i])
df
df.info()
labels = ['Clean','Dirty']
labels
#Data Visulazation
sns.set(style="darkgrid") 
plt.figure(figsize=(12, 6)) 
ax = sns.countplot(x=df['radiator'], data=df, palette="Set3")
plt.title("Count Plot")  
plt.xlabel("Categories") 
plt.ylabel("Count") 
ax.set_xticklabels(labels)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')
plt.show()  
df
#Declaring independent and dependent variable
x = df.drop(['radiator'],axis = 1)
x.head()
y = df['radiator']
y
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.10, random_state = 42)
x_train.shape
y_train.shape
#performance evalution
precision = []
recall = []
fscore = []
accuracy = []
def performance_metrics(algorithm, predict, testY):
    testY = testY.astype('int')
    predict = predict.astype('int')
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100 
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    print(algorithm+' Accuracy    : '+str(a))
    print(algorithm+' Precision   : '+str(p))
    print(algorithm+' Recall      : '+str(r))
    print(algorithm+' FSCORE      : '+str(f))
    report=classification_report(predict, testY,target_names=labels)
    print('\n',algorithm+" classification report\n",report)
    conf_matrix = confusion_matrix(testY, predict) 
    plt.figure(figsize =(5, 5)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="Blues" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()
# Passive Aggressive Classifier model building
pa_model_path = 'model/PassiveAggressiveClassifier.npy'
if os.path.exists(pa_model_path):
    # Load the Passive Aggressive Classifier model
    pa_classifier = np.load(pa_model_path, allow_pickle=True).item()
else:                       
    # Train and save the Passive Aggressive Classifier model
    pa_classifier = PassiveAggressiveClassifier()
    pa_classifier.fit(x_train, y_train)
    np.save(pa_model_path, pa_classifier)
# Predict using the trained Passive Aggressive Classifier model
y_pred_pa = pa_classifier.predict(x_test)
# Evaluate the Passive Aggressive Classifier model
performance_metrics('PassiveAggressiveClassifier', y_pred_pa, y_test)
#Decision Tree Classifier model building
from sklearn.tree import DecisionTreeClassifier
# Check if the model file exists
model_path = 'model/DecisionTreeClassifier.npy'
if os.path.exists(model_path):
    # Load the model
    classifier = np.load(model_path, allow_pickle=True).item()
else:                       
    # Train and save the model
    classifier = DecisionTreeClassifier()
    classifier.fit(x_train, y_train)
    np.save(model_path, classifier)

# Predict using the trained model
y_pred = classifier.predict(x_test)

# Evaluate the model
performance_metrics('DecisionTreeClassifier', y_pred, y_test)
#Tabular form of Performance Metrics
#showing all algorithms performance values
columns = ["Algorithm Name","Precison","Recall","FScore","Accuracy"]
values = []
algorithm_names = ["Passive Aggressive Classifier", "Decision Tree Classifier"]
for i in range(len(algorithm_names)):
    values.append([algorithm_names[i],precision[i],recall[i],fscore[i],accuracy[i]])
    
temp = pd.DataFrame(values,columns=columns)
temp
#Uploading testing dataset
test=pd.read_csv("test.csv")
test
Test_Labels = ['bearings','wpump','exvalve','acmotor']

for i in Test_Labels:
    test[i] = LabelEncoder().fit_transform(test[i])
test
#Model prediction on test data
predict = classifier.predict(test)

for i, p in enumerate(predict):
    if p == 0:
        print(test.iloc[i]) 
        print("Model Predicted of Row {} Test Data is--->".format(i),labels[0])
    elif p == 1:
        print(test.iloc[i])  
        print("Model Predicted of Row {} Test Data is--->".format(i),labels[1])
