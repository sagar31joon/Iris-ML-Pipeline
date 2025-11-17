import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import time

# Loading preprocessed data
x_train = np.load("dataset/x_train.npy")
x_test = np.load("dataset/x_test.npy")
y_train = np.load("dataset/y_train.npy")
y_test = np.load("dataset/y_test.npy") 

# Loading models
with open("models/model_LR.pkl", "rb") as f:
    model_LR = pickle.load(f)
with open("models/model_SVC.pkl", "rb") as f: #loading svc model
    model_SVC = pickle.load(f)
with open("models/model_DT.pkl", "rb") as f: #loading DT model
    model_DT = pickle.load(f)

# Testing models
def model_testing(model, x_test, y_test):
    one_sample = x_test[0].reshape(1, -1) #inference time for single prediction

    # Batch inference time
    t1 = time.time()
    predictions = model.predict(x_test)
    t2 = time.time()

    # Single inference time
    t3 = time.time()
    model.predict(one_sample)
    t4 = time.time()

    # Accuracy and correct predidctions
    accuracy = accuracy_score(y_test, predictions)
    correct = (predictions == y_test).sum()
    total = len(y_test)
    batch_time = t2 - t1
    single_time = t4 - t3
    
    

    return accuracy, correct, total, single_time, batch_time

# Getting values for each model
acc_lr, cor_lr, tot_lr, one_lr, batch_lr = model_testing(model_LR, x_test, y_test)
acc_svc, cor_svc, tot_svc, one_svc, batch_svc = model_testing(model_SVC, x_test, y_test)
acc_dt, cor_dt, tot_dt, one_dt, batch_dt = model_testing(model_DT, x_test, y_test)

# Printing comparision table
print("\n" + "-"*95)
print("{:<20} {:<10} {:<15} {:<20} {:<15}".format(
    "Model", "Accuracy", "Correct", "Single Inference", "Batch Time"
))
print("-"*95)
print("{:<20} {:<10} {:<15} {:<20} {:<15}".format(
    "Logistic Regression",
    f"{acc_lr*100:.2f}%",
    f"{cor_lr}/{tot_lr}",
    f"{one_lr:.6f}s",
    f"{batch_lr:.6f}s"
))

print("{:<20} {:<10} {:<15} {:<20} {:<15}".format(
    "Decision Tree",
    f"{acc_dt*100:.2f}%",
    f"{cor_dt}/{tot_dt}",
    f"{one_dt:.6f}s",
    f"{batch_dt:.6f}s"
))

print("{:<20} {:<10} {:<15} {:<20} {:<15}".format(
    "Support Vector",
    f"{acc_svc*100:.2f}%",
    f"{cor_svc}/{tot_svc}",
    f"{one_svc:.6f}s",
    f"{batch_svc:.6f}s"
))
print("-"*95)

results = [
    {
        "Model": "Logistic Regression",
        "Accuracy": acc_lr * 100,
        "Correct": f"{cor_lr}/{tot_lr}",
        "Single_Inference_Time": one_lr,
        "Batch_Time": batch_lr
    },
    {
        "Model": "Decision Tree",
        "Accuracy": acc_dt * 100,
        "Correct": f"{cor_dt}/{tot_dt}",
        "Single_Inference_Time": one_dt,
        "Batch_Time": batch_dt
    },
    {
        "Model": "Support Vector",
        "Accuracy": acc_svc * 100,
        "Correct": f"{cor_svc}/{tot_svc}",
        "Single_Inference_Time": one_svc,
        "Batch_Time": batch_svc
    }
    
]

# Exporting comparision table
df = pd.DataFrame(results)
df.to_csv("dataset/model_comparision.csv", index = False)
print("\nSaved comparison as model_comparison.csv")