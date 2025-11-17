#Logistic Refression model and sclaler model initialiser
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

# Load preprocessed data
x_train = np.load("dataset/x_train.npy")
x_test = np.load("dataset/x_test.npy")
y_train = np.load("dataset/y_train.npy")
y_test = np.load("dataset/y_test.npy")

#initialising model
model_LR = LogisticRegression(max_iter=500) 
# Training time taken for model to train
train_time_start = time.time()
model_LR.fit(x_train, y_train) #training model
train_time_end = time.time()

#inference time taken for model to predict whole set
inf_time_start = time.time()
predict = model_LR.predict(x_test) #testing x_test 
inf_time_end = time.time()

#inference time for single prediction
one_sample = x_test[0].reshape(1, -1)
one_inf_time_start = time.time()
model_LR.predict(one_sample)
one_inf_time_end = time.time()

for i in range(len(predict)): #loop for showing results
    print(y_test[i], predict[i])

#Evaluation
accuracy = accuracy_score(y_test, predict) #Accuracy
cm = confusion_matrix(y_test, predict) #Confusion Matrix
report = classification_report(y_test, predict, target_names=["Iris-setosa", "Iris-versicolor", "Iris-virginica"]) #Classification report
training_time = train_time_end - train_time_start #finding training time
single_inference_time = one_inf_time_end - one_inf_time_start # inference time for single prediction
inference_time_batch = inf_time_end - inf_time_start # inference time for whole dataset
print(f"\nTraining Time: {training_time:.6f} seconds")
print(f"Inference Time (single sample): {single_inference_time:.8f} seconds")
print(f"Inference Time (batch): {inference_time_batch:.6f} seconds")
print("\nLR Accuracy :", accuracy*100)
print("\nLR Confusion Matrix : \n", cm)
print("\nLR Classification Report : \n", report)

model_save = input("Do you want to save this LR model ? (Y/N)")
match model_save:
    case ("Y" | "y"):
        with open("models/model_LR.pkl", "wb") as f: #saving the LR model as model_LR.pkl
            pickle.dump(model_LR, f)
        print("Logistic Regression model saved as 'model_LR.pkl'")
    case ("N" | "n"):
        print("Very well")
