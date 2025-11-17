#Decision Tree model with scaler object
import numpy as np
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time

# Load preprocessed data
x_train = np.load("dataset/x_train.npy")
x_test = np.load("dataset/x_test.npy")
y_train = np.load("dataset/y_train.npy")
y_test = np.load("dataset/y_test.npy")

#initialising model
model_DT = DecisionTreeClassifier()
# Training time taken for model to train
train_time_start = time.time()
model_DT.fit(x_train, y_train) #training model
train_time_end = time.time()

#inference time taken for model to predict whole set
inf_time_start = time.time()
predict = model_DT.predict(x_test) #testing x_test 
inf_time_end = time.time()

#inference time for single prediction
one_sample = x_test[0].reshape(1, -1)
one_inf_time_start = time.time()
model_DT.predict(one_sample)
one_inf_time_end = time.time()

for i in range(len(predict)): #loop for results
    print(y_test[i], predict[i])

accuracy = accuracy_score(y_test, predict) #Accuracy
cm = confusion_matrix(y_test, predict) #Confusion Matrix
report = classification_report(y_test, predict, target_names=["Iris-setosa", "Iris-versicolor", "Iris-virginica"]) #Classification report
training_time = train_time_end - train_time_start #finding training time
single_inference_time = one_inf_time_end - one_inf_time_start # inference time for single prediction
inference_time_batch = inf_time_end - inf_time_start # inference time for whole dataset
print(f"\nTraining Time: {training_time:.6f} seconds")
print(f"Inference Time (single sample): {single_inference_time:.8f} seconds")
print(f"Inference Time (batch): {inference_time_batch:.6f} seconds")
print("\nDT Accuracy :", accuracy*100)
print("\nDT Confusion Matrix : \n", cm)
print("\nDT Classification Report : \n", report)

model_save = input("Do you want to save this Decision Tree model ? (Y/N)")
match model_save:
    case ("Y" | "y"):
        with open("models/model_DT.pkl", "wb") as f: #saving the DT model as model_DT.pkl
            pickle.dump(model_DT, f)
        print ("Decision Tree Classifier model saved as 'model_DT.pkl'")
    case ("N" | "n"):
        print("Very well")

