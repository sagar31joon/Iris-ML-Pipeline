# Iris Classification System

A clean, modular, end-to-end Machine Learning project that classifies Iris flower species using three different ML models:

- Logistic Regression  (LR)
- Decision Tree Classifier  
- Support Vector Classifier (SVC)

This project includes complete preprocessing, model training, model saving/loading, and a CLI-based prediction system through `main.py`.

---

## 📁 Project Structure

```
iris-classification-system/
│
├── dataset/
│   ├── iris_raw.csv
│   └── iris_processed.csv
│
├── models/
│   ├── scaler.pkl
│   ├── model_LR.pkl
│   ├── model_DT.pkl
│   └── model_SVC.pkl
│
├── src/
│   ├── prepare_data.py
│   ├── train_LR.py
│   ├── train_decision_tree.py
│   ├── train_SVM.py
│   ├── predict_LR.py
│   ├── predict_decision_tree.py
│   └── predict_SVM.py
│
├── samples/
│   └── Visualisation_iris_raw.py
│
├── main.py
├── README.md
└── requirements.txt
```

---

## 🚀 Features

### ✔ End-to-End ML Pipeline
- Load raw dataset  
- Encode labels  
- Scale features  
- Save processed dataset  

### ✔ Single Scaler for All Models
- Fitted on training data only  
- Saved as `scaler.pkl`  
- Loaded by all training/prediction scripts  

### ✔ Three Independently Trained Models
Each model is:
- Trained  
- Evaluated  
- Saved as a `.pkl` file  

### ✔ Prediction Scripts
Each model has its own prediction file:
- Accepts 4 user inputs  
- Scales input using saved scaler  
- Loads model and predicts species  

### ✔ Main CLI Program
`main.py` lets the user choose:
- Logistic Regression  
- Decision Tree  
- SVC  
- Exit  

Runs the corresponding prediction script.

---

## 📊 Dataset Visualization (Optional)

The `samples/Visualisation_iris_raw.py` script is used for Exploratory Data Analysis (EDA).  
It generates a Seaborn pairplot to visualize relationships between features and species.

### ▶ Run the visualization script
```
python3 samples/Visualisation_iris_raw.py
```

This helps in understanding how separable the Iris species are and why certain models perform well on this dataset.

---

## 🧠 Models Used

### 1. Logistic Regression
Simple yet effective linear classifier.

### 2. Decision Tree Classifier
Non-linear model with hierarchical decision rules.

### 3. Support Vector Classifier (SVC)
Margin-based classifier suitable for multi-class problems.

---

## 🔧 Technologies Used
- Python 3.12  
- NumPy  
- Pandas  
- Scikit-learn  
- Pickle  
- Matplotlib
- Seaborn
---

## 🏃‍♂️ How to Run

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Prepare the dataset
```
python3 src/prepare_data.py
> ⚠️ If you want to retrain using your own dataset, adjust the file path inside `prepare_data.py`.  
> Renaming the processed dataset file is optional.
```

### 3. Train all models
```
python3 src/train_LR.py
python3 src/train_decision_tree.py
python3 src/train_SVM.py
```

### 4. Run the main program
```
python3 main.py
```

---

## 🔍 Example Output

```
🌸 IRIS FLOWER CLASSIFICATION SYSTEM 🌸

Choose a model:
1. Logistic Regression
2. Decision Tree Classifier
3. Support Vector Classifier
4. Exit
```

---

## 📌 Notes
- The scaler is saved once and reused across all models for consistency.  
- All training scripts overwrite previous model files when run again.  
- The prediction scripts work independently and can be integrated into any front-end or API later.

---

## 📝 License
This project is for educational and practice purposes.  
Feel free to fork and modify it.

