# Teen Mental Health Prediction (PyTorch – From Scratch)

## 📌 Overview
This project builds a binary classification model to predict mental health conditions (depression) using a structured dataset. The entire pipeline is implemented from scratch, including preprocessing, model design, training, and evaluation.

---

## 📊 Dataset
- Samples: 1200  
- Features: 14–15 (after preprocessing)  
- Target: `depression_label` (0 = No, 1 = Yes)  
- Type: Binary Classification  

---

## ⚙️ Preprocessing Steps
- Handled missing values  
- Encoded categorical variables (one-hot encoding)  
- Applied feature scaling (standardization)  
- Separated features (X) and target (y)  
- Converted data: Pandas → NumPy → PyTorch tensors  
- Manual train-test split (80% training, 20% testing)  

---

## 🧩 Model Architecture

Input Layer → Hidden Layer (8 neurons, ReLU) → Output Layer (1 neuron, Sigmoid)

### Key Choices:
- Hidden neurons: 8 (simple and efficient)  
- Activation (hidden): ReLU  
- Activation (output): Sigmoid (for probability output)  

---

## 🔢 Trainable Parameters
Example (14 features):
- Layer 1: (14 × 8) + 8 = 120  
- Layer 2: (8 × 1) + 1 = 9  
- Total = 129 parameters  

---

## 📉 Loss & Optimization
- Loss Function: Binary Cross Entropy (BCELoss)  
- Optimizer: Adam  

---

## 🔁 Training
- Trained for 100 epochs  
- Steps implemented:
  - Forward pass  
  - Loss computation  
  - Backward pass  
  - Gradient reset  
  - Parameter update  
- Stored loss history  
- Printed loss every 10 epochs  

---

## 📈 Evaluation

Accuracy:
- Main Model: 0.9750  
- Modified Model (lr = 0.001): 0.7542  

---

## 🔍 Analysis

Learning Rate Impact:
- Higher learning rate (0.01):
  - Faster convergence  
  - Better performance  

- Lower learning rate (0.001):
  - Slower training  
  - Lower accuracy  

Conclusion:
The original learning rate performed better for this dataset.

---

## ⚠️ Model Behavior
- Training and testing performance are reasonable  
- Model shows good generalization  

---

## 🧠 Key Learnings
- Importance of feature scaling  
- Manual ML pipeline implementation  
- Effect of learning rate on performance  
- Proper tensor shaping  
- Gradient handling in PyTorch  

---

## 🚀 Technologies Used
- Python  
- NumPy  
- Pandas  
- PyTorch  

---

## 📌 Notes
- No use of Dataset or DataLoader  
- No use of sklearn train_test_split  
- All steps implemented manually as required  

---
