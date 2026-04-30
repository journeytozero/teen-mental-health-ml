import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Load dataset
df = pd.read_csv("data/Teen_Mental_Health_Dataset.csv")

# Handle missing values
df = df.dropna()

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Separate features and target
X = df.drop("depression_label", axis=1)
y = df["depression_label"]

# Feature scaling
X = (X - X.mean()) / X.std()

# Train-test split manually
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))

X_train = X.iloc[indices[:split]]
X_test = X.iloc[indices[split:]]

y_train = y.iloc[indices[:split]]
y_test = y.iloc[indices[split:]]

# Convert to tensors
X_train_tensor = torch.from_numpy(X_train.values).float()
X_test_tensor = torch.from_numpy(X_test.values).float()

y_train_tensor = torch.from_numpy(y_train.values).float().view(-1, 1)
y_test_tensor = torch.from_numpy(y_test.values).float().view(-1, 1)


# Model
class Model(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.linear1 = nn.Linear(num_features, 8)
        self.relu = nn.ReLU()

        self.linear2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        op = self.linear1(features)
        op = self.relu(op)

        op = self.linear2(op)
        op = self.sigmoid(op)

        return op


num_features = X_train_tensor.shape[1]
model = Model(num_features)

# Print trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total trainable parameters:", total_params)

# Loss and optimizer
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
loss_history = []
epochs = 100

for epoch in range(epochs):
    y_pred = model(X_train_tensor)

    loss = loss_function(y_pred, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


# Evaluation
model.eval()

with torch.no_grad():
    train_pred = model(X_train_tensor)
    train_pred_labels = (train_pred >= 0.5).float()
    train_accuracy = (train_pred_labels == y_train_tensor).float().mean()

    test_pred = model(X_test_tensor)
    test_pred_labels = (test_pred >= 0.5).float()
    test_accuracy = (test_pred_labels == y_test_tensor).float().mean()

print("Train Accuracy:", train_accuracy.item())
print("Test Accuracy:", test_accuracy.item())
