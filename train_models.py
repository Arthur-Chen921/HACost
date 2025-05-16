# train_models.py
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans

def prepare_data():
    df = pd.read_csv("CostReporthha_Final_22.csv")
    visit_columns = [col for col in df.columns if "Visits" in col and "Total" not in col]
    X = df[visit_columns].fillna(0)
    y = df["Total Cost"].fillna(df["Total Cost"].mean())
    return X, y, visit_columns

def train_scaler(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'model/scaler.pkl')
    return X_scaled

def train_tree_model(X_train, y_train, visit_columns):
    tree_model = DecisionTreeRegressor(max_depth=3, random_state=42)
    tree_model.fit(X_train, y_train.numpy().ravel())
    joblib.dump(tree_model, 'model/tree_model.pkl')
    
    # 保存决策树可视化
    plt.figure(figsize=(20,10))
    plot_tree(tree_model, feature_names=visit_columns, filled=True)
    plt.savefig('static/tree_visualization.png')
    plt.close()

def train_clusters(X_scaled):
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    joblib.dump(kmeans, 'model/kmeans.pkl')

class CostPredictor(torch.nn.Module):
    # ...保持原有模型结构...

def train_nn(X_train, y_train):
    model = CostPredictor(X_train.shape[1])
    # ...原有训练代码...
    torch.save(model.state_dict(), 'model/cost_predictor.pth')

if __name__ == "__main__":
    X, y, visit_columns = prepare_data()
    X_scaled = train_scaler(X)
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2)
    
    train_tree_model(X_train, y_train, visit_columns)
    train_clusters(X_scaled)
    train_nn(X_train, y_train)
