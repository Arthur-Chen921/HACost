import torch
import joblib
import numpy as np

class Predictor:
    def __init__(self):
        self.scaler = joblib.load('model/scaler.pkl')
        self.tree_model = joblib.load('model/tree_model.pkl')
        self.kmeans = joblib.load('model/kmeans.pkl')
        
        # 加载神经网络
        self.nn_model = CostPredictor(len(self.scaler.feature_names_in_))
        self.nn_model.load_state_dict(torch.load('model/cost_predictor.pth'))
        self.nn_model.eval()
    
    def predict_cost(self, input_data):
        """ 统一预测接口 """
        # 转换为标准格式
        features = self._process_input(input_data)
        
        # 各模型预测
        return {
            'tree': float(self.tree_model.predict(features)),
            'nn': self._predict_nn(features),
            'cluster': int(self.kmeans.predict(features)[0])
        }
    
    def _process_input(self, raw_input):
        """ 处理网页输入到模型需要的格式 """
        # 示例处理逻辑（需根据实际特征顺序调整）
        processed = np.zeros(len(self.scaler.feature_names_in_))
        for idx, col in enumerate(self.scaler.feature_names_in_):
            processed[idx] = raw_input.get(col.split(',')[0], 0)
        
        return self.scaler.transform([processed])
    
    def _predict_nn(self, features):
        with torch.no_grad():
            return float(self.nn_model(torch.tensor(features, dtype=torch.float32)))
