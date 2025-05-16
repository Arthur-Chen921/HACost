from flask import Flask, render_template, request
from predict_service import Predictor

app = Flask(__name__)
predictor = Predictor()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 获取并处理输入
        user_input = {
            'Skilled Nursing Care-RN': request.form.get('nurse_visits', 0),
            'Physical Therapy': request.form.get('therapy_visits', 0),
            'Home Health Aide': request.form.get('aide_visits', 0)
        }
        
        # 获取预测结果
        results = predictor.predict_cost(user_input)
        
        # 获取聚类信息
        cluster_info = get_cluster_description(results['cluster'])
        
        return render_template('index.html', 
                             prediction=results,
                             cluster_info=cluster_info)
    
    return render_template('index.html')

def get_cluster_description(cluster_id):
    """ 返回聚类特征描述 """
    clusters = {
        0: "高护理需求型机构：注册护士服务量较大，适合高端定价",
        1: "综合平衡型机构：各项服务均衡，建议标准定价",
        2: "社区服务型机构：家庭护理为主，建议优惠定价"
    }
    return clusters.get(cluster_id, "未知类型")

if __name__ == '__main__':
    app.run(debug=True)
