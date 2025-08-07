from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import json
from datetime import datetime
import torch
import pandas as pd
import numpy as np
import os

# 导入数据库模型
from models import db, User, InvestmentRecord
# 导入PPO模型和环境
from ppo_model import ActorCriticNet, preprocess_obs, sample_action
from real_estate_env import RealEstateEnv, RentalYieldScorer
from utils import load_growth_index

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///real_estate_ai.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 初始化扩展
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

@login_manager.unauthorized_handler
def unauthorized():
    # 如果是AJAX请求，返回JSON错误
    if request.is_json or request.headers.get('Content-Type') == 'application/json':
        return jsonify({'error': 'Authentication required', 'redirect': url_for('login')}), 401
    # 否则重定向到登录页面
    return redirect(url_for('login', next=request.url))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# 全局变量存储模型和数据
ppo_model = None
all_properties = None
growth_index = None

def load_ppo_model_and_data():
    """加载PPO模型和相关数据"""
    global ppo_model, all_properties, growth_index
    
    try:
        # 加载房产数据
        properties_file = 'all_properties.xlsx'
        if os.path.exists(properties_file):
            print("开始加载房产数据...")
            try:
                # 尝试加载更少的数据以避免内存问题
                df = pd.read_excel(properties_file, nrows=1000)  # 只读取前1000行
                all_properties = df.to_dict('records')
                print(f"房产数据加载成功，共{len(all_properties)}条记录")
            except Exception as e:
                print(f"加载房产数据失败: {e}")
                print(f"错误类型: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                all_properties = []
        else:
            print(f"警告: 未找到房产数据文件 {properties_file}")
            all_properties = []
        
        # 加载增长指数
        index_file = 'price_rent_index.xlsx'
        if os.path.exists(index_file):
            print("开始加载增长指数...")
            try:
                growth_index = load_growth_index(index_file)
                print("加载了价格租金增长指数")
            except Exception as e:
                print(f"加载增长指数失败: {e}")
                growth_index = {}
        else:
            print(f"警告: 未找到增长指数文件 {index_file}")
            growth_index = {}
        
        # 加载PPO模型
        model_file = 'ppo_policy_right.pth'
        if os.path.exists(model_file):
            print("开始加载PPO模型...")
            try:
                # 模型参数
                state_dim = 9  # cash, initial_cash, income, age, marriage, children, start_year, current_year, year_remaining
                buy_candidates_dim = 10  # 每个候选房产的特征维度
                portfolio_dim = 10  # 每个持仓房产的特征维度
                
                ppo_model = ActorCriticNet(state_dim, buy_candidates_dim, portfolio_dim)
                ppo_model.load_state_dict(torch.load(model_file, map_location='cpu'))
                ppo_model.eval()
                print("PPO模型加载成功")
            except Exception as e:
                print(f"加载PPO模型失败: {e}")
                ppo_model = None
        else:
            print(f"警告: 未找到PPO模型文件 {model_file}")
            ppo_model = None
            
    except Exception as e:
        print(f"加载模型和数据时出错: {e}")
        ppo_model = None
        all_properties = []
        growth_index = {}

@app.route('/')
def welcome():
    """首页 - 所有用户都可以访问"""
    return render_template('index.html')

@app.route('/index')
def index():
    """重定向到首页"""
    return render_template('index.html')

@app.route('/invest')
@login_required
def invest():
    return render_template('index.html')

@app.route('/analysis')
@login_required
def analysis():
    # 添加调试信息
    print(f"Analysis route accessed by user: {current_user.username if current_user.is_authenticated else 'Anonymous'}")
    return render_template('analysis.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        # 验证输入
        if not username or not email or not password:
            return jsonify({'success': False, 'message': 'Please fill in all required fields'})
        
        # 检查用户是否已存在
        if User.query.filter_by(username=username).first():
            return jsonify({'success': False, 'message': 'Username already exists'})
        
        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'message': 'Email already registered'})
        
        # 创建新用户
        user = User(username=username, email=email)
        user.set_password(password)
        
        try:
            db.session.add(user)
            db.session.commit()
            return jsonify({'success': True, 'message': 'Registration successful! Please log in.'})
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'message': 'Registration failed, please try again'})
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'success': False, 'message': 'Please enter username and password'})
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            return jsonify({'success': True, 'message': 'Login successful!', 'redirect': url_for('dashboard')})
        else:
            return jsonify({'success': False, 'message': 'Invalid username or password'})
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('welcome'))

@app.route('/dashboard')
@login_required
def dashboard():
    # 获取用户的投资记录
    records = InvestmentRecord.query.filter_by(user_id=current_user.id).order_by(InvestmentRecord.created_at.desc()).all()
    return render_template('dashboard.html', user=current_user, records=records)

@app.route('/simulate', methods=['POST'])
@login_required
def simulate():
    try:
        data = request.json
        
        # 构建agent profile
        agent_profile = {
            "start_year": int(data.get('start_year', 2024)),
            "age": int(data.get('age', 30)),
            "marriage": int(data.get('marriage', 0)),
            "children": int(data.get('children', 0)),
            "income": int(data.get('income', 80000)),
            "initial_cash": int(data.get('initial_cash', 200000))
        }
        
        # 检查是否有PPO模型和数据
        if ppo_model is None or not all_properties or not growth_index:
            print("使用随机模拟（模型或数据未加载）")
            return simulate_random(agent_profile)
        
        # 使用PPO模型进行真实投资分析
        result = run_ppo_simulation(agent_profile)
        
        # 保存投资记录到数据库
        try:
            investment_record = InvestmentRecord(
                user_id=current_user.id,
                start_year=agent_profile['start_year'],
                agent_age=agent_profile['age'],
                agent_income=agent_profile['income'],
                agent_cash=agent_profile['initial_cash'],
                marital_status='married' if agent_profile['marriage'] == 1 else 'single',
                children_count=agent_profile['children'],
                final_assets=result['final_total_assets'],
                roi=result['total_return'],
                portfolio_count=len(result['final_portfolio'])
            )
            investment_record.set_portfolio_details(result['final_portfolio'])
            investment_record.set_investment_history(result.get('investment_history', {}))
            
            db.session.add(investment_record)
            db.session.commit()
        except Exception as db_error:
            print(f"保存投资记录时出错: {db_error}")
            db.session.rollback()
        
        return jsonify(result)
        
    except Exception as e:
        print(f"模拟过程中出错: {e}")
        return jsonify({"error": f"请求处理错误: {str(e)}"})

def simulate_random(agent_profile):
    """随机模拟（备用方案）"""
    import random
    final_cash = agent_profile['initial_cash'] * (0.8 + random.random() * 0.4)
    final_portfolio_value = agent_profile['initial_cash'] * (0.5 + random.random() * 1.0)
    final_total_assets = final_cash + final_portfolio_value
    total_return = ((final_total_assets - agent_profile['initial_cash']) / agent_profile['initial_cash']) * 100
    
    return {
        'success': True,
        'final_cash': final_cash,
        'final_portfolio_value': final_portfolio_value,
        'final_total_assets': final_total_assets,
        'initial_cash': agent_profile['initial_cash'],
        'total_return': total_return,
        'final_portfolio': []
    }

def run_ppo_simulation(agent_profile):
    """使用PPO模型运行投资分析"""
    try:
        # 创建环境
        env = RealEstateEnv(
            all_properties=all_properties.copy(),
            agent_profile=agent_profile,
            growth_index=growth_index,
            max_inventory=10,
            candidate_top_k=20,
            scorer=RentalYieldScorer()
        )
        
        # 重置环境
        obs, _ = env.reset()
        done = False
        step_count = 0
        max_steps = 15  # 最多15年
        
        investment_history = []
        
        with torch.no_grad():
            while not done and step_count < max_steps:
                # 预处理观察值
                state_flat, buy_candidates, portfolio = preprocess_obs(obs)
                
                # 获取模型输出
                buy_scores, sell_scores, value = ppo_model(state_flat, buy_candidates, portfolio)
                
                # 生成有效的买入和卖出索引
                buy_indices = list(range(len(obs["buy_candidates"])))
                sell_ids = [int(p[0]) for p in obs["portfolio"] if len(p) > 0]  # house_id
                
                # 采样动作
                action, log_prob = sample_action(
                    buy_scores, sell_scores, buy_candidates, 
                    buy_indices, sell_ids, temperature=0.5, action_top_k=5
                )
                
                # 记录投资决策
                year_info = {
                    'year': env.current_year,
                    'cash': float(env.cash),
                    'portfolio_count': len(env.portfolio),
                    'total_assets': float(env._compute_total_assets()),
                    'action': action
                }
                investment_history.append(year_info)
                
                # 执行动作
                obs, reward, done, _ = env.step(action)
                step_count += 1
        
        # 计算最终结果
        final_cash = float(env.cash)
        final_portfolio = []
        final_portfolio_value = 0.0
        
        for p in env.portfolio:
            property_info = {
                'house_id': int(p['house_id']),
                'purchase_price': float(p['purchase_price']),
                'market_price': float(p['market_price']),
                'monthly_rent': float(p['base_rent'] + p['service_charge']),
                'plz': p.get('plz', 'N/A'),
                'aggregate_rent': float(p.get('aggregate_rent', 0))
            }
            final_portfolio.append(property_info)
            final_portfolio_value += property_info['market_price']
        
        final_total_assets = final_cash + final_portfolio_value
        total_return = ((final_total_assets - agent_profile['initial_cash']) / agent_profile['initial_cash']) * 100
        
        return {
            'success': True,
            'final_cash': final_cash,
            'final_portfolio_value': final_portfolio_value,
            'final_total_assets': final_total_assets,
            'initial_cash': agent_profile['initial_cash'],
            'total_return': total_return,
            'final_portfolio': final_portfolio,
            'investment_history': investment_history
        }
        
    except Exception as e:
        print(f"PPO模拟过程中出错: {e}")
        # 如果PPO模拟失败，回退到随机模拟
        return simulate_random(agent_profile)

@app.route('/api/dashboard_stats')
@login_required
def get_dashboard_stats():
    """获取仪表板统计数据"""
    try:
        # 获取用户的所有投资记录
        records = InvestmentRecord.query.filter_by(user_id=current_user.id).all()
        
        if not records:
            return jsonify({
                'success': True,
                'stats': {
                    'total_simulations': 0,
                    'average_roi': 0,
                    'best_roi': 0,
                    'total_properties': 0
                }
            })
        
        # 计算统计数据
        total_simulations = len(records)
        roi_values = [record.roi for record in records if record.roi is not None]
        average_roi = sum(roi_values) / len(roi_values) if roi_values else 0
        best_roi = max(roi_values) if roi_values else 0
        total_properties = sum(record.portfolio_count for record in records if record.portfolio_count is not None)
        
        return jsonify({
            'success': True,
            'stats': {
                'total_simulations': total_simulations,
                'average_roi': round(average_roi, 2),
                'best_roi': round(best_roi, 2),
                'total_properties': total_properties
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error fetching stats: {str(e)}'})

@app.route('/api/recent_records')
@login_required
def get_recent_records():
    """获取最近的投资记录"""
    try:
        records = InvestmentRecord.query.filter_by(user_id=current_user.id)\
                                       .order_by(InvestmentRecord.created_at.desc())\
                                       .limit(5).all()
        
        records_data = []
        for record in records:
            records_data.append({
                'id': record.id,
                'start_year': record.start_year,
                'final_assets': record.final_assets,
                'roi': record.roi,
                'portfolio_count': record.portfolio_count,
                'created_at': record.created_at.strftime('%Y-%m-%d %H:%M')
            })
        
        return jsonify({
            'success': True,
            'records': records_data
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error fetching records: {str(e)}'})

def create_tables():
    """创建数据库表"""
    with app.app_context():
        db.create_all()
        print("数据库表创建完成")



@app.route('/test')
def test():
    """测试路由（无需登录）"""
    return "Test route works!"

@app.route('/test_login')
@login_required
def test_login():
    """测试需要登录的路由"""
    return "Login required route works!"

@app.route('/debug_routes')
def debug_routes():
    """调试路由 - 显示所有已注册的路由"""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'rule': rule.rule
        })
    return jsonify(routes)

@app.route('/debug_auth')
def debug_auth():
    """调试认证状态"""
    return jsonify({
        'is_authenticated': current_user.is_authenticated,
        'user_id': current_user.id if current_user.is_authenticated else None,
        'username': current_user.username if current_user.is_authenticated else None
    })

@app.route('/test_analysis')
def test_analysis():
    """测试analysis路由（无需登录）"""
    return "Analysis route is working! This is a test without login requirement."

if __name__ == '__main__':
    create_tables()
    load_ppo_model_and_data()  # 加载PPO模型和数据
    print("启动Web服务器...")
    app.run(debug=True, host='127.0.0.1', port=5001)