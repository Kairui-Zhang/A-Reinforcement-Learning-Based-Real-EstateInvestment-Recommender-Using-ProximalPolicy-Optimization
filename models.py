from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import json

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """用户模型"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # 用户个人信息
    age = db.Column(db.Integer)
    income = db.Column(db.Float)
    initial_cash = db.Column(db.Float)
    marital_status = db.Column(db.String(20))
    children_count = db.Column(db.Integer)
    
    # 关联投资记录
    investment_records = db.relationship('InvestmentRecord', backref='user', lazy=True)
    
    def set_password(self, password):
        """设置密码哈希"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """验证密码"""
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'age': self.age,
            'income': self.income,
            'initial_cash': self.initial_cash,
            'marital_status': self.marital_status,
            'children_count': self.children_count,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class InvestmentRecord(db.Model):
    """投资记录模型"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # 投资参数
    start_year = db.Column(db.Integer, nullable=False)
    agent_age = db.Column(db.Integer, nullable=False)
    agent_income = db.Column(db.Float, nullable=False)
    agent_cash = db.Column(db.Float, nullable=False)
    marital_status = db.Column(db.String(20), nullable=False)
    children_count = db.Column(db.Integer, nullable=False)
    
    # 投资结果
    final_assets = db.Column(db.Float, nullable=False)
    roi = db.Column(db.Float, nullable=False)
    portfolio_count = db.Column(db.Integer, nullable=False)
    
    # 详细投资组合（JSON格式存储）
    portfolio_details = db.Column(db.Text)  # JSON字符串
    investment_history = db.Column(db.Text)  # JSON字符串
    
    def set_portfolio_details(self, portfolio):
        """设置投资组合详情"""
        self.portfolio_details = json.dumps(portfolio, ensure_ascii=False)
    
    def get_portfolio_details(self):
        """获取投资组合详情"""
        if self.portfolio_details:
            return json.loads(self.portfolio_details)
        return []
    
    def set_investment_history(self, history):
        """设置投资历史"""
        self.investment_history = json.dumps(history, ensure_ascii=False)
    
    def get_investment_history(self):
        """获取投资历史"""
        if self.investment_history:
            return json.loads(self.investment_history)
        return []
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'start_year': self.start_year,
            'agent_age': self.agent_age,
            'agent_income': self.agent_income,
            'agent_cash': self.agent_cash,
            'marital_status': self.marital_status,
            'children_count': self.children_count,
            'final_assets': self.final_assets,
            'roi': self.roi,
            'portfolio_count': self.portfolio_count,
            'portfolio_details': self.get_portfolio_details(),
            'investment_history': self.get_investment_history()
        }