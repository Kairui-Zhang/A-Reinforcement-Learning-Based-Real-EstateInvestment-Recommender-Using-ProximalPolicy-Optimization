import random
import pandas as pd
from typing import Dict, Any

def generate_random_agent_profile(start_year_range=(2015, 2025),
                                  age_range=(22, 78),
                                  income_range=(50000, 150000),
                                  cash_multiplier_range=(1.0, 3.0),
                                  max_children=3,
                                  random_seed: int = None) -> Dict[str, Any]:
    """生成随机的智能体配置文件"""
    rng = random.Random(random_seed) if random_seed is not None else random
    
    start_year = rng.randint(*start_year_range)
    age = rng.randint(*age_range)
    marriage = rng.randint(0, 1)
    
    # 根据年龄分布生成子女数量
    if age <= 25:
        children = rng.choices([0, 1, 2], weights=[0.6, 0.3, 0.1])[0]
    elif age <= 35:
        children = rng.choices([0, 1, 2, 3], weights=[0.3, 0.4, 0.25, 0.05])[0]
    elif age <= 50:
        children = rng.choices([0, 1, 2, 3], weights=[0.1, 0.25, 0.45, 0.2])[0]
    else:
        children = rng.choices([1, 2, 3], weights=[0.2, 0.5, 0.3])[0]
    
    min_income, max_income = income_range
    income_range_size = max_income - min_income
    
    # 根据年龄调整收入分布
    if age <= 30:
        income_position = rng.uniform(0.0, 0.4)
    elif age <= 45:
        income_position = rng.uniform(0.6, 1.0)
    elif age <= 60:
        income_position = rng.uniform(0.5, 0.9)
    else:
        income_position = rng.uniform(0.2, 0.7)
    
    income = int(min_income + income_range_size * income_position)
    cash_multiplier = rng.uniform(*cash_multiplier_range)
    initial_cash = int(income * cash_multiplier)
    
    return {
        "start_year": start_year,
        "age": age,
        "marriage": marriage,
        "children": children,
        "income": income,
        "initial_cash": initial_cash
    }

def load_growth_index(file_path: str) -> Dict[str, Any]:
    """加载价格租金增长指数"""
    try:
        df = pd.read_excel(file_path)
        # 将DataFrame转换为字典格式
        growth_index = {}
        for _, row in df.iterrows():
            year = int(row.get('year', 0))
            growth_index[year] = {
                'price_growth': float(row.get('price_growth', 1.0)),
                'rent_growth': float(row.get('rent_growth', 1.0))
            }
        return growth_index
    except Exception as e:
        print(f"加载增长指数时出错: {e}")
        return {}