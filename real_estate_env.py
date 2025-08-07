import gym
from gym import spaces
from typing import List, Dict, Any
import copy
import numpy as np
import random

class RentalYieldScorer:
    """租金收益率评分器"""
    def score(self, properties: List[Dict[str, Any]], state: Dict[str, Any]) -> List[float]:
        scores = []
        agent_income = state.get("income")[0]
        affordability_ratio = 10  # 设定合理房价上限为年收入的 10 倍
        
        for p in properties:
            price = p.get("price", 1.0)
            base_rent = p.get("base_rent", 0.0)
            service_charge = p.get("service_charge", 0.0)
            
            # 计算每月总租金收入
            total_rent = base_rent + service_charge
            rental_yield = (total_rent * 12) / price if price > 0 else 0.0
            
            # 可负担性惩罚：超过上限的部分将降低得分
            max_affordable_price = agent_income * affordability_ratio
            if price > max_affordable_price:
                penalty = (price / max_affordable_price) ** 0.5  # 惩罚因子 sqrt 放缓梯度
                rental_yield /= penalty
            
            scores.append(rental_yield)
        
        return scores

class RealEstateEnv(gym.Env):
    """房地产投资环境"""
    def __init__(
        self,
        all_properties: List[Dict[str, Any]],
        agent_profile: Dict[str, Any],
        growth_index: Dict[str, Dict[int, float]],
        max_inventory=10,
        candidate_top_k=20,
        scorer=None,
        reward_scale=10.0,
        cash_bonus_scale=5.0
    ):
        super().__init__()
        self.all_properties = all_properties
        self.agent_profile = agent_profile
        self.growth_index = growth_index
        self.max_inventory = max_inventory
        self.candidate_top_k = candidate_top_k
        self.max_year = 2035 - agent_profile["start_year"]
        
        self.start_year = agent_profile["start_year"]
        self.end_year = 2035
        
        self.scorer = scorer or RentalYieldScorer()
        self.reward_scale = reward_scale
        self.cash_bonus_scale = cash_bonus_scale
        
        self.action_space = spaces.Dict({
            "buy_index": spaces.Discrete(self.candidate_top_k + 1),
            "sell_house_ids": spaces.Box(0, 1e6, (max_inventory,), dtype=np.int32)
        })
        
        self.observation_space = spaces.Dict({
            "cash": spaces.Box(0, 1e10, (1,), dtype=np.float32),
            "initial_cash": spaces.Box(0, 1e10, (1,), dtype=np.float32),
            "income": spaces.Box(0, 1e6, (1,), dtype=np.float32),
            "age": spaces.Box(0, 100, (1,), dtype=np.int32),
            "marriage": spaces.Box(0, 1, (1,), dtype=np.int32),
            "children": spaces.Box(0, 10, (1,), dtype=np.int32),
            "start_year": spaces.Box(2000, 2100, (1,), dtype=np.int32),
            "current_year": spaces.Box(2000, 2100, (1,), dtype=np.int32),
            "year_remaining": spaces.Box(0, self.max_year, (1,), dtype=np.int32),
            "buy_candidates": spaces.Box(0, 1e6, (candidate_top_k, 10), dtype=np.float32),
            "portfolio": spaces.Box(0, 1e6, (max_inventory, 10), dtype=np.float32),
        })
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        self.year = 0
        self.current_year = self.start_year
        self.remaining_years = self.end_year - self.current_year
        
        self.agent_age = self.agent_profile["age"]
        self.marriage = self.agent_profile["marriage"]
        self.children = self.agent_profile["children"]
        self.income = self.agent_profile["income"]
        self.initial_cash = self.agent_profile["initial_cash"]
        self.cash = self.initial_cash
        self.portfolio = []
        self.current_market = []
        
        self._score_and_select_candidates()
        
        obs = self._get_obs()
        return obs, {}
    
    def step(self, action):
        TA_t = self._compute_total_assets()
        
        sell_ids = set(int(x) for x in action["sell_house_ids"] if x > 0)
        new_portfolio = []
        for p in self.portfolio:
            if p["house_id"] in sell_ids:
                self._sell_property(p)
            else:
                new_portfolio.append(p)
        self.portfolio = new_portfolio
        
        buy_index = int(action["buy_index"])
        max_idx = len(self.current_market) - 1
        buy_index = min(max(buy_index, -1), max_idx)
        
        if buy_index >= 0 and len(self.portfolio) < self.max_inventory:
            p = self.current_market[buy_index]
            if p["house_id"] not in set(x["house_id"] for x in self.portfolio):
                tax_transfer = p.get("tax_transfer", 0.05)
                cost = p["price"] * (1 + tax_transfer)
                if cost <= self.cash:  # 确保有足够现金
                    self.cash -= cost
                    self.portfolio.append({
                        "house_id": p["house_id"],
                        "region_type": p["region_type"],
                        "house_type": p["house_type"],
                        "plz": p["plz"],
                        "purchase_price": p["price"],
                        "market_price": p["price"],
                        "base_rent": p["base_rent"],
                        "service_charge": p["service_charge"],
                        "purchase_date": self.current_year,
                        "aggregate_rent": 0.0
                    })
                else:
                    buy_index = -1
            else:
                buy_index = -1
        
        self.year += 1
        self.current_year = self.start_year + self.year
        self.remaining_years = self.end_year - self.current_year
        self.agent_age += 1
        self.cash += self.income
        
        holding_tax_rate = 0.0164
        holding_tax_total = sum(p["market_price"] * holding_tax_rate for p in self.portfolio)
        self.cash -= holding_tax_total
        
        for p in self.portfolio:
            monthly_rent = p["base_rent"] + p["service_charge"]
            self.cash += monthly_rent * 12
            p["aggregate_rent"] += monthly_rent * 12
        
        self._update_market()
        self._score_and_select_candidates()
        
        TA_tp1 = self._compute_total_assets()
        cash_ratio_t = self.cash / (TA_t + 1e-6)
        cash_ratio_tp1 = self.cash / (TA_tp1 + 1e-6)
        
        reward_main = (TA_tp1 - TA_t) / (TA_t + 1e-6) * self.reward_scale
        reward_cash = (cash_ratio_tp1 - cash_ratio_t) * self.cash_bonus_scale
        reward = reward_main + reward_cash
        done = self.year >= self.max_year or self.agent_age >= 100
        return self._get_obs(), reward, done, {}
    
    def _sell_property(self, p):
        self.cash += p["market_price"]
        return True
    
    def _update_market(self):
        year = self.current_year
        
        for p in self.all_properties:
            # 处理house_type字段，支持字符串和数值
            house_type_val = p["house_type"]
            if isinstance(house_type_val, str):
                if house_type_val.lower() in ['apartment', 'wohnung']:
                    house_type_str = "Apartment"
                else:
                    house_type_str = "Detached"
            else:
                house_type_str = "Apartment" if house_type_val == 0 else "Detached"
            
            region_type = p["region_type"]
            key = f"{house_type_str}_Type-{region_type}"
            
            price_index = self.growth_index.get(key, {}).get(year, 1.0)
            rent_index = self.growth_index.get("ColdRent_Prophet", {}).get(year, 1.0)
            service_index = self.growth_index.get("ServiceCharge_Prophet", {}).get(year, 1.0)
            
            if random.random() < 0.05:
                shock = random.uniform(0.95, 1.10)
                price_index *= shock
            
            p["price"] *= price_index
            p["base_rent"] *= rent_index
            p["service_charge"] *= service_index
        
        for p in self.portfolio:
            house_type_str = "Apartment" if p["house_type"] == 0 else "Detached"
            region_type = p["region_type"]
            key = f"{house_type_str}_Type-{region_type}"
            
            price_index = self.growth_index.get(key, {}).get(year, 1.0)
            rent_index = self.growth_index.get("ColdRent_Prophet", {}).get(year, 1.0)
            service_index = self.growth_index.get("ServiceCharge_Prophet", {}).get(year, 1.0)
            
            if random.random() < 0.05:
                shock = random.uniform(0.95, 1.10)
                price_index *= shock
            
            p["market_price"] *= price_index
            p["base_rent"] *= rent_index
            p["service_charge"] *= service_index
    
    def _score_and_select_candidates(self):
        current_house_ids = set(p["house_id"] for p in self.portfolio)
        candidate_props = [
            p for p in self.all_properties
            if p["house_id"] not in current_house_ids
            and p["price"] * (1 + p.get("tax_transfer", 0.05)) <= self.cash
        ]
        
        if candidate_props:
            scores = self.scorer.score(candidate_props, self._get_obs())
            sorted_props = [p for _, p in sorted(zip(scores, candidate_props), key=lambda x: -x[0])]
            self.current_market = sorted_props[:self.candidate_top_k]
        else:
            self.current_market = []
    
    def _compute_total_assets(self):
        return self.cash + sum(p["market_price"] for p in self.portfolio)
    
    def _get_obs(self):
        return {
            "cash": np.array([self.cash], dtype=np.float32),
            "initial_cash": np.array([self.initial_cash], dtype=np.float32),
            "income": np.array([self.income], dtype=np.float32),
            "age": np.array([self.agent_age], dtype=np.int32),
            "marriage": np.array([self.marriage], dtype=np.int32),
            "children": np.array([self.children], dtype=np.int32),
            "start_year": np.array([self.start_year], dtype=np.int32),
            "current_year": np.array([self.current_year], dtype=np.int32),
            "year_remaining": np.array([self.remaining_years], dtype=np.int32),
            "buy_candidates": np.array(self._pad_props(self.current_market, self.candidate_top_k, mode="market"), dtype=np.float32),
            "portfolio": np.array(self._pad_props(self.portfolio, self.max_inventory, mode="portfolio"), dtype=np.float32),
        }
    
    def _pad_props(self, props, length, mode="market"):
        padded = []
        for p in props[:length]:
            # 转换house_type字符串为数值
            house_type_val = p.get("house_type", 0)
            if isinstance(house_type_val, str):
                if house_type_val.lower() in ['apartment', 'wohnung']:
                    house_type_val = 0
                elif house_type_val.lower() in ['detachedhouse', 'detached', 'house', 'haus']:
                    house_type_val = 1
                else:
                    house_type_val = 0  # 默认为公寓
            
            if mode == "market":
                padded.append([
                    p.get("house_id", 0),
                    p.get("price", 0),
                    p.get("base_rent", 0) + p.get("service_charge", 0),
                    p.get("plz", 0),
                    p.get("region_type", 0),
                    house_type_val,
                    0, 0, 0, 0
                ])
            elif mode == "portfolio":
                padded.append([
                    p.get("house_id", 0),
                    p.get("purchase_price", 0),
                    p.get("market_price", 0),
                    p.get("base_rent", 0) + p.get("service_charge", 0),
                    p.get("plz", 0),
                    p.get("purchase_date", 0),
                    p.get("aggregate_rent", 0),
                    p.get("region_type", 0),
                    house_type_val,
                    0
                ])
        while len(padded) < length:
            padded.append([0.0] * 10)
        return padded