import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
from typing import List, Dict, Any

class ActorCriticNet(nn.Module):
    """Actor-Critic网络"""
    def __init__(self, state_dim, buy_candidates_dim, portfolio_dim, hidden_dim=256):
        super().__init__()
        
        # 全局状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 买入候选编码器
        self.buy_encoder = nn.Sequential(
            nn.Linear(buy_candidates_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 持仓房产编码器
        self.portfolio_encoder = nn.Sequential(
            nn.Linear(portfolio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Actor Heads
        self.buy_head = nn.Linear(hidden_dim, 1)     # 对每个候选房产打分
        self.sell_head = nn.Linear(hidden_dim, 1)    # 对每个持仓房产打分
        
        # Critic Head
        self.value_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state_flat, buy_candidates, portfolio):
        # state 特征
        state_feat = self.state_encoder(state_flat)  # [B, H]
        
        # buy 特征
        buy_feat = self.buy_encoder(buy_candidates)  # [B, K, H]
        buy_scores = self.buy_head(buy_feat).squeeze(-1)  # [B, K]
        buy_feat_mean = buy_feat.mean(dim=1)  # [B, H]
        
        # portfolio 特征
        portfolio_feat = self.portfolio_encoder(portfolio)  # [B, M, H]
        sell_scores = self.sell_head(portfolio_feat).squeeze(-1)  # [B, M]
        port_feat_mean = portfolio_feat.mean(dim=1)  # [B, H]
        
        # 拼 joint_feat
        joint_feat = torch.cat([state_feat, buy_feat_mean, port_feat_mean], dim=-1)  # [B, 3H]
        
        # value
        value = self.value_head(joint_feat)  # [B, 1]
        return buy_scores, sell_scores, value

def generate_action_combinations(buy_scores, sell_scores, buy_indices, sell_ids, top_k, buy_candidates):
    """生成动作组合"""
    combinations = []
    sell_ids = list(set(sell_ids))
    
    # 所有卖出子集组合
    for r in range(len(sell_ids) + 1):
        for sell_subset in itertools.combinations(sell_ids, r):
            # 不买，纯卖
            combinations.append({
                "buy_index": -1,
                "sell_house_ids": list(sell_subset)
            })
    
    # 对每个可买入房产，搭配所有可能的卖出组合
    for buy_idx in buy_indices:
        for r in range(len(sell_ids) + 1):
            for sell_subset in itertools.combinations(sell_ids, r):
                if buy_idx < len(buy_candidates):
                    buy_house_id = int(buy_candidates[buy_idx][0].item())
                    if buy_house_id in sell_subset:
                        continue
                
                combinations.append({
                    "buy_index": buy_idx,
                    "sell_house_ids": list(sell_subset)
                })
    
    def score_combo(combo):
        buy_score = 0.0
        if combo["buy_index"] >= 0 and combo["buy_index"] < len(buy_scores):
            buy_score = buy_scores[combo["buy_index"]].item()
        
        sell_score = sum(
            sell_scores[sell_ids.index(sid)].item()
            for sid in combo["sell_house_ids"]
            if sid in sell_ids and sell_ids.index(sid) < len(sell_scores)
        )
        return buy_score + sell_score
    
    sorted_combos = sorted(combinations, key=score_combo, reverse=True)
    return sorted_combos[:top_k]

def preprocess_obs(obs):
    """预处理观察值"""
    scalar_keys = [
        "cash", "initial_cash", "income", "age",
        "marriage", "children", "start_year",
        "current_year", "year_remaining"
    ]
    state_flat = np.concatenate([obs[k] for k in scalar_keys], axis=0)
    state_flat = torch.tensor(state_flat, dtype=torch.float32).unsqueeze(0)
    
    buy_candidates = torch.tensor(obs["buy_candidates"], dtype=torch.float32).unsqueeze(0)
    portfolio = torch.tensor(obs["portfolio"], dtype=torch.float32).unsqueeze(0)
    
    return state_flat, buy_candidates, portfolio

def sample_action(buy_scores, sell_scores, buy_candidates, buy_indices, sell_ids, temperature=1.0, action_top_k=5):
    """从给定得分中采样动作组合"""
    combos = generate_action_combinations(
        buy_scores=buy_scores.squeeze(0),   # [K]
        sell_scores=sell_scores.squeeze(0), # [M]
        buy_indices=buy_indices,            # list of valid buy indices
        sell_ids=sell_ids,                  # list of valid sell house_ids
        top_k=action_top_k,
        buy_candidates=buy_candidates.squeeze(0)
    )
    
    if not combos:
        # 如果没有有效组合，返回默认动作
        return {
            "buy_index": -1,
            "sell_house_ids": []
        }, torch.tensor(0.0)
    
    # 组合得分
    scores = []
    for combo in combos:
        buy_score = 0.0
        if combo["buy_index"] >= 0 and combo["buy_index"] < len(buy_scores.squeeze(0)):
            buy_score = buy_scores.squeeze(0)[combo["buy_index"]].item()
        
        sell_score = sum(
            sell_scores.squeeze(0)[sell_ids.index(sid)].item()
            for sid in combo["sell_house_ids"]
            if sid in sell_ids and sell_ids.index(sid) < len(sell_scores.squeeze(0))
        )
        scores.append(buy_score + sell_score)
    
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    probs = F.softmax(scores_tensor / temperature, dim=0)
    dist = torch.distributions.Categorical(probs)
    idx = dist.sample()
    
    return combos[idx.item()], dist.log_prob(idx)