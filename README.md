# Advanced Deep Reinforcement Learning for Real Estate Investment

## Project Overview
This project demonstrates a complete Deep Reinforcement Learning (DRL) research pipeline applying **Proximal Policy Optimization (PPO)** algorithm to real estate investment decision-making.

### Core Research Content: **PPO 1.3.ipynb**
**This is the main research notebook containing the complete DRL pipeline:**
- **Data Preparation**: Real estate dataset processing and feature engineering
- **MDP Modeling**: Markov Decision Process formulation for investment scenarios
- **PPO Algorithm**: Custom Actor-Critic implementation with advanced optimization
- **Results & Comparison**: Comprehensive performance analysis and benchmarking against baseline strategies

### Web Application Demo: **app.py**
Interactive web interface showcasing the trained DRL model in action with real-time investment recommendations and portfolio management.

## Key Research Contributions:
- **Complete DRL Pipeline** - From raw data to trained PPO agent with comprehensive evaluation
- **MDP Formulation** - Novel state-action space design for real estate investment scenarios
- **PPO Implementation** - Custom Actor-Critic architecture with clipped surrogate objective
- **Performance Analysis** - Extensive benchmarking with risk-return metrics and statistical validation
- **Practical Application** - Real-time web deployment demonstrating DRL model integration

## Quick Start

### For Research Review: 
**Simply open `PPO 1.3.ipynb` in Jupyter Notebook to explore the complete DRL research pipeline.**

### For Web Demo (Optional Environment Setup):
If you want to run the web application demo locally:

#### Step 1: Clone the Repository
```bash
git clone https://github.com/Kairui-Zhang/A-Reinforcement-Learning-Based-Real-EstateInvestment-Recommender-Using-ProximalPolicy-Optimization.git
cd A-Reinforcement-Learning-Based-Real-EstateInvestment-Recommender-Using-ProximalPolicy-Optimization
```

#### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Run Web Application
```bash
python app.py
```
Access the web demo at: **http://127.0.0.1:5000/**

**Note**: 
- All required data files (`all_properties.xlsx`, `price_rent_index.xlsx`, `ppo_policy_right.pth`) are already included in the repository.
- If you encounter path-related errors, you may need to update the `base_path` variable in `app.py` line 52 to match your local directory structure.

## System Architecture
This system is a Flask-based DRL application that integrates:

- **Backend**: Flask, SQLAlchemy, PyTorch
- **Frontend**: HTML, CSS, JavaScript
- **AI Engine**: Custom PPO implementation with Actor-Critic networks
- **Database**: SQLite for user management and investment records
- **Visualization**: Interactive charts for performance analysis

## Project Structure
```
A-Reinforcement-Learning-Based-Real-EstateInvestment-Recommender-Using-ProximalPolicy-Optimization/
├── PPO 1.3.ipynb              # Complete PPO implementation & training
├── EDA.ipynb                  # Exploratory Data Analysis
├── templates/
│   ├── dashboard.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── analysis.html
│   └── property_detail.html
├── app.py                     # Main Flask web application
├── app_simple.py              # Simplified Flask application
├── ppo_model.py              # PPO algorithm implementation
├── real_estate_env.py        # MDP environment modeling
├── models.py                 # Database models
├── utils.py                  # Utility functions
├── all_properties.xlsx       # Property dataset
├── price_rent_index.xlsx     # Market indices
├── ppo_policy_right.pth      # Trained PPO model
├── real_estate_data.db       # Main database
├── instance/
│   └── real_estate_ai.db     # User authentication database
├── requirements.txt
└── readme.md
```

## Technical Highlights
- **MDP Formulation**: Multi-dimensional state space with property features, market conditions, and portfolio status
- **PPO Implementation**: Clipped surrogate objective with Actor-Critic architecture
- **Risk Management**: Sophisticated reward engineering with penalty mechanisms
- **Real-time Integration**: Seamless DRL model deployment in web environment