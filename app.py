from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import torch
import numpy as np
import pandas as pd
import sqlite3
import os
from copy import deepcopy
import json
from typing import Dict, List, Any
from datetime import datetime, timedelta

# Import classes and functions from the project
from real_estate_env import RealEstateEnv, RentalYieldScorer
from ppo_model import ActorCriticNet, preprocess_obs, sample_action
from utils import generate_random_agent_profile
from models import db, User, InvestmentRecord

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///real_estate_ai.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Session configuration
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour
app.config['SESSION_COOKIE_SECURE'] = False  # Set to False for development environment
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

@login_manager.unauthorized_handler
def unauthorized():
    if request.is_json:
        return jsonify({'error': 'Authentication required', 'authenticated': False}), 401
    return redirect(url_for('login', next=request.url))

# Global variables to store data
property_pool = None
growth_index = None
policy_net = None

def load_data_from_db():
    """Load data from SQLite database"""
    base_path = r"F:\Advanced DRL\New_DRL"
    db_path = os.path.join(base_path, "real_estate_data.db")
    
    if not os.path.exists(db_path):
        print(f"Database file does not exist: {db_path}")
        return None, None
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Load property data
        properties_df = pd.read_sql_query("SELECT * FROM properties", conn)
        property_pool = properties_df.to_dict(orient="records")
        
        # Process house type mapping - use actual column names
        HOUSE_TYPE_MAP = {"Apartment": 0, "Detachedhouse": 1}
        for p in property_pool:
            # Use actual column names
            if 'house_type' in p and isinstance(p['house_type'], str):
                p["house_type"] = HOUSE_TYPE_MAP.get(p["house_type"], 0)
            # Add compatibility fields
            if 'house_id' in p:
                p['Property_ID'] = p['house_id']
            if 'obj_regio1' in p:
                p['region'] = p['obj_regio1']
            if 'obj_livingSpace' in p:
                p['area'] = p['obj_livingSpace']
            if 'obj_noRooms' in p:
                p['rooms'] = p['obj_noRooms']
            if 'obj_yearConstructed' in p:
                p['year_built'] = p['obj_yearConstructed']
            if 'base_rent' in p:
                p['monthly_rent'] = p['base_rent']
        
        # Load price rent index
        index_df = pd.read_sql_query("SELECT * FROM price_rent_index", conn)
        growth_index = {}
        for region in index_df['region'].unique():
            region_data = index_df[index_df['region'] == region]
            growth_index[region] = dict(zip(region_data['year'], region_data['index_value']))
        
        conn.close()
        
        print(f"Loaded {len(property_pool)} property records and {len(growth_index)} regions' index data from database")
        return property_pool, growth_index
        
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return None, None

def load_data():
    """Load property data and growth index"""
    global property_pool, growth_index, policy_net
    
    print("Loading data and model...")
    
    # Data file paths - load directly from specified directory
    base_path = r"F:\Advanced DRL\New_DRL"
    properties_path = os.path.join(base_path, "all_properties.xlsx")
    index_path = os.path.join(base_path, "price_rent_index.xlsx")
    model_path = os.path.join(base_path, "ppo_policy_right.pth")
    
    print(f"Loading files from specified directory: {base_path}")
    
    # If files are not found in the specified directory, try the current directory
    if not os.path.exists(properties_path):
        properties_path = "all_properties.xlsx"
        print(f"all_properties.xlsx not found in specified directory, trying current directory")
    if not os.path.exists(index_path):
        index_path = "price_rent_index.xlsx"
        print(f"price_rent_index.xlsx not found in specified directory, trying current directory")
    if not os.path.exists(model_path):
        model_path = "ppo_policy_right.pth"
        print(f"ppo_policy_right.pth not found in specified directory, trying current directory")
    
    # Try loading data from SQLite database first
    print("Attempting to load data from SQLite database...")
    db_property_pool, db_growth_index = load_data_from_db()
    
    if db_property_pool and db_growth_index:
        property_pool = db_property_pool
        growth_index = db_growth_index
        print("Successfully loaded data from database")
    else:
        # If database loading fails, try loading from Excel files
        print("Database loading failed, attempting to load from Excel files...")
        property_path = properties_path
        
        if os.path.exists(property_path) and os.path.exists(index_path):
            print(f"Successfully found data files: {property_path}, {index_path}")
            try:
                property_df = pd.read_excel(property_path)
                index_df = pd.read_excel(index_path)
                
                property_pool = property_df.to_dict(orient="records")
                print(f"Loaded {len(property_pool)} property records")
                
                growth_index = {}
                for col in index_df.columns:
                    if col != "Year":
                        growth_index[col] = dict(zip(index_df["Year"], index_df[col]))
                print(f"Loaded growth index data, containing {len(growth_index)} regions")
                
                # Process house type mapping
                HOUSE_TYPE_MAP = {"Apartment": 0, "Detachedhouse": 1}
                for p in property_pool:
                    if 'house_type' in p and isinstance(p['house_type'], str):
                        p["house_type"] = HOUSE_TYPE_MAP.get(p["house_type"], 0)
            except Exception as e:
                print(f"Error reading Excel files: {e}")
                property_pool = []
                growth_index = {}
        else:
            # If files are not found, use sample data
            print(f"Data files not found: {property_path} or {index_path}")
            property_pool = []
            growth_index = {}
    
    # Load trained model
    try:
        state_dim = 9
        buy_candidates_dim = 10
        portfolio_dim = 10
        policy_net = ActorCriticNet(state_dim, buy_candidates_dim, portfolio_dim)
        
        if os.path.exists(model_path):
            policy_net.load_state_dict(torch.load(model_path, map_location="cpu"))
            policy_net.eval()
            print(f"Successfully loaded model: {model_path}")
        else:
            print(f"Warning: Trained model file not found: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        policy_net = None
    
    print(f"Data loading complete - Properties: {len(property_pool) if property_pool else 0} records, Growth index: {len(growth_index) if growth_index else 0} regions, Model: {'Loaded' if policy_net else 'Not loaded'}")

def run_investment_simulation(agent_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Run investment simulation based on PPO algorithm"""
    if not property_pool or not growth_index or policy_net is None:
        return {"error": "Data or model not properly loaded"}
    
    try:
        # Create environment - use configuration from PPO 1.3.ipynb
        env = RealEstateEnv(
            all_properties=[deepcopy(p) for p in property_pool],
            agent_profile=agent_profile,
            growth_index=growth_index,
            max_inventory=10,
            candidate_top_k=20,
            scorer=RentalYieldScorer(),
            reward_scale=10.0,
            cash_bonus_scale=0.1
        )
        
        obs, _ = env.reset()
        portfolio_history = []
        cash_history = []
        year_history = []
        
        # Run simulation using PPO policy
        for step in range(200):  # Maximum 200 steps
            state_flat, buy_candidates, portfolio = preprocess_obs(obs)
            
            # Get valid buy and sell options
            buy_indices = [i for i, p in enumerate(obs["buy_candidates"]) if p[0] > 0]
            sell_ids = [int(p[0]) for p in obs["portfolio"] if p[0] > 0]
            
            # Use PPO policy network for decision-making
            with torch.no_grad():
                buy_scores, sell_scores, _ = policy_net(state_flat, buy_candidates, portfolio)
            
            # Sample action
            if buy_indices:  # If there are properties available to buy
                action, _ = sample_action(
                    buy_scores, sell_scores,
                    buy_candidates=buy_candidates.squeeze(0),
                    buy_indices=buy_indices,
                    sell_ids=sell_ids,
                    temperature=1.0,
                    action_top_k=5
                )
                
                padded_sell_ids = action["sell_house_ids"] + [0] * (env.max_inventory - len(action["sell_house_ids"]))
                action_tensor = {
                    "buy_index": np.array([action["buy_index"]]),
                    "sell_house_ids": np.array(padded_sell_ids)
                }
            else:
                # No properties available to buy, hold only
                action_tensor = {
                    "buy_index": np.array([-1]),
                    "sell_house_ids": np.array([0] * env.max_inventory)
                }
            
            # Execute action
            obs, reward, done, _ = env.step(action_tensor)
            
            # Record history data - only record properties retrieved by the model
            current_cash = obs["cash"][0]
            current_portfolio = []
            
            for p in obs["portfolio"]:
                if p[0] > 0:  # Valid property
                    house_id = int(p[0])
                    # Look up full information from property_pool
                    property_details = None
                    for prop in property_pool:
                        if (prop.get('house_id') == house_id or 
                            prop.get('Property_ID') == house_id or 
                            prop.get('id') == house_id):
                            property_details = prop
                            break
                    
                    # Handle NaN values to ensure proper JSON serialization
                    def safe_float(value, default=0):
                        try:
                            result = float(value)
                            return result if not (result != result) else default  # NaN check
                        except (ValueError, TypeError):
                            return default
                    
                    def safe_int(value, default=0):
                        try:
                            result = int(value)
                            return result if not (result != result) else default  # NaN check
                        except (ValueError, TypeError):
                            return default
                    
                    portfolio_item = {
                        "house_id": house_id,
                        "purchase_price": safe_float(p[1]),
                        "market_price": safe_float(p[2]),
                        "monthly_rent": safe_float(p[3]),
                        "plz": safe_int(p[4]) if p[4] > 0 else 0
                    }
                    
                    # Add detailed property information
                    if property_details:
                        area_value = safe_float(property_details.get('obj_livingSpace', property_details.get('area', 0)))
                        rooms_value = safe_float(property_details.get('obj_noRooms', property_details.get('rooms', 0)))
                        year_built_value = safe_int(property_details.get('obj_yearConstructed', property_details.get('year_built', 0)))
                        
                        portfolio_item.update({
                            "address": property_details.get('address', 'Unknown'),
                            "region": property_details.get('obj_regio1', property_details.get('region', 'Unknown')),
                            "postcode": property_details.get('plz', property_details.get('postcode', 'Unknown')),
                            "area": area_value,
                            "rooms": rooms_value,
                            "year_built": year_built_value,
                            "house_type": safe_int(property_details.get('house_type', 0)),
                            "rental_yield": (safe_float(p[3]) * 12 / safe_float(p[2])) if safe_float(p[2]) > 0 else 0,
                            "price": safe_float(property_details.get('price', safe_float(p[2])))
                        })
                    
                    current_portfolio.append(portfolio_item)
            
            portfolio_history.append(current_portfolio)
            cash_history.append(float(current_cash))
            year_history.append(int(obs["current_year"][0]))
            
            if done:
                break
        
        # Calculate final results
        final_cash = cash_history[-1] if cash_history else agent_profile["initial_cash"]
        final_portfolio_value = sum(p["market_price"] for p in portfolio_history[-1]) if portfolio_history else 0
        final_total_assets = final_cash + final_portfolio_value
        
        # Count the number of properties considered by the model
        total_properties_considered = len(env.current_market) if hasattr(env, 'current_market') else 0
        properties_purchased = len(portfolio_history[-1]) if portfolio_history else 0
        
        return {
            "success": True,
            "final_cash": final_cash,
            "final_portfolio_value": final_portfolio_value,
            "final_total_assets": final_total_assets,
            "initial_cash": agent_profile["initial_cash"],
            "total_return": ((final_total_assets - agent_profile["initial_cash"]) / agent_profile["initial_cash"]) * 100,
            "portfolio_history": portfolio_history,
            "cash_history": cash_history,
            "year_history": year_history,
            "final_portfolio": portfolio_history[-1] if portfolio_history else [],
            "total_properties_considered": total_properties_considered,
            "properties_purchased": properties_purchased,
            "model_type": "PPO",
            "algorithm": "Proximal Policy Optimization",
            "environment": "MDP-based Real Estate Investment"
        }
        
    except Exception as e:
        return {"error": f"Error during simulation: {str(e)}"}

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/invest')
@login_required
def invest():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        # Validate input
        if not username or not email or not password:
            return jsonify({'success': False, 'message': 'Please fill in all required fields'})
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            return jsonify({'success': False, 'message': 'Username already exists'})
        
        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'message': 'Email already registered'})
        
        # Create new user
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
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'success': False, 'message': 'Please enter username and password'})
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user, remember=True, duration=timedelta(hours=24))
            session.permanent = True
            return jsonify({'success': True, 'message': 'Login successful!', 'redirect': url_for('dashboard')})
        else:
            return jsonify({'success': False, 'message': 'Invalid username or password'})
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get user's investment records
    records = InvestmentRecord.query.filter_by(user_id=current_user.id).order_by(InvestmentRecord.created_at.desc()).all()
    return render_template('dashboard.html', user=current_user, records=records)

@app.route('/analysis')
@login_required
def analysis():
    print(f"Analysis route accessed by user: {current_user.username if current_user.is_authenticated else 'Anonymous'}")
    print(f"User authenticated: {current_user.is_authenticated}")
    return render_template('analysis.html')

@app.route('/debug_auth')
def debug_auth():
    return jsonify({
        'authenticated': current_user.is_authenticated,
        'user_id': current_user.id if current_user.is_authenticated else None,
        'username': current_user.username if current_user.is_authenticated else None
    })

@app.route('/test_analysis')
def test_analysis():
    return "Analysis route is working! This is a test without login requirement."

@app.route('/debug_routes')
def debug_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'rule': str(rule)
        })
    return jsonify({'routes': routes})

@app.route('/simulate', methods=['POST'])
@login_required
def simulate():
    try:
        data = request.json
        
        # Build agent profile
        agent_profile = {
            "start_year": int(data.get('start_year', 2024)),
            "age": int(data.get('age', 30)),
            "marriage": int(data.get('marriage', 0)),
            "children": int(data.get('children', 0)),
            "income": int(data.get('income', 80000)),
            "initial_cash": int(data.get('initial_cash', 200000))
        }
        
        # Run simulation
        result = run_investment_simulation(agent_profile)
        
        # Save investment record to database
        if result.get('success'):
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
                investment_record.set_investment_history({
                    'cash_history': result['cash_history'],
                    'year_history': result['year_history'],
                    'portfolio_history': result['portfolio_history']
                })
                
                db.session.add(investment_record)
                db.session.commit()
            except Exception as db_error:
                print(f"Error saving investment record: {db_error}")
                db.session.rollback()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Error processing request: {str(e)}"})

@app.route('/property/<int:property_id>')
@login_required
def property_detail(property_id):
    """Property detail page"""
    # Look up property from property_pool
    property_info = None
    for prop in property_pool:
        if prop.get('id') == property_id or prop.get('Property_ID') == property_id:
            property_info = prop
            break
    
    if not property_info:
        return "Property not found", 404
    
    return render_template('property_detail.html', property=property_info)

@app.route('/api/investment_history/<int:record_id>')
@login_required
def get_investment_history(record_id):
    """Get investment history details"""
    record = InvestmentRecord.query.filter_by(id=record_id, user_id=current_user.id).first()
    if not record:
        return jsonify({'success': False, 'message': 'Record not found'})
    
    return jsonify({
        'success': True,
        'record': record.to_dict()
    })

@app.route('/api/dashboard_stats')
@login_required
def get_dashboard_stats():
    """Get dashboard statistics"""
    try:
        # Get all investment records for the user
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
        
        # Calculate statistics
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
    """Get recent investment records"""
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
    """Create database tables"""
    with app.app_context():
        db.create_all()
        print("Database tables created successfully")

if __name__ == '__main__':
    load_data()
    create_tables()
    print("Starting web server...")
    app.run(debug=True, host='0.0.0.0', port=5000)