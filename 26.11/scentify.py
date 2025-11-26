import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import random

# ============================================================================
# API CONFIGURATION
# ============================================================================
# Load environment variables from .env file for secure API key management
load_dotenv()

# Get Fragella API key from environment variables
FRAGELLA_API_KEY = os.getenv("FRAGELLA_API_KEY")

# Verify API key is loaded before proceeding
if not FRAGELLA_API_KEY:
    st.error("FRAGELLA_API_KEY not found. Please create a .env file with your API key.")
    st.stop()

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
# Set up the Streamlit page with title and layout
st.set_page_config(
    page_title="Scentify - Perfume Finder",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
def apply_custom_styling():
    """
    Apply custom CSS styling to create a clean, elegant interface.
    Uses pastel purple, gray, white, and black color scheme.
    Adds minimal floral background elements without emojis.
    """
    st.markdown("""
        <style>
        /* Import Google Fonts for elegant typography */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Great+Vibes&display=swap');
        
        /* Main application background with subtle gradient */
        .stApp {
            background: linear-gradient(135deg, #faf9fc 0%, #f0eef5 100%);
            font-family: 'Poppins', sans-serif;
        }
        
        /* Add minimal floral background pattern */
        .stApp::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(200, 180, 220, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 90% 80%, rgba(200, 180, 220, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 50% 50%, rgba(200, 180, 220, 0.04) 0%, transparent 50%);
            pointer-events: none;
            z-index: 0;
        }
        
        /* Main header styling */
        .main-header {
            font-size: 70px;
            font-weight: 600;
            color: #6b5b95;
            text-align: center;
            margin-bottom: 10px;
            letter-spacing: 3px;
        }
        
        /* Section card styling for landing page */
        .section-card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 20px 0;
            border: 2px solid #e8e4f0;
            height: 250px;  /* Fixed height */
            min-height: 250px;
            max-height: 250px;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;  /* Align all content to top */
            overflow: hidden;  /* Prevent content overflow */
        }
        
        /* Section title styling */
        .section-title {
            font-size: 28px;
            font-weight: 600;
            color: #6b5b95;
            margin-bottom: 30px;  /* 30px spacing below title */
            text-align: center;
            height: 35px;  /* Fixed height for alignment */
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;  /* Prevent title from shrinking */
        }
        
        /* Section description styling */
        .section-description {
            font-size: 16px;
            color: #666;
            margin-top: 0px;  /* No extra spacing (title handles the gap) */
            margin-bottom: 20px;
            line-height: 1.6;
            text-align: center;
            flex: 1;  /* Take remaining space */
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #6b5b95;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 30px;
            font-size: 16px;
            font-weight: 500;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton > button:hover {
            background-color: #5a4a7f;
            color: white !important;
            box-shadow: 0 4px 12px rgba(107, 91, 149, 0.4);
            transform: translateY(-2px);
        }
        
        /* Prevent red outline/border on buttons and inputs */
        button:focus-visible,
        button:focus,
        input:focus-visible {
            outline: none !important;
        }
        
        /* Logo container styling - centered on page */
        .logo-container {
            cursor: pointer;
            display: block;
            text-align: center;
            transition: opacity 0.3s ease;
            margin: 0 auto 5px auto;
            width: 100%;
        }
        
        .logo-container:hover {
            opacity: 0.8;
        }
        
        .logo-text {
            font-size: 70px;
            font-weight: 600;
            color: #6b5b95;
            letter-spacing: 3px;
            text-align: center;
            user-select: none;
            display: block;
            width: 100%;
        }
        
        .logo-image {
            max-height: 80px;
            width: auto;
            display: block;
            margin: 0 auto;
        }
        
        /* Tagline styling below logo */
        .logo-tagline {
            font-family: 'Great Vibes', cursive;
            font-size: 24px;
            color: #6b5b95;
            text-align: center;
            margin-top: 0px;
            margin-bottom: 10px;
            font-weight: 400;
            display: block;
            width: 100%;
        }
        
        /* Ensure equal column widths on landing page */
        .stColumn {
            min-width: 0;
            flex: 1 1 0px;
        }
        
        /* Responsive adjustments for smaller screens */
        @media (max-width: 768px) {
            .section-card {
                height: auto;
                min-height: 300px;
                max-height: none;
            }
            
            .main-header {
                font-size: 36px;
            }
        }
        
        /* Filter tag styling */
        .filter-tag {
            display: inline-block;
            background-color: #e8e4f0;
            color: #6b5b95;
            padding: 8px 16px;
            border-radius: 20px;
            margin: 5px;
            font-size: 14px;
            border: 1px solid #c8b8d8;
        }
        
        /* Perfume card styling */
        .perfume-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            border: 2px solid #e8e4f0;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .perfume-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(107, 91, 149, 0.15);
        }
        
        /* Input field styling */
        .stTextInput > div > div > input {
            border-radius: 8px;
            border: 2px solid #e8e4f0;
            padding: 10px;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #6b5b95 !important;
            outline: none !important;
            box-shadow: none !important;
        }
        
        /* Remove red from ALL input and select elements */
        input:focus,
        input:active,
        textarea:focus,
        textarea:active,
        select:focus,
        select:active,
        .stSelectbox:focus,
        .stSelectbox:active,
        .stMultiSelect:focus,
        .stMultiSelect:active {
            border-color: #6b5b95 !important;
            outline: none !important;
            box-shadow: none !important;
            color: #333 !important;
        }
        
        /* Override Streamlit's default red focus colors */
        [data-baseweb="input"]:focus-within,
        [data-baseweb="select"]:focus-within {
            border-color: #6b5b95 !important;
            box-shadow: none !important;
        }
        
        /* Remove red from ALL interactive elements on hover/click/focus */
        button:focus,
        button:active,
        input:focus,
        input:active,
        textarea:focus,
        textarea:active,
        select:focus,
        select:active,
        summary:focus,
        summary:active,
        a:focus,
        a:active,
        [role="button"]:focus,
        [role="button"]:active {
            outline: none !important;
            box-shadow: none !important;
        }
        
        /* Keep borders purple instead of red */
        input:focus,
        textarea:focus,
        select:focus {
            border-color: #6b5b95 !important;
        }
        
        /* Target Streamlit's selectbox specifically */
        .stSelectbox > div > div,
        .stSelectbox > div > div:focus,
        .stSelectbox > div > div:focus-within,
        [data-baseweb="select"],
        [data-baseweb="select"]:focus,
        [data-baseweb="select"]:focus-within,
        [data-baseweb="select"] > div,
        [data-baseweb="select"] > div:focus {
            border-color: #6b5b95 !important;
            outline: none !important;
            box-shadow: none !important;
        }
        
        /* Prevent text from turning red on hover EVERYWHERE */
        p:hover,
        span:hover,
        div:hover,
        label:hover,
        h1:hover, h2:hover, h3:hover, h4:hover, h5:hover, h6:hover,
        summary:hover,
        a:hover {
            color: inherit !important;
        }
        
        /* Prevent expander header text from turning red */
        .streamlit-expanderHeader:hover,
        [data-testid="stExpander"] summary:hover {
            color: #6b5b95 !important;
        }
        
        /* Change dropdown arrow from red to purple - ULTRA AGGRESSIVE + HOVER */
        [data-baseweb="select"] svg,
        [data-baseweb="select"] svg *,
        [data-baseweb="select"] path,
        [data-baseweb="select"]:hover svg,
        [data-baseweb="select"]:hover svg *,
        [data-baseweb="select"]:hover path,
        .stSelectbox svg,
        .stSelectbox svg *,
        .stSelectbox path,
        .stSelectbox:hover svg,
        .stSelectbox:hover svg *,
        .stSelectbox:hover path,
        .stMultiSelect svg,
        .stMultiSelect svg *,
        .stMultiSelect path,
        .stMultiSelect:hover svg,
        .stMultiSelect:hover svg *,
        .stMultiSelect:hover path,
        [role="button"] svg,
        [role="button"] svg *,
        [role="button"] path,
        [role="button"]:hover svg,
        [role="button"]:hover svg *,
        [role="button"]:hover path,
        [aria-expanded] svg,
        [aria-expanded] svg *,
        [aria-expanded] path,
        [aria-expanded]:hover svg,
        [aria-expanded]:hover svg *,
        [aria-expanded]:hover path {
            color: #6b5b95 !important;
            fill: #6b5b95 !important;
            stroke: #6b5b95 !important;
        }
        
        /* Target the dropdown chevron specifically + HOVER */
        svg[viewBox="0 0 10 6"],
        svg[viewBox="0 0 10 6"] *,
        svg[viewBox="0 0 10 6"] path,
        *:hover svg[viewBox="0 0 10 6"],
        *:hover svg[viewBox="0 0 10 6"] *,
        *:hover svg[viewBox="0 0 10 6"] path {
            color: #6b5b95 !important;
            fill: #6b5b95 !important;
            stroke: #6b5b95 !important;
        }
        
        /* Change selected filter tags from red to purple */
        [data-baseweb="tag"],
        .stMultiSelect [data-baseweb="tag"] {
            background-color: #6b5b95 !important;
            color: white !important;
        }
        
        /* Make the X icon on filter tags white */
        [data-baseweb="tag"] svg,
        [data-baseweb="tag"] svg *,
        .stMultiSelect [data-baseweb="tag"] svg,
        .stMultiSelect [data-baseweb="tag"] svg * {
            color: white !important;
            fill: white !important;
            stroke: white !important;
        }
        
        
        
        /* Slider styling */
        .stSlider > div > div > div {
            background-color: #6b5b95;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Back button styling */
        .back-button {
            background-color: #9b8bb5 !important;
        }
        
        /* Perfume detail view styling */
        .perfume-detail-title {
            font-size: 36px;
            font-weight: 600;
            color: #6b5b95;
            margin-bottom: 10px;
        }
        
        .perfume-detail-brand {
            font-size: 20px;
            color: #888;
            font-style: italic;
            margin-bottom: 20px;
        }
        
        /* Note list styling */
        .note-list {
            background: #f8f7fa;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #6b5b95;
        }
        
        .note-category {
            font-weight: 600;
            color: #6b5b95;
            font-size: 16px;
            margin-bottom: 5px;
        }
        
        /* Questionnaire styling - clean, no white box */
        .bipolar-slider {
            padding: 20px 0;
            margin: 20px 0;
        }
        
        .bipolar-label {
            font-size: 16px;
            font-weight: 500;
            color: #333;
        }
        
        /* Questionnaire styling - clean, no white box */
        .bipolar-slider {
            padding: 20px 0;
            margin: 20px 0;
        }
        
        .bipolar-label {
            font-size: 16px;
            font-weight: 500;
            color: #333;
        }
        
        /* Chart container styling */
        .chart-container {
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            border: 2px solid #e8e4f0;
        }
        
        /* Hide Streamlit's default tick marks */
        [data-testid="stSlider"] [data-testid="stTickBar"] {
            display: none !important;
        }
        
        /* Custom number scale */
        .scale-numbers {
            display: flex;
            justify-content: space-between;
            margin-top: 10px;
            font-size: 13px;
            color: #333;
            padding: 0 4px;
            font-weight: 400;
        }
        
        /* HIDE ONLY TICK MARKS - KEEP SLIDER HANDLE */
        
        /* Target only tick mark and dot elements, NOT the handle */
        .stSlider .rc-slider-mark,
        .stSlider .rc-slider-mark-text,
        .stSlider .rc-slider-dot,
        .stSlider .rc-slider-dot-active,
        div[data-baseweb="slider"] .rc-slider-mark,
        div[data-baseweb="slider"] .rc-slider-mark-text,
        div[data-baseweb="slider"] .rc-slider-dot,
        div[data-baseweb="slider"] .rc-slider-dot-active,
        [data-baseweb="slider"] .rc-slider-mark,
        [data-baseweb="slider"] .rc-slider-mark-text,
        [data-baseweb="slider"] .rc-slider-dot,
        [data-baseweb="slider"] .rc-slider-dot-active {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
        }
        
        /* Hide tick bars */
        .stSlider [data-testid="stTickBar"],
        [data-testid="stTickBar"] {
            display: none !important;
        }
        
        /* Make sure the handle/thumb is visible and functional */
        .stSlider .rc-slider-handle,
        div[data-baseweb="slider"] .rc-slider-handle,
        [data-baseweb="slider"] .rc-slider-handle,
        div[role="slider"] {
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
            cursor: pointer !important;
        }
        </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA PERSISTENCE FILES
# ============================================================================
# File paths for storing user data and interactions (organized in data folder)
USER_INTERACTIONS_FILE = "data/user_interactions.json"
USER_INVENTORY_FILE = "data/user_perfume_inventory.json"
PERFUME_RANKINGS_FILE = "data/perfume_rankings.json"
ML_MODEL_FILE = "data/ml_model.pkl"
ML_SCALER_FILE = "data/ml_scaler.pkl"

# ============================================================================
# ML CONFIGURATION
# ============================================================================
# Configuration for machine learning recommendation system
ML_CONFIG = {
    'negative_samples_ratio': 2,  # For each positive sample, generate N negative samples
    'min_inventory_size': 2,  # Minimum perfumes in inventory to train ML model
    'model_type': 'logistic_regression',  # Options: 'logistic_regression', 'decision_tree'
    'random_state': 42,
    'min_recommendation_probability': 0.5,  # Minimum probability to show recommendation
    'diversity_threshold': 0.3  # Minimum diversity in scent types for recommendations
}

# ============================================================================
# DATA LOADING AND SAVING FUNCTIONS
# ============================================================================

def load_user_interactions() -> List[Dict]:
    """
    Load user interaction history from JSON file.
    This data is used for machine learning recommendations.
    
    Returns:
        List of interaction dictionaries with perfume_id, interaction_type, and timestamp
    """
    if os.path.exists(USER_INTERACTIONS_FILE):
        try:
            with open(USER_INTERACTIONS_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_user_interactions(interactions: List[Dict]):
    """
    Save user interaction history to JSON file.
    
    Args:
        interactions: List of interaction dictionaries to save
    """
    os.makedirs('data', exist_ok=True)
    with open(USER_INTERACTIONS_FILE, 'w') as f:
        json.dump(interactions, f, indent=2)

def load_user_inventory() -> List[Dict]:
    """
    Load user's personal perfume collection from JSON file.
    This persists across sessions.
    
    Returns:
        List of perfumes in user's inventory
    """
    if os.path.exists(USER_INVENTORY_FILE):
        try:
            with open(USER_INVENTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_user_inventory(inventory: List[Dict]):
    """
    Save user's personal perfume collection to JSON file.
    
    Args:
        inventory: List of perfumes to save
    """
    os.makedirs('data', exist_ok=True)
    with open(USER_INVENTORY_FILE, 'w') as f:
        json.dump(inventory, f, indent=2)

def load_perfume_rankings() -> Dict:
    """
    Load perfume ranking scores calculated by ML algorithm.
    
    Returns:
        Dictionary mapping perfume IDs to ranking scores
    """
    if os.path.exists(PERFUME_RANKINGS_FILE):
        try:
            with open(PERFUME_RANKINGS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_perfume_rankings(rankings: Dict):
    """
    Save perfume ranking scores to JSON file.
    
    Args:
        rankings: Dictionary of perfume IDs and their scores
    """
    os.makedirs('data', exist_ok=True)
    with open(PERFUME_RANKINGS_FILE, 'w') as f:
        json.dump(rankings, f, indent=2)

# ============================================================================
# MACHINE LEARNING - USER INTERACTION TRACKING
# ============================================================================

def record_interaction(perfume_id: str, interaction_type: str):
    """
    Record a user interaction with a perfume for machine learning.
    Interaction types: 'view', 'favorite', 'add_to_inventory', 'click'
    
    Note: Only 'click' interactions affect popularity score/ranking.
    
    Args:
        perfume_id: Unique identifier of the perfume
        interaction_type: Type of interaction performed
    """
    # Load existing interactions
    interactions = load_user_interactions()
    
    # Create new interaction record
    interaction = {
        "perfume_id": perfume_id,
        "interaction_type": interaction_type,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add to interactions list
    interactions.append(interaction)
    
    # Save updated interactions
    save_user_interactions(interactions)
    
    # Update perfume rankings based on new interaction
    update_perfume_rankings()

def update_perfume_rankings():
    """
    Machine Learning algorithm to update perfume rankings based on user interactions.
    Only counts 'click' interactions for popularity score.
    """
    # Load all interactions
    interactions = load_user_interactions()
    
    # Calculate scores - only count 'click' interactions
    rankings = {}
    for interaction in interactions:
        # Only count clicks for popularity
        if interaction.get('interaction_type') == 'click':
            perfume_id = interaction['perfume_id']
            if perfume_id in rankings:
                rankings[perfume_id] += 1
            else:
                rankings[perfume_id] = 1
    
    # Save updated rankings
    save_perfume_rankings(rankings)
    
    return rankings

def get_ml_sorted_perfumes(perfumes: List[Dict]) -> List[Dict]:
    """
    Sort perfumes based on machine learning ranking scores.
    Perfumes with more clicks appear first (popularity based on clicks only).
    
    Args:
        perfumes: List of perfume dictionaries to sort
    
    Returns:
        List of perfumes sorted by ML ranking score (highest first)
    """
    # Load current rankings
    rankings = load_perfume_rankings()
    
    # Sort perfumes by their ranking score
    sorted_perfumes = sorted(
        perfumes,
        key=lambda p: rankings.get(p.get('id', p['name']), 0),
        reverse=True
    )
    
    return sorted_perfumes

# ============================================================================
# MACHINE LEARNING RECOMMENDATION SYSTEM
# ============================================================================

def extract_perfume_features(perfume: Dict) -> np.ndarray:
    """
    Extract numerical feature vector from a perfume for ML model.
    Uses all available scent characteristics from the API.
    
    Features include:
    - Main accords (converted to binary/numeric features)
    - Seasonality scores (Winter, Fall, Spring, Summer)
    - Occasion scores (Day, Night)
    - Longevity (converted to numeric)
    - Sillage/Strength (converted to numeric)
    - Gender preference (converted to numeric)
    
    Args:
        perfume: Perfume dictionary
    
    Returns:
        numpy array of numerical features
    """
    features = []
    
    # Define all possible main accords (from common perfume accords)
    all_accords = [
        'floral', 'fresh', 'woody', 'citrus', 'oriental', 'spicy', 
        'sweet', 'gourmand', 'fruity', 'aromatic', 'green', 'aquatic',
        'leather', 'powdery', 'herbal', 'amber', 'musk', 'vanilla',
        'rose', 'jasmine', 'lavender', 'bergamot', 'sandalwood', 'patchouli',
        'oud', 'vetiver', 'tobacco', 'animalic', 'earthy', 'smoky'
    ]
    
    # Extract main accords as binary features (0 or 1)
    perfume_accords = perfume.get('main_accords', [])
    perfume_accords_lower = [accord.lower() if isinstance(accord, str) else '' for accord in perfume_accords]
    
    for accord in all_accords:
        # Check if this accord is present in the perfume
        has_accord = any(accord in pa for pa in perfume_accords_lower)
        features.append(1.0 if has_accord else 0.0)
    
    # Seasonality features (4 features: Winter, Fall, Spring, Summer)
    seasonality = perfume.get('seasonality', {})
    features.append(float(seasonality.get('Winter', 3)) / 5.0)  # Normalize to 0-1
    features.append(float(seasonality.get('Fall', 3)) / 5.0)
    features.append(float(seasonality.get('Spring', 3)) / 5.0)
    features.append(float(seasonality.get('Summer', 3)) / 5.0)
    
    # Occasion features (2 features: Day, Night)
    occasion = perfume.get('occasion', {})
    features.append(float(occasion.get('Day', 3)) / 5.0)
    features.append(float(occasion.get('Night', 3)) / 5.0)
    
    # Longevity - convert to numeric (0-1 scale)
    longevity_map = {
        'very weak': 0.1, 'weak': 0.3, 'moderate': 0.5, 'long lasting': 0.7,
        'long-lasting': 0.7, 'eternal': 0.9, 'very long lasting': 0.9
    }
    longevity_str = str(perfume.get('longevity', 'moderate')).lower()
    longevity_score = longevity_map.get(longevity_str, 0.5)
    features.append(longevity_score)
    
    # Sillage/Strength - convert to numeric (0-1 scale)
    sillage_map = {
        'intimate': 0.2, 'weak': 0.3, 'moderate': 0.5, 'strong': 0.7, 'enormous': 0.9
    }
    sillage_str = str(perfume.get('sillage', 'moderate')).lower()
    sillage_score = sillage_map.get(sillage_str, 0.5)
    features.append(sillage_score)
    
    # Gender - convert to numeric
    gender_map = {'Male': 0.0, 'Unisex': 0.5, 'Female': 1.0}
    gender = perfume.get('gender', 'Unisex')
    features.append(gender_map.get(gender, 0.5))
    
    return np.array(features)

def build_training_dataset(user_inventory: List[Dict], all_perfumes: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build training dataset with positive samples (owned perfumes) and 
    negative samples (randomly selected non-owned perfumes).
    
    Args:
        user_inventory: List of perfumes user owns (positive samples)
        all_perfumes: Complete catalog of available perfumes
    
    Returns:
        Tuple of (X, y) where X is feature matrix and y is labels (1=owned, 0=not owned)
    """
    X_positive = []
    X_negative = []
    
    # Get IDs of owned perfumes
    owned_ids = {p['id'] for p in user_inventory}
    
    # Extract features from positive samples (owned perfumes)
    for perfume in user_inventory:
        features = extract_perfume_features(perfume)
        X_positive.append(features)
    
    # Get all non-owned perfumes
    non_owned = [p for p in all_perfumes if p['id'] not in owned_ids]
    
    # Sample negative examples
    num_negatives = len(user_inventory) * ML_CONFIG['negative_samples_ratio']
    if len(non_owned) > num_negatives:
        negative_samples = random.sample(non_owned, int(num_negatives))
    else:
        negative_samples = non_owned
    
    # Extract features from negative samples
    for perfume in negative_samples:
        features = extract_perfume_features(perfume)
        X_negative.append(features)
    
    # Combine positive and negative samples
    X = np.array(X_positive + X_negative)
    y = np.array([1] * len(X_positive) + [0] * len(X_negative))
    
    return X, y

def train_ml_model(user_inventory: List[Dict], all_perfumes: List[Dict]) -> Optional[Tuple]:
    """
    Train a personalized ML model based on user's perfume inventory.
    
    Uses either Logistic Regression or Decision Tree to learn user preferences
    from the scent characteristics of owned perfumes.
    
    Args:
        user_inventory: List of perfumes user owns
        all_perfumes: Complete catalog of available perfumes
    
    Returns:
        Tuple of (model, scaler, feature_importance) or None if training failed
    """
    # Check if user has enough perfumes to train
    if len(user_inventory) < ML_CONFIG['min_inventory_size']:
        return None
    
    try:
        # Build training dataset
        X, y = build_training_dataset(user_inventory, all_perfumes)
        
        if len(X) == 0:
            return None
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model based on configuration
        if ML_CONFIG['model_type'] == 'decision_tree':
            model = DecisionTreeClassifier(
                max_depth=5,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=ML_CONFIG['random_state']
            )
        else:  # Default to logistic regression
            model = LogisticRegression(
                max_iter=1000,
                random_state=ML_CONFIG['random_state'],
                solver='lbfgs'
            )
        
        # Fit the model
        model.fit(X_scaled, y)
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            feature_importance = np.abs(model.coef_[0])
        else:
            feature_importance = None
        
        # Save model and scaler
        os.makedirs('data', exist_ok=True)
        with open(ML_MODEL_FILE, 'wb') as f:
            pickle.dump(model, f)
        with open(ML_SCALER_FILE, 'wb') as f:
            pickle.dump(scaler, f)
        
        return model, scaler, feature_importance
    
    except Exception as e:
        print(f"Error training ML model: {e}")
        return None

def load_ml_model() -> Optional[Tuple]:
    """
    Load previously trained ML model and scaler from disk.
    
    Returns:
        Tuple of (model, scaler) or None if not found
    """
    try:
        if os.path.exists(ML_MODEL_FILE) and os.path.exists(ML_SCALER_FILE):
            with open(ML_MODEL_FILE, 'rb') as f:
                model = pickle.load(f)
            with open(ML_SCALER_FILE, 'rb') as f:
                scaler = pickle.load(f)
            return model, scaler
        return None
    except Exception as e:
        print(f"Error loading ML model: {e}")
        return None

def get_ml_recommendations(user_inventory: List[Dict], 
                          all_perfumes: List[Dict], 
                          top_n: int = 10,
                          ensure_diversity: bool = True) -> List[Dict]:
    """
    Generate personalized perfume recommendations using trained ML model.
    
    This is the main recommendation function that:
    1. Trains/loads a personalized model based on user's inventory
    2. Scores all non-owned perfumes
    3. Returns top N recommendations with probability scores
    4. Optionally ensures diversity in scent types
    
    Args:
        user_inventory: List of perfumes user owns
        all_perfumes: Complete catalog of available perfumes
        top_n: Number of recommendations to return
        ensure_diversity: Whether to enforce diversity in scent types
    
    Returns:
        List of recommended perfumes with 'ml_score' and 'ml_explanation' fields
    """
    # Check if user has enough perfumes
    if len(user_inventory) < ML_CONFIG['min_inventory_size']:
        return []
    
    # Train or load model
    model_data = train_ml_model(user_inventory, all_perfumes)
    if not model_data:
        # Try loading existing model
        model_data = load_ml_model()
        if not model_data:
            return []
    
    model, scaler = model_data[0], model_data[1]
    
    # Get IDs of owned perfumes
    owned_ids = {p['id'] for p in user_inventory}
    
    # Get candidate perfumes (non-owned)
    candidates = [p for p in all_perfumes if p['id'] not in owned_ids]
    
    if not candidates:
        return []
    
    # Score each candidate
    scored_perfumes = []
    for perfume in candidates:
        try:
            # Extract features
            features = extract_perfume_features(perfume)
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            # Get prediction probability
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(features_scaled)[0][1]
            else:
                prob = float(model.predict(features_scaled)[0])
            
            # Only include if above threshold
            if prob >= ML_CONFIG['min_recommendation_probability']:
                perfume_copy = perfume.copy()
                perfume_copy['ml_score'] = prob
                perfume_copy['ml_explanation'] = generate_ml_explanation(perfume, user_inventory, prob)
                scored_perfumes.append(perfume_copy)
        
        except Exception as e:
            print(f"Error scoring perfume {perfume.get('name', 'Unknown')}: {e}")
            continue
    
    # Sort by probability score
    scored_perfumes.sort(key=lambda p: p['ml_score'], reverse=True)
    
    # Apply diversity if requested
    if ensure_diversity and len(scored_perfumes) > top_n:
        scored_perfumes = apply_diversity_filter(scored_perfumes, top_n)
    
    return scored_perfumes[:top_n]

def apply_diversity_filter(perfumes: List[Dict], top_n: int) -> List[Dict]:
    """
    Ensure diversity in scent types among recommendations.
    
    Args:
        perfumes: List of scored perfumes
        top_n: Number of perfumes to return
    
    Returns:
        Diverse subset of perfumes
    """
    if not perfumes:
        return []
    
    selected = []
    scent_type_counts = {}
    
    # Try to balance scent types
    for perfume in perfumes:
        if len(selected) >= top_n:
            break
        
        scent_type = perfume.get('scent_type', 'Fresh')
        current_count = scent_type_counts.get(scent_type, 0)
        
        # Allow if this scent type is underrepresented or we're still filling
        max_per_type = max(2, top_n // 5)  # At least 2, or top_n/5
        if current_count < max_per_type or len(selected) < top_n // 2:
            selected.append(perfume)
            scent_type_counts[scent_type] = current_count + 1
    
    # Fill remaining slots with highest scores if needed
    if len(selected) < top_n:
        remaining = [p for p in perfumes if p not in selected]
        selected.extend(remaining[:top_n - len(selected)])
    
    return selected

def generate_ml_explanation(perfume: Dict, user_inventory: List[Dict], score: float) -> str:
    """
    Generate human-readable explanation for why a perfume was recommended.
    
    Args:
        perfume: Recommended perfume
        user_inventory: User's owned perfumes
        score: ML probability score
    
    Returns:
        Explanation string
    """
    # Analyze user's preferences from inventory
    all_user_accords = []
    for inv_perfume in user_inventory:
        all_user_accords.extend(inv_perfume.get('main_accords', []))
    
    # Get most common accords in user's collection
    accord_counter = Counter([a.lower() for a in all_user_accords if isinstance(a, str)])
    top_user_accords = [accord for accord, _ in accord_counter.most_common(3)]
    
    # Find matching accords in recommended perfume
    perfume_accords = [a.lower() for a in perfume.get('main_accords', []) if isinstance(a, str)]
    matching_accords = [a for a in perfume_accords if a in top_user_accords]
    
    # Build explanation
    if matching_accords:
        accord_text = ", ".join(matching_accords[:2])
        explanation = f"Matches your preference for {accord_text} scents"
    else:
        explanation = f"Complements your collection with {perfume_accords[0] if perfume_accords else 'unique'} notes"
    
    # Add confidence level
    if score >= 0.8:
        confidence = "Highly recommended"
    elif score >= 0.7:
        confidence = "Strong match"
    else:
        confidence = "Good match"
    
    return f"{confidence} - {explanation} ({int(score * 100)}% match)"

def get_model_insights(user_inventory: List[Dict]) -> Dict:
    """
    Get insights about the trained ML model and user preferences.
    
    Args:
        user_inventory: User's owned perfumes
    
    Returns:
        Dictionary with model insights and statistics
    """
    insights = {
        'inventory_size': len(user_inventory),
        'can_train_model': len(user_inventory) >= ML_CONFIG['min_inventory_size'],
        'model_type': ML_CONFIG['model_type'],
        'top_preferences': [],
        'diversity_score': 0.0
    }
    
    if not user_inventory:
        return insights
    
    # Analyze user preferences
    all_accords = []
    scent_types = []
    for perfume in user_inventory:
        all_accords.extend(perfume.get('main_accords', []))
        scent_types.append(perfume.get('scent_type', 'Fresh'))
    
    # Top preferences
    accord_counter = Counter([a.lower() for a in all_accords if isinstance(a, str)])
    insights['top_preferences'] = [
        {'name': accord.title(), 'count': count} 
        for accord, count in accord_counter.most_common(5)
    ]
    
    # Diversity score (unique scent types / total)
    if scent_types:
        insights['diversity_score'] = len(set(scent_types)) / len(scent_types)
    
    return insights

# ============================================================================
# FRAGELLA API INTEGRATION
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def call_fragella_api(endpoint: str, params: Optional[Dict] = None) -> Optional[List[Dict]]:
    """
    Make API calls to Fragella API with proper authentication.
    Cached for 1 hour to reduce API usage.
    
    Args:
        endpoint: API endpoint URL
        params: Optional query parameters
    
    Returns:
        List of fragrance objects from API or None if error occurs
    """
    # Set up headers with API key
    headers = {
        "x-api-key": FRAGELLA_API_KEY
    }
    
    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=15)
        
        # Check if request was successful
        response.raise_for_status()
        
        # The API returns an array of fragrance objects directly
        return response.json()
    
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def search_fragella_perfumes(query: str, limit: int = 20) -> List[Dict]:
    """
    Search for perfumes using Fragella API.
    Cached for 1 hour to prevent repeated API calls.
    
    Args:
        query: Search query string (minimum 3 characters)
        limit: Number of results to return (max 20)
    
    Returns:
        List of perfume results from API
    """
    # Ensure query is at least 3 characters
    if len(query) < 3:
        return []
    
    # Fragella API search endpoint
    endpoint = "https://api.fragella.com/api/v1/fragrances"
    
    # Set up parameters
    params = {
        "search": query,
        "limit": min(limit, 20)  # Max 20 per API docs
    }
    
    result = call_fragella_api(endpoint, params)
    
    if result:
        return result
    else:
        return []

def transform_api_perfume(api_perfume: Dict) -> Dict:
    """
    Transform Fragella API perfume object to our internal format.
    
    Args:
        api_perfume: Perfume object from Fragella API
    
    Returns:
        Transformed perfume dictionary
    """
    # Extract notes from the Notes object - including imageUrl from API
    notes_obj = api_perfume.get('Notes', {})
    top_notes = []
    heart_notes = []
    base_notes = []
    
    if notes_obj:
        # Top notes - store objects with name and imageUrl
        if 'Top' in notes_obj and notes_obj['Top']:
            top_notes = [{'name': note.get('name', ''), 'imageUrl': note.get('imageUrl', '')} 
                        for note in notes_obj['Top'] if note.get('name')]
        
        # Middle/Heart notes - store objects with name and imageUrl
        if 'Middle' in notes_obj and notes_obj['Middle']:
            heart_notes = [{'name': note.get('name', ''), 'imageUrl': note.get('imageUrl', '')} 
                          for note in notes_obj['Middle'] if note.get('name')]
        
        # Base notes - store objects with name and imageUrl
        if 'Base' in notes_obj and notes_obj['Base']:
            base_notes = [{'name': note.get('name', ''), 'imageUrl': note.get('imageUrl', '')} 
                         for note in notes_obj['Base'] if note.get('name')]
    
    # Parse seasonality from Season Ranking - try multiple possible field names
    seasonality = {"Winter": 3, "Fall": 3, "Spring": 3, "Summer": 3}  # Default values
    season_ranking = api_perfume.get('Season Ranking', api_perfume.get('SeasonRanking', api_perfume.get('season_ranking', [])))
    
    if season_ranking and isinstance(season_ranking, list):
        for season_obj in season_ranking:
            if isinstance(season_obj, dict):
                season_name = season_obj.get('name', season_obj.get('season', '')).title()
                season_score = season_obj.get('score', season_obj.get('value', 3))
                
                # Match season names more flexibly
                if 'winter' in season_name.lower():
                    seasonality['Winter'] = max(1, min(5, round(float(season_score))))
                elif 'fall' in season_name.lower() or 'autumn' in season_name.lower():
                    seasonality['Fall'] = max(1, min(5, round(float(season_score))))
                elif 'spring' in season_name.lower():
                    seasonality['Spring'] = max(1, min(5, round(float(season_score))))
                elif 'summer' in season_name.lower():
                    seasonality['Summer'] = max(1, min(5, round(float(season_score))))
    
    # Parse occasion from Occasion Ranking - simplified to Day/Night only
    occasion = {"Day": 3, "Night": 3}  # Default values
    occasion_ranking = api_perfume.get('Occasion Ranking', api_perfume.get('OccasionRanking', api_perfume.get('occasion_ranking', [])))
    
    if occasion_ranking and isinstance(occasion_ranking, list):
        day_scores = []
        night_scores = []
        
        for occ_obj in occasion_ranking:
            if isinstance(occ_obj, dict):
                occ_name = occ_obj.get('name', occ_obj.get('occasion', '')).lower()
                occ_score = float(occ_obj.get('score', occ_obj.get('value', 3)))
                
                # Map API occasion names to Day or Night
                if any(word in occ_name for word in ['casual', 'daily', 'day', 'office', 'sport', 'work', 'business']):
                    day_scores.append(occ_score)
                elif any(word in occ_name for word in ['evening', 'night', 'date', 'romantic', 'party', 'formal', 'special']):
                    night_scores.append(occ_score)
        
        # Average the scores
        if day_scores:
            occasion['Day'] = max(1, min(5, round(sum(day_scores) / len(day_scores))))
        if night_scores:
            occasion['Night'] = max(1, min(5, round(sum(night_scores) / len(night_scores))))
    
    # Parse price - API returns price as string
    price_str = api_perfume.get('Price', '$100')
    try:
        # Remove currency symbols and convert to float
        price = float(price_str.replace('$', '').replace('€', '').replace(',', '').strip())
    except:
        price = 100.0  # Default price
    
    # Get size/quantity from OilType field or default to 50ml
    oil_type = api_perfume.get('OilType', '')
    # Extract size if present in format like "50ml" or "100 ml"
    size = "50ml"  # Default size
    if oil_type and 'ml' in oil_type.lower():
        size = oil_type
    elif 'eau de parfum' in api_perfume.get('Name', '').lower():
        size = "100ml"  # Standard EDP size
    elif 'eau de toilette' in api_perfume.get('Name', '').lower():
        size = "100ml"  # Standard EDT size
    
    # Get gender - normalize to our format
    gender = api_perfume.get('Gender', 'Unisex')
    if 'women' in gender.lower():
        gender = 'Female'
    elif 'men' in gender.lower():
        gender = 'Male'
    else:
        gender = 'Unisex'
    
    # Determine scent type from main accords
    main_accords = api_perfume.get('Main Accords', [])
    scent_type = "Fresh"  # Default
    if main_accords:
        first_accord = main_accords[0].lower() if isinstance(main_accords, list) else ""
        if 'floral' in first_accord:
            scent_type = "Floral"
        elif 'woody' in first_accord or 'wood' in first_accord:
            scent_type = "Woody"
        elif 'citrus' in first_accord:
            scent_type = "Citrus"
        elif 'oriental' in first_accord or 'spicy' in first_accord:
            scent_type = "Oriental"
        elif 'sweet' in first_accord or 'gourmand' in first_accord:
            scent_type = "Gourmand"
        elif 'green' in first_accord or 'herbal' in first_accord:
            scent_type = "Green"
        elif 'leather' in first_accord:
            scent_type = "Leather"
    
    # Create transformed perfume object
    transformed = {
        "id": f"api_{api_perfume.get('Name', '').replace(' ', '_').lower()}",
        "name": api_perfume.get('Name', 'Unknown'),
        "brand": api_perfume.get('Brand', 'Unknown'),
        "price": price,
        "size": size,
        "gender": gender,
        "scent_type": scent_type,
        "description": f"A {api_perfume.get('Longevity', 'moderate')} fragrance with {api_perfume.get('Sillage', 'moderate')} projection.",
        "image_url": api_perfume.get('Image URL', 'https://via.placeholder.com/300x400/c8b8d8/FFFFFF?text=Perfume'),
        "top_notes": top_notes if top_notes else [{"name": "Bergamot", "imageUrl": ""}, {"name": "Lemon", "imageUrl": ""}],
        "heart_notes": heart_notes if heart_notes else [{"name": "Jasmine", "imageUrl": ""}, {"name": "Rose", "imageUrl": ""}],
        "base_notes": base_notes if base_notes else [{"name": "Musk", "imageUrl": ""}, {"name": "Vanilla", "imageUrl": ""}],
        "main_accords": main_accords if main_accords else ["Fresh", "Floral"],  # ALL accords, not limited
        "seasonality": seasonality,
        "occasion": occasion,
        "longevity": api_perfume.get('Longevity', 'Moderate'),
        "sillage": api_perfume.get('Sillage', 'Moderate')
    }
    
    return transformed

@st.cache_data(ttl=7200, show_spinner="Loading perfume database...")
def get_initial_perfumes() -> List[Dict]:
    """
    Get initial set of perfumes from Fragella API.
    Searches for popular brands and scent terms to populate the database with diverse results.
    Cached for 2 hours to prevent repeated API calls.
    
    Returns:
        List of perfumes from API
    """
    perfumes = []
    
    # Expanded list of popular perfume brands and scent terms for diverse initial database
    search_terms = [
        # Luxury brands
        "Dior", "Chanel", "Gucci", "Versace", "Tom Ford",
        "Prada", "Armani", "Yves Saint Laurent", "Givenchy", "Burberry",
        "Dolce Gabbana", "Calvin Klein", "Hugo Boss", "Valentino", "Hermes",
        # Popular scent families for variety
        "Rose", "Oud", "Vanilla", "Lavender", "Jasmine",
        "Citrus", "Sandalwood", "Amber", "Musk", "Bergamot"
    ]
    
    # Load perfumes from each search term
    for term in search_terms:
        results = search_fragella_perfumes(term, limit=20)
        if results:
            for api_perfume in results:
                transformed = transform_api_perfume(api_perfume)
                # Avoid duplicates by checking ID
                if not any(p['id'] == transformed['id'] for p in perfumes):
                    perfumes.append(transformed)
        
        # Continue loading to build a comprehensive database
        # Target: 200-300 perfumes for good variety
        if len(perfumes) >= 300:
            break
    
    return perfumes

# Note: All perfume data now comes from Fragella API
# No hardcoded sample perfumes

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """
    Initialize all session state variables for the application.
    Session state maintains data across reruns.
    """
    # Current active section/page
    if 'active_section' not in st.session_state:
        st.session_state.active_section = "home"
    
    # Search filters and query
    if 'selected_filters' not in st.session_state:
        st.session_state.selected_filters = {}
    
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    
    # Price range filter
    if 'price_range' not in st.session_state:
        st.session_state.price_range = (0, 200)
    
    # Questionnaire data
    if 'questionnaire_answers' not in st.session_state:
        st.session_state.questionnaire_answers = {}
    
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    
    if 'show_questionnaire_results' not in st.session_state:
        st.session_state.show_questionnaire_results = False
    
    # Perfume database - load from API on first run
    if 'perfume_database' not in st.session_state:
        with st.spinner("Loading perfumes from Fragella API..."):
            st.session_state.perfume_database = get_initial_perfumes()
            if not st.session_state.perfume_database:
                st.warning("Could not load perfumes from API. Please check your internet connection and API key.")
    
    # User's personal inventory
    if 'user_inventory' not in st.session_state:
        st.session_state.user_inventory = load_user_inventory()
    
    # Currently viewed perfume
    if 'current_perfume' not in st.session_state:
        st.session_state.current_perfume = None
    
    # Flag for showing perfume details
    if 'show_perfume_details' not in st.session_state:
        st.session_state.show_perfume_details = False
    
    # Track where detail view was opened from (search, questionnaire, inventory)
    if 'detail_view_source' not in st.session_state:
        st.session_state.detail_view_source = 'search'
    
    # Search context for similar recommendations
    if 'search_context' not in st.session_state:
        st.session_state.search_context = {}
    
    # ML recommendations cache
    if 'ml_recommendations' not in st.session_state:
        st.session_state.ml_recommendations = []
    
    # Flag for adding perfumes to inventory
    if 'adding_perfume' not in st.session_state:
        st.session_state.adding_perfume = False

# ============================================================================
# LOGO CONFIGURATION
# ============================================================================
# 
# HOW TO USE YOUR OWN LOGO:
# -------------------------
# 1. Save your logo PNG file in the same folder as this Python file
# 2. Update LOGO_IMAGE_PATH with your logo filename (e.g., "my_logo.png")
# 3. Change LOGO_TYPE from "text" to "image"
# 4. Adjust LOGO_HEIGHT to resize your logo (default: 80 pixels)
# 5. Customize LOGO_TAGLINE to change the text below the logo
#
# EXAMPLE - Using a PNG logo:
#   LOGO_TYPE = "image"
#   LOGO_IMAGE_PATH = "scentify_logo.png"
#   LOGO_HEIGHT = 100
#   LOGO_TAGLINE = "Your Custom Tagline"
#
# EXAMPLE - Using text logo (default):
#   LOGO_TYPE = "text"
#   LOGO_TEXT = "SCENTIFY"
#   LOGO_TAGLINE = "Find Your Signature Scent"
#

LOGO_TYPE = "text"  # Options: "text" or "image"
LOGO_IMAGE_PATH = "logo.png"  # Path to your logo PNG file (if using image)
LOGO_TEXT = "SCENTIFY"  # Text to display (if using text)
LOGO_HEIGHT = 80  # Height in pixels for image logo
LOGO_TAGLINE = "Find Your Signature Scent"  # Tagline displayed below logo

# ============================================================================
# HEADER AND NAVIGATION
# ============================================================================

def render_header():
    """
    Render the header with clickable logo (text or image) centered.
    """
    # Create centered logo and tagline
    if LOGO_TYPE == "image":
        # Check if logo file exists
        if os.path.exists(LOGO_IMAGE_PATH):
            # Use image logo
            logo_html = f"""
            <div style="position: relative; margin-top: -45px;">
                <div class="logo-container" id="logo-home-link">
                    <img src="data:image/png;base64,{get_image_base64(LOGO_IMAGE_PATH)}" class="logo-image" style="max-height: {LOGO_HEIGHT}px;">
                </div>
                <div class="logo-tagline">{LOGO_TAGLINE}</div>
            </div>
            """
        else:
            # Fallback to text if image not found
            st.warning(f"Logo image not found at '{LOGO_IMAGE_PATH}'. Using text logo instead.")
            logo_html = f"""
            <div style="position: relative; margin-top: -45px;">
                <div class="logo-container" id="logo-home-link">
                    <div class="logo-text">{LOGO_TEXT}</div>
                </div>
                <div class="logo-tagline">{LOGO_TAGLINE}</div>
            </div>
            """
    else:
        # Use text logo
        logo_html = f"""
        <div style="position: relative; margin-top: -45px;">
            <div class="logo-container" id="logo-home-link">
                <div class="logo-text">{LOGO_TEXT}</div>
            </div>
            <div class="logo-tagline">{LOGO_TAGLINE}</div>
        </div>
        """
    
    # Display centered logo with tagline
    st.markdown(logo_html, unsafe_allow_html=True)
    
    st.markdown("---")

def get_image_base64(image_path: str) -> str:
    """
    Convert image file to base64 string for embedding in HTML.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Base64 encoded string of the image
    """
    import base64
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def render_back_button(target_section: str, label: str = "Back"):
    """
    Render a back button to navigate to a different section.
    
    Args:
        target_section: Section to navigate to when clicked
        label: Button label text
    """
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if st.button(f"← {label}", key=f"back_to_{target_section}"):
            st.session_state.active_section = target_section
            st.session_state.show_perfume_details = False
            st.session_state.current_perfume = None
            st.rerun()

# ============================================================================
# LANDING PAGE - THREE MAIN SECTIONS
# ============================================================================

def scroll_to_top():
    """
    Scroll to top of page using JavaScript.
    """
    st.markdown("""
        <script>
            window.parent.document.querySelector('section.main').scrollTo(0, 0);
        </script>
    """, unsafe_allow_html=True)

def render_landing_page():
    """
    Render the landing page with three main section cards:
    1. Search
    2. Questionnaire
    3. Perfume Inventory (with AI recommendations inside)
    """
    scroll_to_top()
    
    # Check if showing perfume details from User Favorites
    if st.session_state.show_perfume_details and st.session_state.current_perfume:
        render_perfume_detail_view(st.session_state.current_perfume)
        return
    
    # Create three equal columns for the sections
    col1, col2, col3 = st.columns(3)
    
    # SECTION 1: SEARCH
    with col1:
        st.markdown("""
            <div class="section-card">
                <div class="section-title">Search</div>
                <div class="section-description">
                    Find your perfect perfume using our advanced filter system. 
                    Search by brand, price, scent type, and more to discover fragrances tailored to your preferences.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("Open Search", key="btn_search", use_container_width=True):
            st.session_state.active_section = "search"
            st.rerun()
    
    # SECTION 2: QUESTIONNAIRE
    with col2:
        st.markdown("""
            <div class="section-card">
                <div class="section-title">Questionnaire</div>
                <div class="section-description">
                    Not sure what you're looking for? Take our personalized questionnaire 
                    to discover perfumes that match your unique scent profile and preferences.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("Take Questionnaire", key="btn_questionnaire", use_container_width=True):
            st.session_state.active_section = "questionnaire"
            st.session_state.current_question = 0
            st.session_state.show_questionnaire_results = False
            st.rerun()
    
    # SECTION 3: PERFUME INVENTORY
    with col3:
        st.markdown("""
            <div class="section-card">
                <div class="section-title">Perfume Inventory</div>
                <div class="section-description">
                    Manage your personal perfume collection. Add fragrances you own and 
                    view detailed analytics and AI-powered recommendations based on your collection.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("View My Perfumes", key="btn_inventory", use_container_width=True):
            st.session_state.active_section = "inventory"
            st.rerun()
    
    # USER FAVORITES - Top 3 Most Popular Perfumes
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<h2 style="color: #6b5b95; text-align: center;">User Favorites</h2>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    render_user_favorites()

def render_user_favorites():
    """
    Display top 3 most popular perfumes based on clicks.
    """
    # Get rankings
    rankings = load_perfume_rankings()
    
    if not rankings:
        st.info("No user favorites yet. Start exploring perfumes!")
        return
    
    # Sort by clicks and get top 3
    top_perfumes_ids = sorted(rankings.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Get perfume details for top 3
    all_perfumes = st.session_state.perfume_database
    top_perfumes = []
    for perfume_id, clicks in top_perfumes_ids:
        perfume = next((p for p in all_perfumes if p.get('id') == perfume_id or p.get('name') == perfume_id), None)
        if perfume:
            perfume_copy = perfume.copy()
            perfume_copy['clicks'] = clicks
            top_perfumes.append(perfume_copy)
    
    if not top_perfumes:
        st.info("No user favorites yet. Start exploring perfumes!")
        return
    
    # Display in columns
    cols = st.columns(3)
    
    rank_colors = ['#FFD700', '#C0C0C0', '#CD7F32']  # Gold, Silver, Bronze
    rank_labels = ['1', '2', '3']
    
    for idx, (col, perfume) in enumerate(zip(cols, top_perfumes)):
        with col:
            # Rank badge and card styling
            rank_color = rank_colors[idx] if idx < len(rank_colors) else '#6b5b95'
            rank_label = rank_labels[idx] if idx < len(rank_labels) else str(idx + 1)
            
            st.markdown(f"""
                <div style="position: relative; border: 3px solid {rank_color}; border-radius: 15px; padding: 20px; background: white; height: 500px;">
                    <div style="position: absolute; top: -25px; left: 50%; transform: translateX(-50%); background: {rank_color}; color: white; 
                                width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                                font-size: 24px; font-weight: bold; box-shadow: 0 4px 6px rgba(0,0,0,0.2);">
                        {rank_label}
                    </div>
                    <div style="text-align: right; margin-top: 10px;">
                        <span style="background: #e8d5f0; color: #6b5b95; padding: 5px 10px; border-radius: 5px; font-size: 12px; font-weight: 600;">
                            {perfume.get('gender', 'UNISEX').upper()}
                        </span>
                    </div>
                    <div style="text-align: center; margin: 20px 0;">
                        <img src="{perfume.get('image_url', '')}" style="max-width: 150px; max-height: 150px; object-fit: contain;">
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 18px; font-weight: 600; color: #333; margin: 10px 0;">
                            {perfume.get('name', 'Unknown')}
                        </div>
                        <div style="font-size: 14px; font-style: italic; color: #6b5b95; margin: 5px 0;">
                            by {perfume.get('brand', 'Unknown')}
                        </div>
                        <div style="font-size: 14px; color: #666; margin-top: 10px;">
                            {perfume.get('clicks', 0)} interactions
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("View Details", key=f"fav_view_{perfume.get('id', idx)}", use_container_width=True):
                record_interaction(perfume['id'], 'click')
                st.session_state.current_perfume = perfume
                st.session_state.show_perfume_details = True
                st.session_state.detail_view_source = 'home'
                st.rerun()

# ============================================================================
# SEARCH SECTION
# ============================================================================

def render_search_section():
    """
    Render the complete search section with filters and results.
    """
    scroll_to_top()
    
    # Check if showing perfume details
    if st.session_state.show_perfume_details and st.session_state.current_perfume:
        render_perfume_detail_view(st.session_state.current_perfume)
        return
    
    # Show back button
    render_back_button("home", "Back to Home")
    
    st.markdown('<h2 style="color: #6b5b95;">Search for Perfumes</h2>', unsafe_allow_html=True)
    
    # SEARCH BAR - for brand or name
    st.markdown('<h3 style="color: #6b5b95;">Search for brand / name</h3>', unsafe_allow_html=True)
    search_query = st.text_input(
        "Enter brand or perfume name...",
        value=st.session_state.search_query,
        placeholder="Enter brand or perfume name...",
        key="search_input",
        label_visibility="collapsed"
    )
    st.session_state.search_query = search_query
    
    st.markdown("---")
    
    # FILTERS
    st.markdown('<h3 style="color: #6b5b95;">Filters</h3>', unsafe_allow_html=True)
    
    # Comprehensive list of ALL perfume brands (A-Z) - no duplicates, over 100 brands
    all_brands = sorted(list(set([
        "3Lab", "4711", "7 Virtues", "8 Muse",
        "Abercrombie & Fitch", "Acqua di Parma", "Adidas", "Aerin", "Agent Provocateur", "Aigner", 
        "Ajmal", "Al Haramain", "Alaia", "Alexander McQueen", "Alfred Sung", "Amouage", 
        "Annick Goutal", "Anna Sui", "Aramis", "Ariana Grande", "Armaf", "Armani", "Atelier Cologne",
        "Azzaro", "Balenciaga", "Balmain", "Banana Republic", "Benefit", "Bentley", "Boucheron",
        "Bond No. 9", "Bottega Veneta", "Brecourt", "Britney Spears", "Bulgari", "Burberry", "Bvlgari", "Byredo",
        "Cacharel", "Calvin Klein", "Carolina Herrera", "Cartier", "Carven", "Celine", "Cerruti",
        "Chanel", "Chloe", "Chopard", "Christian Dior", "Christian Louboutin", "Clarins", "Clinique",
        "Coach", "Commodity", "Creed", "Curve", "Davidoff", "Diesel", "Diptyque", "DKNY", 
        "Dolce & Gabbana", "Donna Karan", "Dunhill", "Elizabeth Arden", "Elizabeth Taylor", 
        "Emporio Armani", "Escada", "Estee Lauder", "Eternity", "Etat Libre d'Orange",
        "Fendi", "Ferragamo", "Frederic Malle", "Gianni Versace", "Giorgio Armani", "Giorgio Beverly Hills",
        "Givenchy", "Gucci", "Guerlain", "Guess", "Halston", "Hermes", "Histoires de Parfums",
        "Hugo Boss", "Issey Miyake", "Jacquemus", "James Bond", "Jean Paul Gaultier", "Jennifer Aniston",
        "Jennifer Lopez", "Jessica Simpson", "Jimmy Choo", "Jo Malone", "John Varvatos", "Joop",
        "Juicy Couture", "Juliette Has a Gun", "Karl Lagerfeld", "Kate Spade", "Kenzo", "Kilian",
        "Lacoste", "Lalique", "Lancome", "Lanvin", "Laura Mercier", "Le Labo", "Loewe", "Lolita Lempicka",
        "Maison Francis Kurkdjian", "Maison Margiela", "Marc Jacobs", "Memo Paris", "Michael Kors", 
        "Missoni", "Miu Miu", "Molton Brown", "Montblanc", "Moschino", "Mugler", "Narciso Rodriguez",
        "Nasomatto", "Nautica", "Nest", "Nina Ricci", "Nishane", "Olivier Durbano",
        "Paco Rabanne", "Penhaligon's", "Philosophy", "Prada", "Ralph Lauren", "Rihanna", "Roberto Cavalli",
        "Rochas", "Salvatore Ferragamo", "Sarah Jessica Parker", "Serge Lutens", "Shiseido", "Sì",
        "Stella McCartney", "Thierry Mugler", "Tiffany & Co", "Tom Ford", "Tommy Hilfiger", "Tory Burch",
        "Trussardi", "Valentino", "Van Cleef & Arpels", "Vera Wang", "Versace", "Viktor & Rolf",
        "Vilhelm Parfumerie", "Xerjoff", "Yves Saint Laurent", "Zadig & Voltaire", "Zara"
    ] + [p['brand'] for p in st.session_state.perfume_database])))
    
    # Create filter layout - 2 filters only
    col1, col2 = st.columns(2)
    
    # SCENT TYPE FILTER - First filter
    with col1:
        with st.expander("Scent Type", expanded=False):
            scent_types = ["Floral", "Woody", "Fresh", "Citrus", "Oriental", "Gourmand", "Green", "Leather"]
            selected_scents = st.multiselect(
                "Select scent types",
                options=scent_types,
                default=st.session_state.selected_filters.get('scent_type', []),
                key="filter_scent"
            )
            if st.button("Save", key="save_scent"):
                if selected_scents:
                    st.session_state.selected_filters['scent_type'] = selected_scents
                elif 'scent_type' in st.session_state.selected_filters:
                    del st.session_state.selected_filters['scent_type']
                st.success("Scent type filter saved")
    
    # FOR WHOM FILTER - Second filter
    with col2:
        with st.expander("For Whom", expanded=False):
            selected_gender = st.multiselect(
                "Select gender",
                options=["Male", "Female", "Unisex"],
                default=st.session_state.selected_filters.get('gender', []),
                key="filter_gender"
            )
            if st.button("Save", key="save_gender"):
                if selected_gender:
                    st.session_state.selected_filters['gender'] = selected_gender
                elif 'gender' in st.session_state.selected_filters:
                    del st.session_state.selected_filters['gender']
                st.success("Gender filter saved")
    
    st.markdown("---")
    
    # DISPLAY SELECTED FILTERS AS TAGS
    if st.session_state.selected_filters:
        st.markdown('<h4 style="color: #6b5b95;">Selected Filters</h4>', unsafe_allow_html=True)
        render_filter_tags()
        st.markdown("---")
    
    # RESET AND SEARCH BUTTONS
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("Reset Filters", key="reset_filters", use_container_width=True):
            st.session_state.selected_filters = {}
            st.session_state.price_range = (0, 200)
            st.session_state.search_query = ""
            st.rerun()
    
    with col_btn2:
        if st.button("Search", key="do_search", use_container_width=True):
            st.session_state.search_context = st.session_state.selected_filters.copy()
            st.rerun()
    
        st.markdown("---")
    
    # SEARCH RESULTS
    st.markdown('<h3 style="color: #6b5b95;">Search Results</h3>', unsafe_allow_html=True)
    display_search_results()

def render_filter_tags():
    """
    Display selected filters as tags with X buttons to remove them.
    """
    tags_html = '<div style="margin-bottom: 15px;">'
    
    for filter_name, values in st.session_state.selected_filters.items():
        if filter_name == 'price':
            tags_html += f'<span class="filter-tag">Price: ${values[0]} - ${values[1]}</span>'
        elif isinstance(values, list):
            for value in values:
                tags_html += f'<span class="filter-tag">{filter_name.title()}: {value}</span>'
    
    tags_html += '</div>'
    st.markdown(tags_html, unsafe_allow_html=True)
    
    # Remove tag buttons
    cols = st.columns(len(st.session_state.selected_filters) + 1)
    idx = 0
    filters_to_remove = []
    
    for filter_name in list(st.session_state.selected_filters.keys()):
        with cols[idx]:
            if st.button(f"Remove {filter_name}", key=f"remove_{filter_name}"):
                filters_to_remove.append(filter_name)
        idx += 1
    
    for filter_name in filters_to_remove:
        del st.session_state.selected_filters[filter_name]
    
    if filters_to_remove:
                        st.rerun()
                
def filter_perfumes(perfumes: List[Dict]) -> List[Dict]:
    """
    Filter perfumes based on search query and selected filters.
    
    Args:
        perfumes: List of all perfumes
    
    Returns:
        Filtered list of perfumes
    """
    filtered = perfumes.copy()
    
    # Filter by search query (handles brand or name search from main search bar)
    if st.session_state.search_query:
        query = st.session_state.search_query.lower()
        filtered = [p for p in filtered if query in p['name'].lower() or query in p['brand'].lower()]
    
    # Filter by gender
    if 'gender' in st.session_state.selected_filters:
        genders = st.session_state.selected_filters['gender']
        filtered = [p for p in filtered if p['gender'] in genders]
    
    # Filter by scent type
    if 'scent_type' in st.session_state.selected_filters:
        scents = st.session_state.selected_filters['scent_type']
        filtered = [p for p in filtered if p['scent_type'] in scents]
    
    return filtered

def display_search_results():
    """
    Display filtered perfume results.
    Shows results when user searches OR applies filters OR clicks Search button.
    Performs live API search if search query is provided.
    """
    # Check if user has searched or applied filters or clicked search button
    has_search_query = st.session_state.search_query and len(st.session_state.search_query) >= 3
    has_filters = bool(st.session_state.selected_filters)
    force_show = st.session_state.get('force_show_results', False)
    
    # If there's a search query with 3+ characters, do live API search
    if has_search_query:
        with st.spinner("Searching Fragella database..."):
            api_results = search_fragella_perfumes(st.session_state.search_query, limit=20)
            if api_results:
                # Transform API results
                transformed_results = [transform_api_perfume(p) for p in api_results]
                # Update database with new perfumes (avoid duplicates)
                for perfume in transformed_results:
                    if not any(p['id'] == perfume['id'] for p in st.session_state.perfume_database):
                        st.session_state.perfume_database.append(perfume)
                # Apply any filters to the search results
                if has_filters:
                    filtered_perfumes = filter_perfumes(transformed_results)
                else:
                    filtered_perfumes = transformed_results
            else:
                filtered_perfumes = []
    elif has_filters or force_show:
        # Use existing database with filters only
        filtered_perfumes = filter_perfumes(st.session_state.perfume_database)
        # Reset force_show flag
        if force_show:
            st.session_state.force_show_results = False
    else:
        # No search and no filters - show message
        st.info("Enter a perfume name or apply filters to see results")
        return
    
    # Apply ML ranking to sort perfumes by popularity (based on clicks only)
    sorted_perfumes = get_ml_sorted_perfumes(filtered_perfumes)
    
    st.write(f"Found {len(sorted_perfumes)} perfume(s)")
    
    if sorted_perfumes:
        # Display in grid (3 columns for cleaner layout)
        for i in range(0, len(sorted_perfumes), 3):
            col1, col2, col3 = st.columns(3, gap="medium")
            
            with col1:
                if i < len(sorted_perfumes):
                    display_perfume_card(sorted_perfumes[i], show_ml_badge=True)
            
            with col2:
                if i + 1 < len(sorted_perfumes):
                    display_perfume_card(sorted_perfumes[i + 1], show_ml_badge=True)
            
            with col3:
                if i + 2 < len(sorted_perfumes):
                    display_perfume_card(sorted_perfumes[i + 2], show_ml_badge=True)
    else:
        st.info("No perfumes match your search criteria. Try different search terms or adjust your filters.")

def display_perfume_card(perfume: Dict, show_ml_badge: bool = False, source: str = 'search'):
    """
    Display a perfume card with key information including smaller image and clean formatting.
    PERFECTLY ALIGNED with fixed heights for all elements.
    
    Args:
        perfume: Perfume dictionary
        show_ml_badge: Whether to show ML ranking badge
        source: Where the card is displayed ('search', 'questionnaire', or 'current')
    """
    # Get ML ranking for badge
    rankings = load_perfume_rankings()
    rank_score = rankings.get(perfume['id'], 0)
    
    # Use perfume icon if no image available
    image_url = perfume.get('image_url', '')
    if not image_url or image_url == '':
        # SVG perfume bottle icon as fallback
        image_url = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjI1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjAwIiBoZWlnaHQ9IjI1MCIgZmlsbD0iI2U4ZTRmMCIvPjxwYXRoIGQ9Ik04MCA2MGgyMHY0MEg4MHoiIGZpbGw9IiM2YjViOTUiLz48cmVjdCB4PSI2MCIgeT0iMTAwIiB3aWR0aD0iNjAiIGhlaWdodD0iMTIwIiByeD0iMTAiIGZpbGw9IiM2YjViOTUiIG9wYWNpdHk9IjAuOCIvPjxyZWN0IHg9IjcwIiB5PSIxMTAiIHdpZHRoPSI0MCIgaGVpZ2h0PSI5MCIgZmlsbD0iI2M4YjhkOCIgb3BhY2l0eT0iMC42Ii8+PC9zdmc+'
    
    with st.container():
        # Get gender for badge
        gender = perfume.get('gender', 'UNISEX').upper()
        
        st.markdown(f"""
            <div class="perfume-card" style="padding: 20px; height: 500px; display: flex; flex-direction: column; position: relative;">
                <div style="position: absolute; top: 15px; right: 15px;">
                    <span style="background: #6b5b95; color: white; padding: 6px 14px; border-radius: 8px; font-size: 12px; font-weight: 600;">
                        {gender}
                    </span>
                </div>
                <div style="height: 220px; display: flex; align-items: center; justify-content: center; margin-bottom: 20px; margin-top: 10px;">
                    <img src="{image_url}" 
                         onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjI1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjAwIiBoZWlnaHQ9IjI1MCIgZmlsbD0iI2U4ZTRmMCIvPjxwYXRoIGQ9Ik04MCA2MGgyMHY0MEg4MHoiIGZpbGw9IiM2YjViOTUiLz48cmVjdCB4PSI2MCIgeT0iMTAwIiB3aWR0aD0iNjAiIGhlaWdodD0iMTIwIiByeD0iMTAiIGZpbGw9IiM2YjViOTUiIG9wYWNpdHk9IjAuOCIvPjxyZWN0IHg9IjcwIiB5PSIxMTAiIHdpZHRoPSI0MCIgaGVpZ2h0PSI5MCIgZmlsbD0iI2M4YjhkOCIgb3BhY2l0eT0iMC42Ii8+PC9zdmc+'"
                         style="max-width: 180px; max-height: 220px; object-fit: contain;">
                </div>
                <div style="text-align: center; margin-bottom: 8px;">
                    <h3 style="color: #333; margin: 0; font-size: 18px; font-weight: 600; line-height: 1.3;">
                        {perfume['name']}
                    </h3>
                </div>
                <div style="text-align: center; margin-bottom: 15px;">
                    <p style="color: #6b5b95; font-style: italic; font-size: 15px; margin: 0;">
                        by {perfume['brand']}
                    </p>
                </div>
                <hr style="border: none; border-top: 1px solid #e8e4f0; margin: 15px 0;">
                <div style="text-align: center; flex-grow: 1;">
                    <p style="font-size: 15px; color: #666; margin: 0 0 8px 0; font-weight: 600;">
                        Accords:
                    </p>
                    <p style="font-size: 14px; color: #666; margin: 0;">
                        {', '.join(perfume.get('main_accords', ['Fresh', 'Floral'])[:3])}
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Show ML badge if enabled and has interactions
        if show_ml_badge and rank_score > 0:
            st.markdown(f'<div style="text-align: center; height: 30px; margin-bottom: 5px;"><span style="background-color: #6b5b95; color: white; padding: 4px 12px; border-radius: 12px; font-size: 11px; display: inline-block;">Popular (Score: {rank_score})</span></div>', unsafe_allow_html=True)
        elif show_ml_badge:
            st.markdown('<div style="height: 30px; margin-bottom: 5px;"></div>', unsafe_allow_html=True)
        
        button_clicked = st.button("View Full Details", key=f"view_btn_{perfume['id']}", use_container_width=True, type="primary")
        
        if button_clicked:
            # Record interaction
            record_interaction(perfume['id'], 'click')
            
            # Clear any previous states
            if 'adding_perfume' in st.session_state:
                st.session_state.adding_perfume = False
            
            # Set current perfume and show details
            st.session_state.current_perfume = perfume.copy()
            st.session_state.show_perfume_details = True
            
            # Set source for back button (keep current source if viewing similar perfumes)
            if source != 'current':
                st.session_state.detail_view_source = source
            
            st.rerun()

# ============================================================================
# PERFUME DETAIL VIEW
# ============================================================================

def render_perfume_detail_view(perfume: Dict):
    """
    Render detailed view of a perfume with all information.
    
    Args:
        perfume: Perfume dictionary with full details
    """
    # Scroll to top when opening detail view
    st.markdown("""
        <script>
            window.parent.document.querySelector('section.main').scrollTo(0, 0);
        </script>
    """, unsafe_allow_html=True)
    
    # Back button - context aware based on where we came from
    if st.session_state.detail_view_source == 'home':
        # Coming from home/landing page User Favorites
        if st.button("← Back to Home", key="back_to_home"):
            st.session_state.show_perfume_details = False
            st.session_state.current_perfume = None
            st.session_state.detail_view_source = 'search'  # Reset to default
            st.rerun()
    elif st.session_state.detail_view_source == 'inventory':
        # Coming from inventory
        if st.button("← Back to Inventory", key="back_to_inventory"):
            st.session_state.show_perfume_details = False
            st.session_state.current_perfume = None
            st.session_state.detail_view_source = 'search'  # Reset to default
            st.rerun()
    elif st.session_state.show_questionnaire_results:
        # Coming from questionnaire results
        if st.button("← Back to Recommendations", key="back_to_questionnaire"):
            st.session_state.show_perfume_details = False
            st.session_state.current_perfume = None
            st.session_state.detail_view_source = 'search'  # Reset to default
            st.rerun()
    else:
        # Coming from search section
        render_back_button("search", "Back to Results")
    
    # Record view interaction
    record_interaction(perfume['id'], 'view')
    
    # Two-column layout: Left for image+accords, Right for details
    col_left, col_right = st.columns([1, 1.5], gap="large")
    
    # LEFT COLUMN - Picture at top, accords below
    with col_left:
        # Display perfume image - centered
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(perfume['image_url'], use_container_width=True)
        
        # Add to inventory button
        if st.button("Add to My Perfumes", key="add_to_inventory", use_container_width=True):
            add_to_user_inventory(perfume)
            record_interaction(perfume['id'], 'add_to_inventory')
            st.success(f"Added {perfume['name']} to your collection")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # MAIN ACCORDS below picture
        st.markdown('<h4 style="color: #6b5b95; margin: 15px 0; font-size: 18px; font-weight: 600;">main accords</h4>', unsafe_allow_html=True)
        
        # Define accord colors - COMPREHENSIVE list with thematic colors for each accord
        accord_colors = {
            # Animal & Musk
            'animalic': '#8B7355', 'musk': '#D3C5B5', 'musky': '#E6D5E6', 'castoreum': '#6B5B4B',
            # Floral
            'floral': '#E8A5D4', 'white floral': '#F5F5F5', 'rose': '#FF69B4', 'jasmine': '#FFF8DC',
            'ylang ylang': '#FFE4B5', 'tuberose': '#FADADD', 'iris': '#9370DB', 'violet': '#8A2BE2',
            'lavender': '#B19CD9', 'orange blossom': '#FFE4B2', 'lily': '#FFFACD',
            # Fresh & Green
            'fresh': '#A8E6CF', 'green': '#90EE90', 'herbal': '#8FBC8F', 'aromatic': '#9370DB',
            'aquatic': '#7FCDCD', 'marine': '#5F9EA0', 'ozonic': '#B0E0E6', 'watery': '#ADD8E6',
            # Citrus
            'citrus': '#FFD93D', 'lemon': '#FFF44F', 'bergamot': '#F4D03F', 'orange': '#FFA500',
            'mandarin': '#FF8C00', 'grapefruit': '#FFB6C1', 'lime': '#BFFF00',
            # Woody
            'woody': '#8B6F47', 'cedar': '#A0826D', 'sandalwood': '#C19A6B', 'patchouli': '#6B5B4B',
            'vetiver': '#7C7356', 'oud': '#4A3728', 'agarwood': '#3E2723', 'pine': '#4A7856', 'cypress': '#5F7356',
            # Spicy & Warm
            'spicy': '#DC143C', 'warm spicy': '#D97548', 'cinnamon': '#B87333', 'clove': '#8B4513',
            'pepper': '#A9A9A9', 'pink pepper': '#F4A7B9', 'cardamom': '#E6BE8A', 'nutmeg': '#8B7355', 'ginger': '#DAA520',
            # Sweet & Gourmand
            'sweet': '#E85D75', 'gourmand': '#DDA15E', 'vanilla': '#F3E5AB', 'caramel': '#D2691E',
            'chocolate': '#7B3F00', 'honey': '#F4C542', 'tonka bean': '#D2B48C', 'almond': '#FFEBCD',
            'coconut': '#FFFFF0', 'coffee': '#6F4E37',
            # Fruity
            'fruity': '#FF6B6B', 'tropical': '#F4D03F', 'berry': '#8B008B', 'peach': '#FFDAB9',
            'apple': '#90EE90', 'pear': '#D1E231', 'plum': '#8E4585', 'cherry': '#DE3163', 'blackcurrant': '#2E0854',
            # Earthy & Mossy
            'earthy': '#8B8B7A', 'mossy': '#8A9A5B', 'oakmoss': '#6B8E23', 'peat': '#4A4A3A',
            # Oriental & Resinous
            'oriental': '#B8860B', 'amber': '#FFBF00', 'incense': '#8B7D6B', 'myrrh': '#8B7355',
            'benzoin': '#D2B48C', 'labdanum': '#8B7765', 'resinous': '#A0826D',
            # Leather & Tobacco
            'leather': '#654321', 'tobacco': '#7C5936', 'suede': '#8B7E66', 'birch tar': '#4A4A4A',
            # Powdery & Soft
            'powdery': '#E6C9E3', 'soft': '#F5E6E8', 'aldehydic': '#F0F0F0',
            # Balsamic
            'balsamic': '#8B7355', 'peru balsam': '#8B6914',
            # Lactonic & Creamy
            'lactonic': '#FFF8DC', 'creamy': '#FFFACD', 'milky': '#FFFEF0',
            # Fresh Spicy
            'fresh spicy': '#FF8C69',
            # Soapy & Clean
            'soapy': '#E0FFFF', 'clean': '#F0FFFF',
            # Coniferous
            'coniferous': '#228B22'
        }
        
        # Get ALL main accords
        main_accords = perfume.get('main_accords', ['Fresh', 'Floral', 'Sweet'])
        
        # Assign intensity values
        if len(main_accords) > 0:
            decrement = min(7, 70 / len(main_accords))
            intensities = [max(100 - (i * decrement), 30) for i in range(len(main_accords))]
        else:
            intensities = []
        
        # Create horizontal bar chart data
        accord_data = []
        for i, accord in enumerate(main_accords):
            accord_lower = accord.lower().strip()
            color = accord_colors.get(accord_lower, '#6b5b95')
            accord_data.append({
                'Accord': accord_lower,
                'Intensity': intensities[i],
                'Color': color
            })
        
        # Display as horizontal bars using plotly
        if accord_data:
            fig_accords = go.Figure()
            
            for item in accord_data:
                fig_accords.add_trace(go.Bar(
                    y=[item['Accord']],
                    x=[item['Intensity']],
                    orientation='h',
                    marker=dict(
                        color=item['Color'],
                        line=dict(width=0),
                        cornerradius=8
                    ),
                    name=item['Accord'],
                    showlegend=False,
                    text=item['Accord'],
                    textposition='inside',
                    insidetextanchor='middle',
                    textfont=dict(color='white', size=14, family='Arial, sans-serif', weight='bold'),
                    hoverinfo='skip',
                    width=0.75
                ))
            
            # Calculate dynamic height based on number of accords
            chart_height = max(300, len(main_accords) * 50)
            
            fig_accords.update_layout(
                height=chart_height,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#333', size=14, family='Arial, sans-serif'),
                xaxis=dict(
                    title='',
                    showgrid=False,
                    showticklabels=False,
                    range=[0, 100],
                    fixedrange=True,
                    zeroline=False
                ),
                yaxis=dict(
                    title='',
                    showticklabels=False,
                    fixedrange=True,
                    autorange='reversed',
                    zeroline=False
                ),
                margin=dict(l=0, r=0, t=5, b=5),
                bargap=0.15,
                hovermode=False,
                autosize=True
            )
            
            # Accord bars below picture
            st.plotly_chart(fig_accords, use_container_width=True, config={'displayModeBar': False})
    
    # RIGHT COLUMN - Details with bigger text and stacked Longevity/Sillage
    with col_right:
        # Perfume name - BIGGER
        st.markdown(f'''
            <h1 style="color: #6b5b95; margin: 0 0 8px 0; font-size: 36px; font-weight: 600;">
                {perfume["name"]}
            </h1>
        ''', unsafe_allow_html=True)
        
        # Brand - BIGGER
        st.markdown(f'''
            <p style="color: #6b5b95; font-style: italic; font-size: 18px; margin: 0 0 25px 0;">
                Brand: {perfume["brand"]}
            </p>
        ''', unsafe_allow_html=True)
        
        # Description - BIGGER
        st.markdown(f'''
            <p style="color: #333; margin: 0 0 15px 0; font-size: 16px; line-height: 1.7;">
                {perfume['description']}
            </p>
        ''', unsafe_allow_html=True)
        
        # LONGEVITY AND SILLAGE - Side by side with transparent background
        longevity = perfume.get('longevity', 'Moderate')
        sillage = perfume.get('sillage', 'Moderate')
        
        col_long, col_sil = st.columns(2)
        
        with col_long:
            # Longevity box - transparent
            st.markdown(f'''
                <div style="background: transparent; padding: 20px; border-radius: 12px; text-align: center; height: 90px; display: flex; flex-direction: column; justify-content: center;">
                    <h4 style="color: #6b5b95; margin: 0 0 8px 0; font-size: 20px; font-weight: 600;">Longevity</h4>
                    <p style="font-size: 22px; font-weight: bold; color: #333; margin: 0;">{longevity}</p>
                </div>
            ''', unsafe_allow_html=True)
        
        with col_sil:
            # Sillage box - transparent
            st.markdown(f'''
                <div style="background: transparent; padding: 20px; border-radius: 12px; text-align: center; height: 90px; display: flex; flex-direction: column; justify-content: center;">
                    <h4 style="color: #6b5b95; margin: 0 0 8px 0; font-size: 20px; font-weight: 600;">Sillage</h4>
                    <p style="font-size: 22px; font-weight: bold; color: #333; margin: 0;">{sillage}</p>
                </div>
            ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # PERFUME PYRAMID (Top, Heart, Base Notes)
    st.markdown('<h3 style="color: #6b5b95;">Perfume Pyramid</h3>', unsafe_allow_html=True)
    
    # Custom CSS for note cards
    st.markdown("""
        <style>
        .note-card {
            background: #f5f3f8;
            border-radius: 12px;
            padding: 12px 16px;
            margin: 8px 0;
            display: flex;
            align-items: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .note-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        .note-icon {
            width: 45px;
            height: 45px;
            border-radius: 10px;
            margin-right: 15px;
            object-fit: cover;
            background: #e8e4ee;
            padding: 8px;
        }
        .note-name {
            font-size: 15px;
            color: #333;
            font-weight: 500;
        }
        .note-category-title {
            font-size: 16px;
            font-weight: 600;
            color: #6b5b95;
            margin-bottom: 10px;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    col_top, col_heart, col_base = st.columns(3)
    
    with col_top:
        st.markdown('<div class="note-category-title">Top Notes</div>', unsafe_allow_html=True)
        for note in perfume['top_notes']:
            # Check if note is a dict (from API) or string (old format)
            if isinstance(note, dict):
                note_name = note.get('name', '')
                note_image = note.get('imageUrl', '')
                if note_image:
                    # Display card with image from API
                    st.markdown(f"""
                        <div class="note-card">
                            <img src="{note_image}" class="note-icon">
                            <span class="note-name">{note_name}</span>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    # Display card with empty icon box
                    st.markdown(f"""
                        <div class="note-card">
                            <div class="note-icon"></div>
                            <span class="note-name">{note_name}</span>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                # Old string format
                st.markdown(f"""
                    <div class="note-card">
                        <div class="note-icon"></div>
                        <span class="note-name">{note}</span>
                    </div>
                """, unsafe_allow_html=True)
    
    with col_heart:
        st.markdown('<div class="note-category-title">Heart Notes</div>', unsafe_allow_html=True)
        for note in perfume['heart_notes']:
            # Check if note is a dict (from API) or string (old format)
            if isinstance(note, dict):
                note_name = note.get('name', '')
                note_image = note.get('imageUrl', '')
                if note_image:
                    # Display card with image from API
                    st.markdown(f"""
                        <div class="note-card">
                            <img src="{note_image}" class="note-icon">
                            <span class="note-name">{note_name}</span>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    # Display card with empty icon box
                    st.markdown(f"""
                        <div class="note-card">
                            <div class="note-icon"></div>
                            <span class="note-name">{note_name}</span>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                # Old string format
                st.markdown(f"""
                    <div class="note-card">
                        <div class="note-icon"></div>
                        <span class="note-name">{note}</span>
                    </div>
                """, unsafe_allow_html=True)
    
    with col_base:
        st.markdown('<div class="note-category-title">Base Notes</div>', unsafe_allow_html=True)
        for note in perfume['base_notes']:
            # Check if note is a dict (from API) or string (old format)
            if isinstance(note, dict):
                note_name = note.get('name', '')
                note_image = note.get('imageUrl', '')
                if note_image:
                    # Display card with image from API
                    st.markdown(f"""
                        <div class="note-card">
                            <img src="{note_image}" class="note-icon">
                            <span class="note-name">{note_name}</span>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    # Display card with empty icon box
                    st.markdown(f"""
                        <div class="note-card">
                            <div class="note-icon"></div>
                            <span class="note-name">{note_name}</span>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                # Old string format
                st.markdown(f"""
                    <div class="note-card">
                        <div class="note-icon"></div>
                        <span class="note-name">{note}</span>
                    </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # SEASONALITY - Horizontal bar chart with rounded corners
    st.markdown('<h3 style="color: #6b5b95;">Seasonality</h3>', unsafe_allow_html=True)
    
    # Get seasonality data with proper structure
    seasonality = perfume.get('seasonality', {'Winter': 3, 'Spring': 3, 'Summer': 3, 'Fall': 3})
    
    # Create horizontal bar chart
    seasons = ['Summer', 'Spring', 'Winter', 'Fall']
    values = [seasonality.get(season, 3) for season in seasons]
    
    fig_season = go.Figure()
    
    for i, season in enumerate(seasons):
        fig_season.add_trace(go.Bar(
            y=[season],
            x=[values[i]],
            orientation='h',
            marker=dict(
                color='#8b7aa8',  # Purple to match website theme
                line=dict(width=0)
            ),
            showlegend=False,
            text='',
            hoverinfo='skip'
        ))
    
    fig_season.update_layout(
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333', size=15, family='Arial, sans-serif'),
        xaxis=dict(
            title='',
            range=[0, 5.5],
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        yaxis=dict(
            title='',
            tickfont=dict(size=14),
            showgrid=False
        ),
        margin=dict(l=100, r=40, t=20, b=40),
        bargap=0.3,
        bargroupgap=0.1
    )
    
    st.plotly_chart(fig_season, use_container_width=True)
    
    # OCCASION - Full width stacked bar (NO percentages)
    st.markdown('<h3 style="color: #6b5b95;">Occasion</h3>', unsafe_allow_html=True)
    
    # Get occasion data with proper structure
    occasion = perfume.get('occasion', {'Day': 3, 'Night': 3})
    
    # Calculate percentages
    total = occasion.get('Day', 3) + occasion.get('Night', 3)
    day_percent = (occasion.get('Day', 3) / total * 100) if total > 0 else 50
    night_percent = (occasion.get('Night', 3) / total * 100) if total > 0 else 50
    
    # Create horizontal stacked bar chart
    fig_occasion = go.Figure()
    
    # Day portion (left, lighter purple)
    fig_occasion.add_trace(go.Bar(
        y=[''],
        x=[day_percent],
        orientation='h',
        marker=dict(color='#9b8bb5'),  # Lighter purple
        showlegend=False,
        text='',
        hoverinfo='skip',
        name='Day'
    ))
    
    # Night portion (right, dark purple)
    fig_occasion.add_trace(go.Bar(
        y=[''],
        x=[night_percent],
        orientation='h',
        marker=dict(color='#6b5b95'),  # Dark purple
        showlegend=False,
        text='',
        hoverinfo='skip',
        name='Night'
    ))
    
    # Add text annotations for Day (left) and Night (right)
    fig_occasion.add_annotation(
        x=-8,
        y=0,
        text='Day',
        showarrow=False,
        font=dict(size=16, color='#6b5b95', weight='bold'),
        xanchor='right',
        yanchor='middle'
    )
    
    fig_occasion.add_annotation(
        x=108,
        y=0,
        text='Night',
        showarrow=False,
        font=dict(size=16, color='#6b5b95', weight='bold'),
        xanchor='left',
        yanchor='middle'
    )
    
    fig_occasion.update_layout(
        height=150,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        barmode='stack',
        showlegend=False,
        xaxis=dict(
            title='',
            range=[-15, 115],
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        yaxis=dict(
            title='',
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        margin=dict(l=60, r=60, t=20, b=20)
    )
    st.plotly_chart(fig_occasion, use_container_width=True)
    
    st.markdown("---")
    
    # SIMILAR PERFUME RECOMMENDATIONS
    st.markdown('<h3 style="color: #6b5b95;">Similar Perfumes You Might Like</h3>', unsafe_allow_html=True)
    
    similar_perfumes = get_similar_perfumes(perfume)
    
    if similar_perfumes:
        # Display similar perfumes in 3 columns with proper alignment
        for i in range(0, len(similar_perfumes), 3):
            col1, col2, col3 = st.columns(3, gap="medium")
            
            with col1:
                if i < len(similar_perfumes):
                    display_perfume_card(similar_perfumes[i], show_ml_badge=True, source='current')
            
            with col2:
                if i + 1 < len(similar_perfumes):
                    display_perfume_card(similar_perfumes[i + 1], show_ml_badge=True, source='current')
            
            with col3:
                if i + 2 < len(similar_perfumes):
                    display_perfume_card(similar_perfumes[i + 2], show_ml_badge=True, source='current')

def get_similar_perfumes(perfume: Dict, limit: int = 4) -> List[Dict]:
    """
    Get similar perfumes based on scent type, gender, and price range.
    Also considers current search filters if available.
    
    Args:
        perfume: The reference perfume
        limit: Maximum number of similar perfumes to return
    
    Returns:
        List of similar perfumes
    """
    all_perfumes = st.session_state.perfume_database
    similar = []
    
    for p in all_perfumes:
        # Skip the same perfume
        if p['id'] == perfume['id']:
            continue
        
        score = 0
        
        # Same scent type (high weight)
        if p['scent_type'] == perfume['scent_type']:
            score += 3
        
        # Same gender
        if p['gender'] == perfume['gender'] or p['gender'] == 'Unisex' or perfume['gender'] == 'Unisex':
            score += 2
        
        # Similar price range (+/- $30)
        if abs(p['price'] - perfume['price']) <= 30:
            score += 1
        
        # Match search context filters if available
        if st.session_state.search_context:
            if 'brand' in st.session_state.search_context:
                if p['brand'] in st.session_state.search_context['brand']:
                    score += 1
            
            if 'scent_type' in st.session_state.search_context:
                if p['scent_type'] in st.session_state.search_context['scent_type']:
                    score += 2
        
        if score > 0:
            similar.append((p, score))
    
    # Sort by score and return top matches
    similar.sort(key=lambda x: x[1], reverse=True)
    
    # Apply ML ranking to similar perfumes
    similar_perfumes = [p for p, score in similar[:limit * 2]]
    ml_sorted = get_ml_sorted_perfumes(similar_perfumes)
    
    return ml_sorted[:limit]

# ============================================================================
# QUESTIONNAIRE SECTION
# ============================================================================

def render_questionnaire_section():
    """
    Render the questionnaire section with bipolar sliders.
    """
    # Check if showing perfume details (when user clicks View Full Details from recommendations)
    if st.session_state.show_perfume_details and st.session_state.current_perfume:
        render_perfume_detail_view(st.session_state.current_perfume)
        return
    
    # Check if showing results
    if st.session_state.show_questionnaire_results:
        render_questionnaire_results()
        return
    
    # Back button
    render_back_button("home", "Back to Home")
    
    st.markdown('<h2 style="color: #6b5b95;">Perfume Questionnaire</h2>', unsafe_allow_html=True)
    st.write("Answer these questions to discover your perfect scent profile.")
    
    st.markdown("---")
    
    # Define questionnaire questions (bipolar sliders)
    questions = [
        {
            "id": "intensity",
            "title": "Question 1 of 5",
            "left_label": "1 - Subtle",
            "right_label": "5 - Strong/Noticeable",
            "key": "q1_intensity"
        },
        {
            "id": "warmth",
            "title": "Question 2 of 5",
            "left_label": "1 - Fresh/Light",
            "right_label": "5 - Warm/Intense",
            "key": "q2_warmth"
        },
        {
            "id": "sweetness",
            "title": "Question 3 of 5",
            "left_label": "1 - Dry/Herbal",
            "right_label": "5 - Sweet/Gourmand",
            "key": "q3_sweetness"
        },
        {
            "id": "occasion",
            "title": "Question 4 of 5",
            "left_label": "1 - Daily/Office",
            "right_label": "5 - Evening/Event/Date",
            "key": "q4_occasion"
        },
        {
            "id": "character",
            "title": "Question 5 of 5",
            "left_label": "1 - Feminine",
            "right_label": "5 - Masculine",
            "key": "q5_character"
        }
    ]
    
    # Get current question
    current_q = questions[st.session_state.current_question]
    
    # Display question
    st.markdown(f'<h3 style="color: #6b5b95;">{current_q["title"]}</h3>', unsafe_allow_html=True)
    
    # Bipolar slider layout - clean and full width
    st.markdown('<div class="bipolar-slider">', unsafe_allow_html=True)
    
    # Labels at the top
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="bipolar-label" style="text-align: left;">{current_q["left_label"]}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="bipolar-label" style="text-align: right;">{current_q["right_label"]}</div>', unsafe_allow_html=True)
    
    # Simple slider - let Streamlit handle it
    answer = st.slider(
        "Move slider to select",
        min_value=1,
        max_value=5,
        value=st.session_state.questionnaire_answers.get(current_q['id'], 3),
        step=1,
        key=current_q['key'],
        label_visibility="collapsed",
        format=""
    )
    
    # Custom clean numbers below slider
    st.markdown("""
    <div class="scale-numbers">
        <span>1</span>
        <span>2</span>
        <span>3</span>
        <span>4</span>
        <span>5</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Store answer
    st.session_state.questionnaire_answers[current_q['id']] = answer
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation buttons
    col_back, col_next = st.columns(2)
    
    with col_back:
        if st.session_state.current_question > 0:
            if st.button("← Back", key="q_back", use_container_width=True):
                st.session_state.current_question -= 1
                st.rerun()
    
    with col_next:
        if st.session_state.current_question < len(questions) - 1:
            if st.button("Next →", key="q_next", use_container_width=True):
                st.session_state.current_question += 1
                st.rerun()
        else:
            if st.button("Submit", key="q_submit", use_container_width=True):
                st.session_state.show_questionnaire_results = True
                st.rerun()

def render_questionnaire_results():
    """
    Display questionnaire results with radar chart and perfume recommendations.
    """
    # Back button
    if st.button("← Back to Questionnaire", key="back_from_results"):
        st.session_state.show_questionnaire_results = False
        st.rerun()
    
    st.markdown('<h2 style="color: #6b5b95;">Your Scent Profile</h2>', unsafe_allow_html=True)
    
    # Get answers
    answers = st.session_state.questionnaire_answers
    
    # RADAR CHART
    st.markdown('<h3 style="color: #6b5b95;">Profile Visualization</h3>', unsafe_allow_html=True)
    
    # Prepare data for radar chart
    categories = ['Intensity', 'Warmth', 'Sweetness', 'Occasion', 'Gender']
    values = [
        answers.get('intensity', 3),
        answers.get('warmth', 3),
        answers.get('sweetness', 3),
        answers.get('occasion', 3),
        answers.get('character', 3)
    ]
    
    # Close the polygon by appending first value to end
    values += values[:1]
    categories += categories[:1]
    
    # Create radar chart
    fig_radar = go.Figure()
    
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(107, 91, 149, 0.3)',
        line=dict(color='#6b5b95', width=3),  # Thicker line (width=3) for consistent outline
        name='Your Profile'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5],
                tickmode='linear',
                tick0=0,
                dtick=1
            )
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14, color='#333')
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    st.markdown("---")
    
    # PERFUME RECOMMENDATIONS
    st.markdown('<h3 style="color: #6b5b95;">Recommended Perfumes</h3>', unsafe_allow_html=True)
    
    recommendations = get_questionnaire_recommendations()
    
    if recommendations:
        st.success(f"Based on your profile, we found {len(recommendations)} perfume(s) for you")
        
        # Display recommendations in 3 columns with proper alignment
        for i in range(0, len(recommendations), 3):
            col1, col2, col3 = st.columns(3, gap="medium")
            
            with col1:
                if i < len(recommendations):
                    display_perfume_card(recommendations[i], show_ml_badge=True, source='questionnaire')
            
            with col2:
                if i + 1 < len(recommendations):
                    display_perfume_card(recommendations[i + 1], show_ml_badge=True, source='questionnaire')
            
            with col3:
                if i + 2 < len(recommendations):
                    display_perfume_card(recommendations[i + 2], show_ml_badge=True, source='questionnaire')
    else:
        st.info("We couldn't find perfect matches. Try browsing our search section.")

def get_questionnaire_recommendations() -> List[Dict]:
    """
    Get perfume recommendations based on questionnaire answers.
    Uses scoring algorithm to match user profile with perfume attributes.
    
    Returns:
        List of recommended perfumes
    """
    answers = st.session_state.questionnaire_answers
    all_perfumes = st.session_state.perfume_database
    scored_perfumes = []
    
    for perfume in all_perfumes:
        score = 0
        
        # Intensity (1-5): Subtle to Strong
        intensity = answers.get('intensity', 3)
        # Map to scent types
        if intensity <= 2:  # Subtle
            if perfume['scent_type'] in ['Fresh', 'Citrus', 'Green']:
                score += 3
        elif intensity >= 4:  # Strong
            if perfume['scent_type'] in ['Oriental', 'Leather', 'Woody']:
                score += 3
        else:  # Medium
            score += 1
        
        # Warmth (1-5): Fresh/Light to Warm/Intense
        warmth = answers.get('warmth', 3)
        if warmth <= 2:  # Fresh/Light
            if perfume['scent_type'] in ['Fresh', 'Citrus', 'Green']:
                score += 3
        elif warmth >= 4:  # Warm/Intense
            if perfume['scent_type'] in ['Oriental', 'Gourmand', 'Woody']:
                score += 3
        
        # Sweetness (1-5): Dry/Herbal to Sweet/Gourmand
        sweetness = answers.get('sweetness', 3)
        if sweetness <= 2:  # Dry/Herbal
            if perfume['scent_type'] in ['Green', 'Woody', 'Fresh']:
                score += 3
        elif sweetness >= 4:  # Sweet/Gourmand
            if perfume['scent_type'] in ['Gourmand', 'Floral']:
                score += 3
        
        # Occasion (1-5): Daily/Office to Evening/Event/Date
        occasion = answers.get('occasion', 3)
        occasion_scores = perfume.get('occasion', {'Day': 3, 'Night': 3})
        if occasion <= 2:  # Daily/Office
            score += occasion_scores.get('Day', 3)
        elif occasion >= 4:  # Evening/Event/Date
            score += occasion_scores.get('Night', 3)
        else:
            score += 1
        
        # Character (1-5): Feminine to Masculine
        character = answers.get('character', 3)
        if character <= 2:  # Feminine
            if perfume['gender'] == 'Female':
                score += 3
            elif perfume['gender'] == 'Unisex':
                score += 1
        elif character >= 4:  # Masculine
            if perfume['gender'] == 'Male':
                score += 3
            elif perfume['gender'] == 'Unisex':
                score += 1
        else:  # Neutral
            if perfume['gender'] == 'Unisex':
                score += 3
        
        scored_perfumes.append((perfume, score))
    
    # Sort by score
    scored_perfumes.sort(key=lambda x: x[1], reverse=True)
    
    # Get top recommendations
    top_perfumes = [p for p, score in scored_perfumes if score >= 5][:8]
    
    # Apply ML ranking
    return get_ml_sorted_perfumes(top_perfumes)

# ============================================================================
# PERFUME INVENTORY SECTION
# ============================================================================

def render_inventory_section():
    """
    Render the personal perfume inventory section.
    Shows user's collection with statistics and analytics.
    """
    scroll_to_top()
    
    # Check if showing perfume details
    if st.session_state.show_perfume_details and st.session_state.current_perfume:
        render_perfume_detail_view(st.session_state.current_perfume)
        return
    
    # Check if adding perfume
    if 'adding_perfume' in st.session_state and st.session_state.adding_perfume:
        render_add_perfume_view()
        return
    
    # Back button
    render_back_button("home", "Back to Home")
    
    st.markdown('<h2 style="color: #6b5b95;">My Perfumes</h2>', unsafe_allow_html=True)
    
    # Get user inventory
    inventory = st.session_state.user_inventory
    
    if not inventory:
        st.info("Your perfume collection is empty. Click the + button below to add perfumes.")
    
    # Display inventory grid - cleaner layout
    # First item is "Add" button
    cols_per_row = 4  # 4 columns for better spacing
    all_items = ['add'] + inventory
    
    for i in range(0, len(all_items), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j in range(cols_per_row):
            idx = i + j
            if idx < len(all_items):
                with cols[j]:
                    if all_items[idx] == 'add':
                        # Add perfume button wrapper to match total height with buttons
                        st.markdown("""
                            <div class="perfume-card" style="padding: 15px; height: 330px; display: flex; flex-direction: column; justify-content: center; align-items: center; background: white; border: 3px dashed #6b5b95; border-radius: 12px;">
                                <div style="flex-grow: 1; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                                    <p style="font-size: 60px; color: #6b5b95; margin: 0; line-height: 1;">+</p>
                                    <p style="color: #6b5b95; margin-top: 15px; font-weight: 600; font-size: 16px;">Add Perfume</p>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Button outside to align with other cards' buttons
                        if st.button("Add New", key="btn_add_perfume", use_container_width=True, type="primary"):
                            st.session_state.adding_perfume = True
                            st.rerun()
                    else:
                        # Display perfume in inventory
                        display_inventory_perfume_card(all_items[idx], idx - 1)
    
    # Show statistics only if inventory is not empty
    if inventory:
        st.markdown("---")
        render_inventory_statistics(inventory)
        
        # Show ML recommendations if user has enough perfumes
        if len(inventory) >= ML_CONFIG['min_inventory_size']:
            st.markdown("---")
            render_ml_recommendations_in_inventory(inventory)

def display_inventory_perfume_card(perfume: Dict, index: int):
    """
    Display a perfume card in the inventory with proper alignment.
    PERFECTLY ALIGNED with fixed heights.
    
    Args:
        perfume: Perfume dictionary
        index: Index in inventory for removal
    """
    # Use perfume icon if no image available
    image_url = perfume.get('image_url', '')
    if not image_url or image_url == '':
        image_url = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjI1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjAwIiBoZWlnaHQ9IjI1MCIgZmlsbD0iI2U4ZTRmMCIvPjxwYXRoIGQ9Ik04MCA2MGgyMHY0MEg4MHoiIGZpbGw9IiM2YjViOTUiLz48cmVjdCB4PSI2MCIgeT0iMTAwIiB3aWR0aD0iNjAiIGhlaWdodD0iMTIwIiByeD0iMTAiIGZpbGw9IiM2YjViOTUiIG9wYWNpdHk9IjAuOCIvPjxyZWN0IHg9IjcwIiB5PSIxMTAiIHdpZHRoPSI0MCIgaGVpZ2h0PSI5MCIgZmlsbD0iI2M4YjhkOCIgb3BhY2l0eT0iMC42Ii8+PC9zdmc+'
    
    with st.container():
        st.markdown(f"""
            <div class="perfume-card" style="padding: 15px; height: 330px; display: flex; flex-direction: column; justify-content: space-between;">
                <div style="height: 180px; display: flex; align-items: center; justify-content: center; margin-bottom: 10px;">
                    <img src="{image_url}" 
                         onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjI1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjAwIiBoZWlnaHQ9IjI1MCIgZmlsbD0iI2U4ZTRmMCIvPjxwYXRoIGQ9Ik04MCA2MGgyMHY0MEg4MHoiIGZpbGw9IiM2YjViOTUiLz48cmVjdCB4PSI2MCIgeT0iMTAwIiB3aWR0aD0iNjAiIGhlaWdodD0iMTIwIiByeD0iMTAiIGZpbGw9IiM2YjViOTUiIG9wYWNpdHk9IjAuOCIvPjxyZWN0IHg9IjcwIiB5PSIxMTAiIHdpZHRoPSI0MCIgaGVpZ2h0PSI5MCIgZmlsbD0iI2M4YjhkOCIgb3BhY2l0eT0iMC42Ii8+PC9zdmc+'"
                         style="max-width: 130px; max-height: 180px; object-fit: contain; border-radius: 8px;">
                </div>
                <h4 style="color: #6b5b95; margin: 0 0 8px 0; text-align: center; height: 45px; display: flex; align-items: center; justify-content: center; font-size: 14px; line-height: 1.2; overflow: hidden; padding: 0 5px;">{perfume['name']}</h4>
                <div style="height: 20px; margin-bottom: 8px; text-align: center; border-bottom: 1px solid #e8e4f0; padding-bottom: 5px;">
                    <p style="color: #888; font-style: italic; font-size: 11px; margin: 0; font-weight: 500;">Brand: {perfume['brand']}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Remove", key=f"remove_inv_{index}", use_container_width=True):
                st.session_state.user_inventory.pop(index)
                save_user_inventory(st.session_state.user_inventory)
                st.rerun()
        with col2:
            view_clicked = st.button("View Details", key=f"view_inventory_{perfume['id']}_{index}", use_container_width=True)
            if view_clicked:
                # View details button - does NOT remove, ONLY shows details
                record_interaction(perfume['id'], 'click')
                st.session_state.current_perfume = perfume.copy()
                st.session_state.show_perfume_details = True
                st.session_state.detail_view_source = 'inventory'  # Track that we came from inventory
                # Make sure we're not in adding mode
                if 'adding_perfume' in st.session_state:
                    st.session_state.adding_perfume = False
                st.rerun()

def render_add_perfume_view():
    """
    Render the view for adding perfumes to inventory.
    Shows search functionality with live API search and perfume selection.
    """
    # Back button
    if st.button("← Back to Inventory", key="back_from_add"):
        st.session_state.adding_perfume = False
        st.rerun()
    
    st.markdown('<h2 style="color: #6b5b95;">Add Perfume to Collection</h2>', unsafe_allow_html=True)
    
    # Search bar
    add_search = st.text_input(
        "Search for perfume to add (minimum 3 characters)",
        placeholder="Enter perfume name or brand...",
        key="add_search_input"
    )
    
    st.markdown("---")
    
    # Show perfumes based on search
    if add_search and len(add_search) >= 3:
        # Do live API search
        with st.spinner("Searching Fragella database..."):
            api_results = search_fragella_perfumes(add_search, limit=20)
            if api_results:
                # Transform results
                filtered = [transform_api_perfume(p) for p in api_results]
                # Update database with new perfumes (avoid duplicates)
                for perfume in filtered:
                    if not any(p['id'] == perfume['id'] for p in st.session_state.perfume_database):
                        st.session_state.perfume_database.append(perfume)
            else:
                filtered = []
    else:
        # Show existing database
        filtered = st.session_state.perfume_database[:20]  # Limit to first 20
        if add_search and len(add_search) < 3:
            st.info("Please enter at least 3 characters to search")
    
    st.write(f"Found {len(filtered)} perfume(s)")
    
    # Display perfumes with add button - 3 columns with proper alignment
    for i in range(0, len(filtered), 3):
        col1, col2, col3 = st.columns(3, gap="medium")
        
        with col1:
            if i < len(filtered):
                display_addable_perfume_card(filtered[i])
        
        with col2:
            if i + 1 < len(filtered):
                display_addable_perfume_card(filtered[i + 1])
        
        with col3:
            if i + 2 < len(filtered):
                display_addable_perfume_card(filtered[i + 2])

def display_addable_perfume_card(perfume: Dict):
    """
    Display perfume card with add and view buttons.
    PERFECTLY ALIGNED with fixed heights.
    
    Args:
        perfume: Perfume dictionary
    """
    # Use perfume icon if no image available
    image_url = perfume.get('image_url', '')
    if not image_url or image_url == '':
        image_url = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjI1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjAwIiBoZWlnaHQ9IjI1MCIgZmlsbD0iI2U4ZTRmMCIvPjxwYXRoIGQ9Ik04MCA2MGgyMHY0MEg4MHoiIGZpbGw9IiM2YjViOTUiLz48cmVjdCB4PSI2MCIgeT0iMTAwIiB3aWR0aD0iNjAiIGhlaWdodD0iMTIwIiByeD0iMTAiIGZpbGw9IiM2YjViOTUiIG9wYWNpdHk9IjAuOCIvPjxyZWN0IHg9IjcwIiB5PSIxMTAiIHdpZHRoPSI0MCIgaGVpZ2h0PSI5MCIgZmlsbD0iI2M4YjhkOCIgb3BhY2l0eT0iMC42Ii8+PC9zdmc+'
    
    with st.container():
        st.markdown(f"""
            <div class="perfume-card" style="padding: 15px; height: 480px; display: flex; flex-direction: column; justify-content: space-between;">
                <div style="height: 200px; display: flex; align-items: center; justify-content: center; margin-bottom: 15px;">
                    <img src="{image_url}" 
                         onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjI1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjAwIiBoZWlnaHQ9IjI1MCIgZmlsbD0iI2U4ZTRmMCIvPjxwYXRoIGQ9Ik04MCA2MGgyMHY0MEg4MHoiIGZpbGw9IiM2YjViOTUiLz48cmVjdCB4PSI2MCIgeT0iMTAwIiB3aWR0aD0iNjAiIGhlaWdodD0iMTIwIiByeD0iMTAiIGZpbGw9IiM2YjViOTUiIG9wYWNpdHk9IjAuOCIvPjxyZWN0IHg9IjcwIiB5PSIxMTAiIHdpZHRoPSI0MCIgaGVpZ2h0PSI5MCIgZmlsbD0iI2M4YjhkOCIgb3BhY2l0eT0iMC42Ii8+PC9zdmc+'"
                         style="max-width: 150px; max-height: 200px; object-fit: contain; border-radius: 8px;">
                </div>
                <h4 style="color: #6b5b95; margin: 0 0 10px 0; text-align: center; height: 50px; display: flex; align-items: center; justify-content: center; font-size: 16px; line-height: 1.2; overflow: hidden; padding: 0 5px;">{perfume['name']}</h4>
                <div style="height: 25px; margin-bottom: 10px; text-align: center; border-bottom: 1px solid #e8e4f0; padding-bottom: 8px;">
                    <p style="color: #888; font-style: italic; font-size: 13px; margin: 0; font-weight: 500;">Brand: {perfume['brand']}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Add", key=f"add_to_inv_{perfume['id']}", use_container_width=True):
                add_to_user_inventory(perfume)
                record_interaction(perfume['id'], 'add_to_inventory')
                st.success(f"Added to inventory")
        with col2:
            view_clicked = st.button("View Details", key=f"view_no_add_{perfume['id']}", use_container_width=True)
            if view_clicked:
                # THIS ONLY VIEWS - does NOT add to inventory
                record_interaction(perfume['id'], 'click')
                st.session_state.current_perfume = perfume.copy()
                st.session_state.show_perfume_details = True
                st.session_state.adding_perfume = False  # Exit add mode
                st.rerun()

def add_to_user_inventory(perfume: Dict):
    """
    Add a perfume to user's inventory.
    
    Args:
        perfume: Perfume dictionary to add
    """
    # Check if already in inventory
    if any(p['id'] == perfume['id'] for p in st.session_state.user_inventory):
        st.warning("This perfume is already in your collection")
        return
    
    # Add to inventory
    st.session_state.user_inventory.append(perfume)
    
    # Save to file
    save_user_inventory(st.session_state.user_inventory)

def render_inventory_statistics(inventory: List[Dict]):
    """
    Render statistics and analytics for user's perfume collection.
    Displays donut charts for notes and bar charts for seasonality and occasion.
    
    Args:
        inventory: List of perfumes in user's collection
    """
    st.markdown('<h2 style="color: #6b5b95;">Collection Statistics</h2>', unsafe_allow_html=True)
    
    # DONUT CHARTS FOR NOTES
    st.markdown('<h3 style="color: #6b5b95;">Note Distribution</h3>', unsafe_allow_html=True)
    
    # Collect all notes - extract names from dict format
    top_notes = []
    heart_notes = []
    base_notes = []
    
    for perfume in inventory:
        # Handle both dict format (from API) and string format (old data)
        for note in perfume['top_notes']:
            top_notes.append(note['name'] if isinstance(note, dict) else note)
        for note in perfume['heart_notes']:
            heart_notes.append(note['name'] if isinstance(note, dict) else note)
        for note in perfume['base_notes']:
            base_notes.append(note['name'] if isinstance(note, dict) else note)
    
    # Create three columns for donut charts
    col_top, col_heart, col_base = st.columns(3)
    
    with col_top:
        st.markdown("**Top Notes**")
        top_counter = Counter(top_notes)
        create_donut_chart(top_counter, "Top Notes")
    
    with col_heart:
        st.markdown("**Heart Notes**")
        heart_counter = Counter(heart_notes)
        create_donut_chart(heart_counter, "Heart Notes")
    
    with col_base:
        st.markdown("**Base Notes**")
        base_counter = Counter(base_notes)
        create_donut_chart(base_counter, "Base Notes")
    
    st.markdown("---")
    
    # SEASONALITY - Horizontal bar chart with rounded corners
    st.markdown('<h3 style="color: #6b5b95;">Seasonality Distribution</h3>', unsafe_allow_html=True)
    
    # Aggregate seasonality scores
    season_totals = {'Winter': 0, 'Fall': 0, 'Spring': 0, 'Summer': 0}
    
    for perfume in inventory:
        for season, score in perfume['seasonality'].items():
            season_totals[season] += score
    
    # Create horizontal bar chart
    seasons = ['Summer', 'Spring', 'Winter', 'Fall']
    values = [season_totals.get(season, 0) for season in seasons]
    
    fig_season = go.Figure()
    
    for i, season in enumerate(seasons):
        fig_season.add_trace(go.Bar(
            y=[season],
            x=[values[i]],
            orientation='h',
            marker=dict(
                color='#8b7aa8',  # Purple to match website theme
                line=dict(width=0)
            ),
            showlegend=False,
            text='',
            hoverinfo='skip'
        ))
    
    fig_season.update_layout(
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333', size=15, family='Arial, sans-serif'),
        xaxis=dict(
            title='',
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        yaxis=dict(
            title='',
            tickfont=dict(size=14),
            showgrid=False
        ),
        margin=dict(l=100, r=40, t=20, b=40),
        bargap=0.3,
        bargroupgap=0.1
    )
    
    st.plotly_chart(fig_season, use_container_width=True)
    
    # OCCASION - Full width stacked bar (NO percentages)
    st.markdown('<h3 style="color: #6b5b95;">Occasion Distribution</h3>', unsafe_allow_html=True)
    
    # Aggregate occasion scores
    occasion_totals = {'Day': 0, 'Night': 0}
    
    for perfume in inventory:
        for occasion, score in perfume.get('occasion', {'Day': 3, 'Night': 3}).items():
            if occasion in occasion_totals:
                occasion_totals[occasion] += score
    
    # Calculate percentages
    total = occasion_totals['Day'] + occasion_totals['Night']
    day_percent = (occasion_totals['Day'] / total * 100) if total > 0 else 50
    night_percent = (occasion_totals['Night'] / total * 100) if total > 0 else 50
    
    # Create horizontal stacked bar chart
    fig_occasion = go.Figure()
    
    # Day portion (left, lighter purple)
    fig_occasion.add_trace(go.Bar(
        y=[''],
        x=[day_percent],
        orientation='h',
        marker=dict(color='#9b8bb5'),  # Lighter purple
        showlegend=False,
        text='',
        hoverinfo='skip',
        name='Day'
    ))
    
    # Night portion (right, dark purple)
    fig_occasion.add_trace(go.Bar(
        y=[''],
        x=[night_percent],
        orientation='h',
        marker=dict(color='#6b5b95'),  # Dark purple
        showlegend=False,
        text='',
        hoverinfo='skip',
        name='Night'
    ))
    
    # Add text annotations for Day (left) and Night (right)
    fig_occasion.add_annotation(
        x=-8,
        y=0,
        text='Day',
        showarrow=False,
        font=dict(size=16, color='#6b5b95', weight='bold'),
        xanchor='right',
        yanchor='middle'
    )
    
    fig_occasion.add_annotation(
        x=108,
        y=0,
        text='Night',
        showarrow=False,
        font=dict(size=16, color='#6b5b95', weight='bold'),
        xanchor='left',
        yanchor='middle'
    )
    
    fig_occasion.update_layout(
        height=150,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        barmode='stack',
        showlegend=False,
        xaxis=dict(
            title='',
            range=[-15, 115],
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        yaxis=dict(
            title='',
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        margin=dict(l=60, r=60, t=20, b=20)
    )
    st.plotly_chart(fig_occasion, use_container_width=True)

def create_donut_chart(note_counter: Counter, title: str):
    """
    Create a donut chart for note distribution.
    Shows top 5 notes and groups the rest as 'Rest'.
    
    Args:
        note_counter: Counter object with note frequencies
        title: Chart title
    """
    if not note_counter:
        st.write("No data")
        return
    
    # Get top 5 notes
    top_5 = note_counter.most_common(5)
    
    # Calculate rest
    total = sum(note_counter.values())
    top_5_total = sum(count for _, count in top_5)
    rest_count = total - top_5_total
    
    # Prepare data
    labels = [note for note, _ in top_5]
    values = [count for _, count in top_5]
    
    if rest_count > 0:
        labels.append('Rest')
        values.append(rest_count)
    
    # Create donut chart
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=['#6b5b95', '#8b7b9f', '#ab9bb9', '#cbbbc9', '#e8e4f0', '#f8f7fa'])
    )])
    
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=0, b=0, l=0, r=0),
        height=250
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# ML RECOMMENDATIONS IN INVENTORY SECTION
# ============================================================================

def render_ml_recommendations_in_inventory(inventory: List[Dict]):
    """
    Render ML-powered recommendations at the bottom of inventory section.
    Auto-generates recommendations based on user's collection.
    
    Args:
        inventory: User's perfume inventory
    """
    st.markdown('<h2 style="color: #6b5b95;">Recommendations Based on Your Scent Profile</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #888; font-size: 14px; margin-bottom: 20px;">Machine learning analyzes your collection to suggest personalized perfumes</p>', unsafe_allow_html=True)
    
    # Get all perfumes from database
    all_perfumes = st.session_state.perfume_database
    
    # Auto-generate recommendations (no button needed)
    with st.spinner("Analyzing your collection..."):
        recommendations = get_ml_recommendations(
            inventory, 
            all_perfumes, 
            top_n=6,  # Show 6 recommendations
            ensure_diversity=True
        )
    
    if not recommendations:
        st.info("No strong matches found in the catalog. Keep adding perfumes to improve recommendations!")
        return
    
    # Display recommendations in grid (3 per row for consistency)
    for i in range(0, len(recommendations), 3):
        cols = st.columns(3)
        
        for j in range(3):
            idx = i + j
            if idx < len(recommendations):
                perfume = recommendations[idx]
                
                with cols[j]:
                    # ML Score badge
                    match_score = int(perfume['ml_score'] * 100)
                    badge_color = "#4CAF50" if match_score >= 80 else "#6b5b95" if match_score >= 70 else "#888"
                    
                    image_url = perfume.get('image_url', '')
                    if not image_url:
                        image_url = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjI1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjAwIiBoZWlnaHQ9IjI1MCIgZmlsbD0iI2U4ZTRmMCIvPjxwYXRoIGQ9Ik04MCA2MGgyMHY0MEg4MHoiIGZpbGw9IiM2YjViOTUiLz48cmVjdCB4PSI2MCIgeT0iMTAwIiB3aWR0aD0iNjAiIGhlaWdodD0iMTIwIiByeD0iMTAiIGZpbGw9IiM2YjViOTUiIG9wYWNpdHk9IjAuOCIvPjxyZWN0IHg9IjcwIiB5PSIxMTAiIHdpZHRoPSI0MCIgaGVpZ2h0PSI5MCIgZmlsbD0iI2M4YjhkOCIgb3BhY2l0eT0iMC42Ii8+PC9zdmc+'
                    
                    # Card with consistent height (matches inventory cards: 330px)
                    st.markdown(f"""
                        <div class="perfume-card" style="padding: 15px; height: 330px; display: flex; flex-direction: column; justify-content: space-between; position: relative;">
                            <div style="position: absolute; top: 10px; right: 10px; background: {badge_color}; color: white; padding: 4px 8px; border-radius: 15px; font-size: 11px; font-weight: bold; z-index: 10;">
                                {match_score}%
                            </div>
                            <div style="height: 180px; display: flex; align-items: center; justify-content: center; margin-bottom: 10px;">
                                <img src="{image_url}" 
                                     onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjI1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjAwIiBoZWlnaHQ9IjI1MCIgZmlsbD0iI2U4ZTRmMCIvPjxwYXRoIGQ9Ik04MCA2MGgyMHY0MEg4MHoiIGZpbGw9IiM2YjViOTUiLz48cmVjdCB4PSI2MCIgeT0iMTAwIiB3aWR0aD0iNjAiIGhlaWdodD0iMTIwIiByeD0iMTAiIGZpbGw9IiM2YjViOTUiIG9wYWNpdHk9IjAuOCIvPjxyZWN0IHg9IjcwIiB5PSIxMTAiIHdpZHRoPSI0MCIgaGVpZ2h0PSI5MCIgZmlsbD0iI2M4YjhkOCIgb3BhY2l0eT0iMC42Ii8+PC9zdmc+'"
                                     style="max-width: 130px; max-height: 180px; object-fit: contain; border-radius: 8px;">
                            </div>
                            <h4 style="color: #6b5b95; margin: 0 0 8px 0; text-align: center; height: 45px; display: flex; align-items: center; justify-content: center; font-size: 14px; line-height: 1.2; overflow: hidden; padding: 0 5px;">{perfume['name']}</h4>
                            <div style="height: 20px; margin-bottom: 8px; text-align: center; border-bottom: 1px solid #e8e4f0; padding-bottom: 5px;">
                                <p style="color: #888; font-style: italic; font-size: 11px; margin: 0; font-weight: 500;">Brand: {perfume['brand']}</p>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Action buttons
                    col_view, col_add = st.columns(2)
                    
                    with col_view:
                        if st.button("View", key=f"ml_view_{idx}", use_container_width=True):
                            record_interaction(perfume['id'], 'click')
                            st.session_state.current_perfume = perfume.copy()
                            st.session_state.show_perfume_details = True
                            st.session_state.detail_view_source = 'inventory'
                            st.rerun()
                    
                    with col_add:
                        if st.button("Add", key=f"ml_add_{idx}", use_container_width=True):
                            add_to_user_inventory(perfume)
                            record_interaction(perfume['id'], 'add_to_inventory')
                            st.rerun()

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main application entry point.
    Coordinates all sections and navigation.
    """
    # Apply custom styling
    apply_custom_styling()
    
    # Initialize session state
    initialize_session_state()
    
    # Render header
    render_header()
    
    # Route to appropriate section
    if st.session_state.active_section == "home":
        render_landing_page()
        
    elif st.session_state.active_section == "search":
        render_search_section()
        
    elif st.session_state.active_section == "questionnaire":
        render_questionnaire_section()
        
    elif st.session_state.active_section == "inventory":
        render_inventory_section()

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()
