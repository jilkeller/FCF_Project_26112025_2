"""
MACHINE LEARNING RECOMMENDATION SYSTEM FOR SCENTIFY
====================================================

This module implements a personalized perfume recommendation system based on user's 
perfume inventory using supervised machine learning.

CORE CONCEPT:
-------------
The system learns user taste from their owned perfumes and recommends new ones based 
on scent characteristics and features.

KEY COMPONENTS:
---------------
1. Feature Extraction: Convert perfume characteristics into numerical vectors
2. Training Dataset: Positive samples (owned) + Negative samples (non-owned)
3. ML Model: Logistic Regression or Decision Tree
4. Prediction: Score non-owned perfumes by probability P(user likes | features)
5. Recommendations: Rank and filter top N perfumes with explanations

AUTHOR: Scentify Development Team
DATE: November 2025
"""

import numpy as np
import pickle
import os
import random
from typing import List, Dict, Optional, Tuple
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths for model persistence
ML_MODEL_FILE = "data/ml_model.pkl"
ML_SCALER_FILE = "data/ml_scaler.pkl"

# ML Configuration parameters
ML_CONFIG = {
    'negative_samples_ratio': 2,  # For each positive sample, generate N negative samples
    'min_inventory_size': 2,  # Minimum perfumes in inventory to train ML model
    'model_type': 'logistic_regression',  # Options: 'logistic_regression', 'decision_tree'
    'random_state': 42,  # For reproducibility
    'min_recommendation_probability': 0.5,  # Minimum probability threshold to show recommendation
}


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_perfume_features(perfume: Dict) -> np.ndarray:
    """
    Extract numerical feature vector from a perfume for ML model.
    
    This function converts perfume characteristics into a numerical vector
    that machine learning models can process.
    
    FEATURES EXTRACTED (39 total):
    ------------------------------
    1. Main Accords (30 features): Binary indicators for common perfume accords
       - floral, fresh, woody, citrus, oriental, spicy, sweet, gourmand, fruity,
       - aromatic, green, aquatic, leather, powdery, herbal, amber, musk, vanilla,
       - rose, jasmine, lavender, bergamot, sandalwood, patchouli, oud, vetiver,
       - tobacco, animalic, earthy, smoky
    
    2. Seasonality (4 features): Normalized scores for Winter, Fall, Spring, Summer
       - Scale: 0-1 (original 1-5 scale normalized)
    
    3. Occasion (2 features): Day and Night suitability scores
       - Scale: 0-1 (original 1-5 scale normalized)
    
    4. Longevity (1 feature): How long the perfume lasts
       - Scale: 0-1 (categorical converted to numeric)
       - Categories: very weak, weak, moderate, long lasting, eternal
    
    5. Sillage/Strength (1 feature): How far the scent projects
       - Scale: 0-1 (categorical converted to numeric)
       - Categories: intimate, weak, moderate, strong, enormous
    
    6. Gender (1 feature): Target gender
       - Scale: 0 (Male), 0.5 (Unisex), 1 (Female)
    
    Args:
        perfume: Dictionary containing perfume data from API
    
    Returns:
        numpy array of shape (39,) containing numerical features
    
    Example:
        >>> perfume = {
        ...     'main_accords': ['floral', 'sweet', 'vanilla'],
        ...     'seasonality': {'Winter': 4, 'Fall': 3, 'Spring': 2, 'Summer': 1},
        ...     'occasion': {'Day': 2, 'Night': 5},
        ...     'longevity': 'long lasting',
        ...     'sillage': 'strong',
        ...     'gender': 'Female',
        ...     'price': 120
        ... }
        >>> features = extract_perfume_features(perfume)
        >>> features.shape
        (39,)
    """
    features = []
    
    # Define all possible main accords (comprehensive list from perfume industry)
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
    # Normalize from 1-5 scale to 0-1 scale
    seasonality = perfume.get('seasonality', {})
    features.append(float(seasonality.get('Winter', 3)) / 5.0)
    features.append(float(seasonality.get('Fall', 3)) / 5.0)
    features.append(float(seasonality.get('Spring', 3)) / 5.0)
    features.append(float(seasonality.get('Summer', 3)) / 5.0)
    
    # Occasion features (2 features: Day, Night)
    # Normalize from 1-5 scale to 0-1 scale
    occasion = perfume.get('occasion', {})
    features.append(float(occasion.get('Day', 3)) / 5.0)
    features.append(float(occasion.get('Night', 3)) / 5.0)
    
    # Longevity - convert categorical to numeric (0-1 scale)
    longevity_map = {
        'very weak': 0.1, 
        'weak': 0.3, 
        'moderate': 0.5, 
        'long lasting': 0.7,
        'long-lasting': 0.7, 
        'eternal': 0.9, 
        'very long lasting': 0.9
    }
    longevity_str = str(perfume.get('longevity', 'moderate')).lower()
    longevity_score = longevity_map.get(longevity_str, 0.5)
    features.append(longevity_score)
    
    # Sillage/Strength - convert categorical to numeric (0-1 scale)
    sillage_map = {
        'intimate': 0.2, 
        'weak': 0.3, 
        'moderate': 0.5, 
        'strong': 0.7, 
        'enormous': 0.9
    }
    sillage_str = str(perfume.get('sillage', 'moderate')).lower()
    sillage_score = sillage_map.get(sillage_str, 0.5)
    features.append(sillage_score)
    
    # Gender - convert to numeric
    # Scale: 0.0 = Male, 0.5 = Unisex, 1.0 = Female
    gender_map = {'Male': 0.0, 'Unisex': 0.5, 'Female': 1.0}
    gender = perfume.get('gender', 'Unisex')
    features.append(gender_map.get(gender, 0.5))
    
    return np.array(features)


# ============================================================================
# TRAINING DATASET CONSTRUCTION
# ============================================================================

def build_training_dataset(user_inventory: List[Dict], all_perfumes: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build training dataset with positive and negative samples.
    
    METHODOLOGY:
    ------------
    - Positive Samples (Label = 1): All perfumes in user's inventory
      These represent perfumes the user likes/owns
    
    - Negative Samples (Label = 0): Randomly selected non-owned perfumes
      These represent perfumes the user doesn't own (proxy for "doesn't like")
      Number of negatives = inventory_size × negative_samples_ratio
    
    WHY THIS APPROACH:
    ------------------
    We assume that perfumes in the user's inventory are liked by the user.
    We randomly sample non-owned perfumes as negatives to train the model
    to distinguish between user's taste and other perfumes.
    
    The negative_samples_ratio (default 2) creates a balanced but slightly
    negative-heavy dataset to avoid overfitting on small inventories.
    
    Args:
        user_inventory: List of perfumes user owns (positive samples)
        all_perfumes: Complete catalog of available perfumes
    
    Returns:
        Tuple of (X, y) where:
        - X: Feature matrix of shape (n_samples, n_features)
        - y: Label vector of shape (n_samples,) with values 0 or 1
    
    Example:
        >>> user_inventory = [perfume1, perfume2, perfume3]  # 3 owned
        >>> X, y = build_training_dataset(user_inventory, all_perfumes)
        >>> # With ratio=2, we get 3 positive + 6 negative = 9 samples
        >>> X.shape
        (9, 39)
        >>> y
        array([1, 1, 1, 0, 0, 0, 0, 0, 0])
    """
    X_positive = []
    X_negative = []
    
    # Get IDs of owned perfumes for fast lookup
    owned_ids = {p['id'] for p in user_inventory}
    
    # Extract features from positive samples (owned perfumes)
    for perfume in user_inventory:
        features = extract_perfume_features(perfume)
        X_positive.append(features)
    
    # Get all non-owned perfumes (candidates for negative samples)
    non_owned = [p for p in all_perfumes if p['id'] not in owned_ids]
    
    # Sample negative examples
    # Number of negatives = inventory_size × ratio
    num_negatives = len(user_inventory) * ML_CONFIG['negative_samples_ratio']
    if len(non_owned) > num_negatives:
        # Randomly sample if we have more non-owned than needed
        negative_samples = random.sample(non_owned, int(num_negatives))
    else:
        # Use all non-owned if we don't have enough
        negative_samples = non_owned
    
    # Extract features from negative samples
    for perfume in negative_samples:
        features = extract_perfume_features(perfume)
        X_negative.append(features)
    
    # Combine positive and negative samples
    X = np.array(X_positive + X_negative)
    y = np.array([1] * len(X_positive) + [0] * len(X_negative))
    
    return X, y


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_ml_model(user_inventory: List[Dict], all_perfumes: List[Dict]) -> Optional[Tuple]:
    """
    Train a personalized ML model based on user's perfume inventory.
    
    TRAINING PROCESS:
    -----------------
    1. Check if user has enough perfumes (min_inventory_size)
    2. Build training dataset (positive + negative samples)
    3. Standardize features using StandardScaler
    4. Train model (Logistic Regression or Decision Tree)
    5. Extract feature importance (for interpretability)
    6. Save model and scaler to disk
    
    MODEL CHOICES:
    --------------
    - Logistic Regression (default):
      * Probabilistic model that outputs P(user likes | features)
      * Linear decision boundary
      * Fast training and prediction
      * Good for interpretability (coefficients = feature importance)
    
    - Decision Tree:
      * Non-linear decision boundary
      * Can capture complex patterns
      * Feature importance from tree splits
      * May overfit on small datasets
    
    STANDARDIZATION:
    ----------------
    Features are standardized (zero mean, unit variance) to ensure
    all features contribute equally regardless of their original scale.
    
    Args:
        user_inventory: List of perfumes user owns
        all_perfumes: Complete catalog of available perfumes
    
    Returns:
        Tuple of (model, scaler, feature_importance) or None if training failed
        - model: Trained sklearn model (LogisticRegression or DecisionTreeClassifier)
        - scaler: Fitted StandardScaler for feature normalization
        - feature_importance: Array of feature importance scores
    
    Example:
        >>> result = train_ml_model(user_inventory, all_perfumes)
        >>> if result:
        ...     model, scaler, importance = result
        ...     print(f"Model trained with {len(user_inventory)} perfumes")
    """
    # Check if user has enough perfumes to train
    if len(user_inventory) < ML_CONFIG['min_inventory_size']:
        return None
    
    try:
        # Build training dataset
        X, y = build_training_dataset(user_inventory, all_perfumes)
        
        if len(X) == 0:
            return None
        
        # Standardize features (zero mean, unit variance)
        # This ensures all features contribute equally
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model based on configuration
        if ML_CONFIG['model_type'] == 'decision_tree':
            model = DecisionTreeClassifier(
                max_depth=5,  # Limit depth to prevent overfitting
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=ML_CONFIG['random_state']
            )
        else:  # Default to logistic regression
            model = LogisticRegression(
                max_iter=1000,  # Max iterations for convergence
                random_state=ML_CONFIG['random_state'],
                solver='lbfgs'  # Efficient solver for small-medium datasets
            )
        
        # Fit the model
        model.fit(X_scaled, y)
        
        # Get feature importance for interpretability
        if hasattr(model, 'feature_importances_'):
            # Decision tree has built-in feature importance
            feature_importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Logistic regression: use absolute value of coefficients
            feature_importance = np.abs(model.coef_[0])
        else:
            feature_importance = None
        
        # Save model and scaler to disk for persistence
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
    
    This allows reusing a trained model without retraining every time.
    
    Returns:
        Tuple of (model, scaler) or None if files not found
    
    Example:
        >>> model_data = load_ml_model()
        >>> if model_data:
        ...     model, scaler = model_data
        ...     print("Model loaded successfully")
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


# ============================================================================
# RECOMMENDATION GENERATION
# ============================================================================

def get_ml_recommendations(user_inventory: List[Dict], 
                          all_perfumes: List[Dict], 
                          top_n: int = 10,
                          ensure_diversity: bool = True) -> List[Dict]:
    """
    Generate personalized perfume recommendations using trained ML model.
    
    This is the MAIN recommendation function that orchestrates the entire process.
    
    PROCESS:
    --------
    1. Check if user has enough perfumes (min_inventory_size)
    2. Train or load ML model
    3. Get all candidate perfumes (non-owned)
    4. For each candidate:
       a. Extract features
       b. Standardize features using scaler
       c. Predict probability P(user likes | features)
       d. Keep if probability >= threshold
    5. Sort by probability (highest first)
    6. Apply diversity filtering (optional)
    7. Return top N recommendations
    
    SCORING:
    --------
    Each perfume gets a score (ml_score) which represents P(user likes this perfume)
    - Score 0.9: 90% probability user will like it (very high match)
    - Score 0.7: 70% probability (good match)
    - Score 0.5: 50% probability (threshold - okay match)
    
    DIVERSITY:
    ----------
    Optional diversity filtering prevents recommending too many similar perfumes.
    It ensures variety in scent types across recommendations.
    
    Args:
        user_inventory: List of perfumes user owns
        all_perfumes: Complete catalog of available perfumes
        top_n: Number of recommendations to return (default 10)
        ensure_diversity: Whether to enforce diversity in scent types (default True)
    
    Returns:
        List of recommended perfumes (dictionaries) with added fields:
        - 'ml_score': Probability score (0-1)
        - 'ml_explanation': Human-readable explanation
    
    Example:
        >>> recommendations = get_ml_recommendations(user_inventory, all_perfumes, top_n=5)
        >>> for rec in recommendations:
        ...     print(f"{rec['name']}: {rec['ml_score']:.2f} - {rec['ml_explanation']}")
        Chanel No 5: 0.92 - Highly recommended - Matches your preference for floral scents
    """
    # Check if user has enough perfumes
    if len(user_inventory) < ML_CONFIG['min_inventory_size']:
        return []
    
    # Train or load model
    model_data = train_ml_model(user_inventory, all_perfumes)
    if not model_data:
        # Try loading existing model if training failed
        model_data = load_ml_model()
        if not model_data:
            return []
    
    model, scaler = model_data[0], model_data[1]
    
    # Get IDs of owned perfumes
    owned_ids = {p['id'] for p in user_inventory}
    
    # Get candidate perfumes (non-owned only)
    candidates = [p for p in all_perfumes if p['id'] not in owned_ids]
    
    if not candidates:
        return []
    
    # Score each candidate perfume
    scored_perfumes = []
    for perfume in candidates:
        try:
            # Extract features
            features = extract_perfume_features(perfume)
            # Standardize features (must match training standardization)
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            # Get prediction probability
            if hasattr(model, 'predict_proba'):
                # Probability of class 1 (user likes)
                prob = model.predict_proba(features_scaled)[0][1]
            else:
                # Fallback for models without predict_proba
                prob = float(model.predict(features_scaled)[0])
            
            # Only include if above minimum threshold
            if prob >= ML_CONFIG['min_recommendation_probability']:
                perfume_copy = perfume.copy()
                perfume_copy['ml_score'] = prob
                perfume_copy['ml_explanation'] = generate_ml_explanation(perfume, user_inventory, prob)
                scored_perfumes.append(perfume_copy)
        
        except Exception as e:
            print(f"Error scoring perfume {perfume.get('name', 'Unknown')}: {e}")
            continue
    
    # Sort by probability score (highest first)
    scored_perfumes.sort(key=lambda p: p['ml_score'], reverse=True)
    
    # Apply diversity filtering if requested
    if ensure_diversity and len(scored_perfumes) > top_n:
        scored_perfumes = apply_diversity_filter(scored_perfumes, top_n)
    
    # Return top N recommendations
    return scored_perfumes[:top_n]


def apply_diversity_filter(perfumes: List[Dict], top_n: int) -> List[Dict]:
    """
    Ensure diversity in scent types among recommendations.
    
    GOAL:
    -----
    Prevent recommending 10 similar perfumes. Instead, provide variety
    across different scent types while maintaining high scores.
    
    ALGORITHM:
    ----------
    1. Iterate through perfumes (sorted by score)
    2. For each perfume, check its scent type
    3. If this scent type is underrepresented, add to selected
    4. Limit max perfumes per scent type (top_n / 5, minimum 2)
    5. Fill remaining slots with highest-scored perfumes
    
    Args:
        perfumes: List of scored perfumes (pre-sorted by ml_score)
        top_n: Number of perfumes to return
    
    Returns:
        Diverse subset of perfumes (still sorted by score)
    
    Example:
        >>> perfumes = [p1, p2, p3, ...]  # 20 perfumes, many similar
        >>> diverse = apply_diversity_filter(perfumes, top_n=10)
        >>> # Result: 10 perfumes with variety in scent types
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
    
    GOAL:
    -----
    Make ML recommendations interpretable and trustworthy by explaining
    the reasoning behind each recommendation.
    
    METHODOLOGY:
    ------------
    1. Analyze user's inventory to find most common accords
    2. Check if recommended perfume shares these accords
    3. Generate explanation based on matching/complementing characteristics
    4. Add confidence level based on score
    
    Args:
        perfume: Recommended perfume
        user_inventory: User's owned perfumes
        score: ML probability score (0-1)
    
    Returns:
        Human-readable explanation string
    
    Example:
        >>> explanation = generate_ml_explanation(perfume, inventory, 0.87)
        >>> print(explanation)
        "Highly recommended - Matches your preference for floral, vanilla scents (87% match)"
    """
    # Analyze user's preferences from inventory
    all_user_accords = []
    for inv_perfume in user_inventory:
        all_user_accords.extend(inv_perfume.get('main_accords', []))
    
    # Get most common accords in user's collection (top 3)
    accord_counter = Counter([a.lower() for a in all_user_accords if isinstance(a, str)])
    top_user_accords = [accord for accord, _ in accord_counter.most_common(3)]
    
    # Find matching accords in recommended perfume
    perfume_accords = [a.lower() for a in perfume.get('main_accords', []) if isinstance(a, str)]
    matching_accords = [a for a in perfume_accords if a in top_user_accords]
    
    # Build explanation based on matches
    if matching_accords:
        accord_text = ", ".join(matching_accords[:2])
        explanation = f"Matches your preference for {accord_text} scents"
    else:
        # No direct matches - complementary recommendation
        explanation = f"Complements your collection with {perfume_accords[0] if perfume_accords else 'unique'} notes"
    
    # Add confidence level based on score
    if score >= 0.8:
        confidence = "Highly recommended"
    elif score >= 0.7:
        confidence = "Strong match"
    else:
        confidence = "Good match"
    
    return f"{confidence} - {explanation} ({int(score * 100)}% match)"


# ============================================================================
# MODEL INSIGHTS & ANALYTICS
# ============================================================================

def get_model_insights(user_inventory: List[Dict]) -> Dict:
    """
    Get insights about the trained ML model and user preferences.
    
    This function provides analytics about:
    - Inventory size and training readiness
    - Model type being used
    - User's top perfume preferences
    - Diversity of user's collection
    
    Args:
        user_inventory: User's owned perfumes
    
    Returns:
        Dictionary with model insights and statistics:
        {
            'inventory_size': int,
            'can_train_model': bool,
            'model_type': str,
            'top_preferences': [{'name': str, 'count': int}, ...],
            'diversity_score': float (0-1)
        }
    
    Example:
        >>> insights = get_model_insights(user_inventory)
        >>> print(f"Can train model: {insights['can_train_model']}")
        >>> print(f"Top preferences: {insights['top_preferences']}")
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
    
    # Analyze user preferences from inventory
    all_accords = []
    scent_types = []
    for perfume in user_inventory:
        all_accords.extend(perfume.get('main_accords', []))
        scent_types.append(perfume.get('scent_type', 'Fresh'))
    
    # Get top 5 most common accords
    accord_counter = Counter([a.lower() for a in all_accords if isinstance(a, str)])
    insights['top_preferences'] = [
        {'name': accord.title(), 'count': count} 
        for accord, count in accord_counter.most_common(5)
    ]
    
    # Calculate diversity score (unique scent types / total perfumes)
    if scent_types:
        insights['diversity_score'] = len(set(scent_types)) / len(scent_types)
    
    return insights


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the ML recommendation system.
    
    This demonstrates how to use the main functions in a real scenario.
    """
    
    # Example perfume data (would come from API in real application)
    example_perfume = {
        'id': 'p123',
        'name': 'Example Perfume',
        'brand': 'Example Brand',
        'main_accords': ['floral', 'sweet', 'vanilla'],
        'seasonality': {'Winter': 4, 'Fall': 3, 'Spring': 2, 'Summer': 1},
        'occasion': {'Day': 2, 'Night': 5},
        'longevity': 'long lasting',
        'sillage': 'strong',
        'gender': 'Female',
        'price': 120,
        'scent_type': 'Oriental'
    }
    
    # Extract features
    print("=== FEATURE EXTRACTION ===")
    features = extract_perfume_features(example_perfume)
    print(f"Feature vector shape: {features.shape}")
    print(f"First 10 features: {features[:10]}")
    
    # Simulate user inventory and catalog
    print("\n=== SIMULATION ===")
    print("In real usage:")
    print("1. User adds perfumes to inventory")
    print("2. System trains ML model: train_ml_model(inventory, all_perfumes)")
    print("3. System generates recommendations: get_ml_recommendations(inventory, all_perfumes)")
    print("4. User sees personalized perfume suggestions with explanations")
    
    print("\n=== CONFIGURATION ===")
    print(f"Model Type: {ML_CONFIG['model_type']}")
    print(f"Min Inventory Size: {ML_CONFIG['min_inventory_size']}")
    print(f"Negative Samples Ratio: {ML_CONFIG['negative_samples_ratio']}")
    print(f"Min Recommendation Probability: {ML_CONFIG['min_recommendation_probability']}")


