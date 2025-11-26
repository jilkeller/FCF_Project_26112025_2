"""
Simple test script to verify ML implementation
Run with: python test_ml.py
"""

import numpy as np
from typing import List, Dict

# Mock perfume data for testing
def create_test_perfume(name: str, accords: List[str], longevity: str = 'moderate', 
                       sillage: str = 'moderate', gender: str = 'Unisex', price: float = 100.0) -> Dict:
    """Create a test perfume with specified characteristics"""
    return {
        'id': f'test_{name.lower().replace(" ", "_")}',
        'name': name,
        'brand': 'Test Brand',
        'price': price,
        'size': '50ml',
        'gender': gender,
        'scent_type': accords[0].title() if accords else 'Fresh',
        'description': 'Test perfume',
        'image_url': 'https://via.placeholder.com/300x400',
        'top_notes': [{'name': 'Bergamot', 'imageUrl': ''}],
        'heart_notes': [{'name': 'Jasmine', 'imageUrl': ''}],
        'base_notes': [{'name': 'Musk', 'imageUrl': ''}],
        'main_accords': accords,
        'seasonality': {'Winter': 3, 'Fall': 3, 'Spring': 3, 'Summer': 3},
        'occasion': {'Day': 3, 'Night': 3},
        'longevity': longevity,
        'sillage': sillage
    }

def test_feature_extraction():
    """Test that feature extraction produces correct dimensions"""
    print("Testing feature extraction...")
    
    # Import the function from scentify
    import sys
    sys.path.insert(0, '.')
    from scentify import extract_perfume_features
    
    test_perfume = create_test_perfume(
        'Test Floral', 
        ['floral', 'sweet', 'powdery'],
        longevity='long lasting',
        sillage='strong',
        gender='Female',
        price=150.0
    )
    
    features = extract_perfume_features(test_perfume)
    
    # Should have 40 features (30 accords + 4 seasons + 2 occasions + longevity + sillage + gender + price)
    expected_features = 40
    
    assert len(features) == expected_features, f"Expected {expected_features} features, got {len(features)}"
    assert isinstance(features, np.ndarray), "Features should be numpy array"
    assert features.dtype == np.float64 or features.dtype == np.float32, "Features should be float"
    
    # Check that values are in reasonable range (0-1 mostly)
    assert np.all(features >= 0), "All features should be non-negative"
    assert np.all(features <= 1.5), "Most features should be normalized to 0-1 range"
    
    print(f"✅ Feature extraction test passed! Got {len(features)} features as expected.")
    print(f"   Feature range: [{features.min():.3f}, {features.max():.3f}]")
    return True

def test_training_dataset():
    """Test training dataset construction"""
    print("\nTesting training dataset construction...")
    
    from scentify import build_training_dataset, ML_CONFIG
    
    # Create test inventory (owned perfumes)
    user_inventory = [
        create_test_perfume('Owned 1', ['floral', 'sweet']),
        create_test_perfume('Owned 2', ['woody', 'spicy']),
        create_test_perfume('Owned 3', ['fresh', 'citrus']),
    ]
    
    # Create test catalog (all perfumes)
    all_perfumes = user_inventory + [
        create_test_perfume('Not Owned 1', ['oriental', 'amber']),
        create_test_perfume('Not Owned 2', ['green', 'herbal']),
        create_test_perfume('Not Owned 3', ['gourmand', 'vanilla']),
        create_test_perfume('Not Owned 4', ['leather', 'tobacco']),
        create_test_perfume('Not Owned 5', ['aquatic', 'marine']),
    ]
    
    X, y = build_training_dataset(user_inventory, all_perfumes)
    
    # Check dimensions
    assert X.shape[0] == y.shape[0], "X and y should have same number of samples"
    assert X.shape[1] == 40, "Each sample should have 40 features"
    
    # Check labels
    num_positive = np.sum(y == 1)
    num_negative = np.sum(y == 0)
    
    assert num_positive == len(user_inventory), f"Should have {len(user_inventory)} positive samples"
    assert num_negative > 0, "Should have some negative samples"
    
    expected_negatives = len(user_inventory) * ML_CONFIG['negative_samples_ratio']
    assert num_negative <= len(all_perfumes) - len(user_inventory), "Negatives should not exceed available"
    
    print(f"✅ Training dataset test passed!")
    print(f"   Positive samples: {num_positive}")
    print(f"   Negative samples: {num_negative}")
    print(f"   Feature matrix shape: {X.shape}")
    return True

def test_model_training():
    """Test that model can be trained"""
    print("\nTesting model training...")
    
    from scentify import train_ml_model
    
    # Create test inventory (need at least 2 perfumes)
    user_inventory = [
        create_test_perfume('Owned 1', ['floral', 'sweet', 'powdery']),
        create_test_perfume('Owned 2', ['woody', 'spicy', 'amber']),
        create_test_perfume('Owned 3', ['fresh', 'citrus', 'aquatic']),
    ]
    
    # Create larger catalog
    all_perfumes = user_inventory + [
        create_test_perfume(f'Catalog {i}', 
                          np.random.choice(['floral', 'woody', 'fresh', 'oriental', 'citrus'], 
                                         size=3, replace=False).tolist())
        for i in range(20)
    ]
    
    result = train_ml_model(user_inventory, all_perfumes)
    
    assert result is not None, "Model training should succeed"
    
    model, scaler, feature_importance = result
    
    assert model is not None, "Should return trained model"
    assert scaler is not None, "Should return fitted scaler"
    
    # Test prediction capability
    test_perfume = create_test_perfume('Test Predict', ['floral', 'sweet'])
    from scentify import extract_perfume_features
    features = extract_perfume_features(test_perfume)
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(features_scaled)[0][1]
        assert 0 <= prob <= 1, "Probability should be between 0 and 1"
        print(f"✅ Model training test passed!")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Test prediction probability: {prob:.3f}")
    else:
        pred = model.predict(features_scaled)[0]
        print(f"✅ Model training test passed!")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Test prediction: {pred}")
    
    return True

def test_recommendations():
    """Test full recommendation pipeline"""
    print("\nTesting recommendation generation...")
    
    from scentify import get_ml_recommendations, ML_CONFIG
    
    # Temporarily lower threshold for testing
    original_threshold = ML_CONFIG['min_recommendation_probability']
    ML_CONFIG['min_recommendation_probability'] = 0.3
    
    # Create user inventory with specific preferences (likes floral and sweet)
    user_inventory = [
        create_test_perfume('Owned 1', ['floral', 'sweet', 'powdery'], price=120),
        create_test_perfume('Owned 2', ['floral', 'rose', 'vanilla'], price=150),
        create_test_perfume('Owned 3', ['sweet', 'gourmand', 'vanilla'], price=100),
    ]
    
    # Create catalog with some similar perfumes and some different ones
    all_perfumes = user_inventory + [
        # Similar to user's taste (should score high)
        create_test_perfume('Similar 1', ['floral', 'sweet', 'fruity'], price=130),
        create_test_perfume('Similar 2', ['floral', 'powdery', 'vanilla'], price=140),
        create_test_perfume('Similar 3', ['sweet', 'gourmand', 'caramel'], price=110),
        # Different from user's taste (should score lower)
        create_test_perfume('Different 1', ['woody', 'leather', 'tobacco'], price=200),
        create_test_perfume('Different 2', ['fresh', 'aquatic', 'marine'], price=90),
        create_test_perfume('Different 3', ['spicy', 'oriental', 'oud'], price=250),
    ]
    
    try:
        recommendations = get_ml_recommendations(
            user_inventory, 
            all_perfumes, 
            top_n=5,
            ensure_diversity=False
        )
    finally:
        # Restore original threshold
        ML_CONFIG['min_recommendation_probability'] = original_threshold
    
    assert len(recommendations) > 0, f"Should generate some recommendations (got {len(recommendations)})"
    assert len(recommendations) <= 5, "Should not exceed requested number"
    
    # Check that recommendations have required fields
    for rec in recommendations:
        assert 'ml_score' in rec, "Each recommendation should have ml_score"
        assert 'ml_explanation' in rec, "Each recommendation should have ml_explanation"
        assert 0 <= rec['ml_score'] <= 1, "ML score should be between 0 and 1"
        assert isinstance(rec['ml_explanation'], str), "Explanation should be string"
        assert len(rec['ml_explanation']) > 0, "Explanation should not be empty"
    
    # Recommendations should be sorted by score (descending)
    scores = [r['ml_score'] for r in recommendations]
    assert scores == sorted(scores, reverse=True), "Recommendations should be sorted by score"
    
    print(f"✅ Recommendation test passed!")
    print(f"   Generated {len(recommendations)} recommendations")
    if recommendations:
        print(f"   Top recommendation: {recommendations[0]['name']} (score: {recommendations[0]['ml_score']:.3f})")
        print(f"   Explanation: {recommendations[0]['ml_explanation']}")
    
    return True

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("SCENTIFY ML IMPLEMENTATION TESTS")
    print("=" * 60)
    
    tests = [
        ("Feature Extraction", test_feature_extraction),
        ("Training Dataset", test_training_dataset),
        ("Model Training", test_model_training),
        ("Recommendations", test_recommendations),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 All tests passed! ML implementation is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

