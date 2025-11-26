# Scentify - AI-Powered Perfume Recommendation System

A Streamlit-based perfume discovery and recommendation application with **machine learning** capabilities.

## 🎯 Project Overview

Scentify is a comprehensive perfume finder application that integrates with the Fragella API to provide:
- Advanced perfume search and filtering
- Personalized questionnaire-based recommendations
- Personal perfume inventory management with analytics
- **NEW: AI-powered personalized recommendations using machine learning**

## 🚀 Key Features

### 1. **Search Engine**
- Real-time integration with Fragella API
- Advanced filtering by brand, price, scent type, gender
- Price range slider
- Multiple filter combinations

### 2. **Smart Questionnaire**
- Interactive preference discovery
- Guided scent profile creation
- Personalized match recommendations

### 3. **Perfume Inventory**
- Personal collection management
- Add/remove perfumes
- Detailed analytics and visualizations
- Scent distribution charts
- Collection insights

### 4. **🤖 AI Recommendations (NEW)**
Machine learning-powered personalized recommendations based on your collection:

#### Features:
- **Dual Algorithm Support**: Choose between Logistic Regression or Decision Tree models
- **Feature Extraction**: 40-dimensional feature vectors including:
  - 30 scent accord features (floral, woody, citrus, etc.)
  - Seasonality preferences (4 features)
  - Occasion suitability (2 features)  
  - Performance metrics (longevity, sillage)
  - Demographics and price normalization
  
- **Smart Training**: Learns from positive samples (owned perfumes) and negative samples (non-owned)
- **Probability Scoring**: Each recommendation includes a match percentage
- **Human Explanations**: AI explains why each perfume was recommended
- **Diversity Filter**: Optional scent variety enforcement
- **Configurable**: Adjust model type, number of recommendations, and match thresholds

#### How It Works:
1. Add at least 2 perfumes to your inventory
2. Navigate to "AI Recommendations"
3. Configure settings (model type, number of results, threshold)
4. Generate recommendations
5. View personalized suggestions with match scores and explanations

## 📋 Requirements

- Python 3.8+
- Fragella API key
- Dependencies (see `requirements.txt`):
  - streamlit>=1.28.0
  - pandas>=2.0.0
  - numpy>=1.24.0
  - plotly>=5.17.0
  - requests>=2.31.0
  - python-dotenv>=1.0.0
  - **scikit-learn>=1.3.0** (for ML features)

## 🛠️ Installation

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up API key
Create a `.env` file in the project root:
```
FRAGELLA_API_KEY=your_api_key_here
```

### 3. Run the application
```bash
streamlit run scentify.py
```

The app will open at `http://localhost:8501`

## 🎓 Academic Context

This project implements the machine learning specification provided for the course, including:

✅ **Personalized Learning**: Learns user preferences from owned perfumes  
✅ **Feature Engineering**: Extracts 40 numerical features from perfume characteristics  
✅ **Training Data Construction**: Positive (owned) and negative (non-owned) samples  
✅ **Multiple Models**: Logistic Regression and Decision Tree implementations  
✅ **Probability Scoring**: Outputs P(user likes perfume | features)  
✅ **Recommendation Pipeline**: Score, filter, rank, and explain suggestions  
✅ **Maximum Information Extraction**: Uses all available API accent/accord data  
✅ **Explanations**: Human-readable reasons for each recommendation  
✅ **Diversity**: Optional scent variety enforcement in results  

See `ML_IMPLEMENTATION.md` for complete technical documentation.

## 📊 Testing

Run the test suite to verify ML implementation:

```bash
python test_ml.py
```

Expected output:
```
✅ Feature extraction test passed!
✅ Training dataset test passed!
✅ Model training test passed!
✅ Recommendation test passed!
🎉 All tests passed!
```

## 📁 Project Structure

```
26.11/
├── scentify.py                  # Main application (3400+ lines)
├── requirements.txt             # Python dependencies
├── .env                         # API key (create this)
├── README.md                    # This file
├── SETUP.md                     # Detailed setup guide
├── ML_IMPLEMENTATION.md         # ML technical documentation
├── test_ml.py                   # Test suite
└── data/                        # Auto-created data directory
    ├── user_perfume_inventory.json
    ├── user_interactions.json
    ├── perfume_rankings.json
    ├── ml_model.pkl             # Trained ML model
    └── ml_scaler.pkl            # Feature scaler
```

## 🎨 UI Design

- **Color Scheme**: Pastel purple, gray, white, and black
- **Layout**: Clean, modern, card-based design
- **Responsive**: Adapts to different screen sizes
- **Interactive**: Real-time filtering and updates

## 🔧 Configuration

### ML Configuration
Located in `scentify.py`:

```python
ML_CONFIG = {
    'negative_samples_ratio': 2,          # Negatives per positive
    'min_inventory_size': 2,              # Min perfumes for ML
    'model_type': 'logistic_regression',  # Default algorithm
    'random_state': 42,                   # Reproducibility
    'min_recommendation_probability': 0.5, # Default threshold
    'diversity_threshold': 0.3            # Diversity level
}
```

### Runtime Configuration
Adjust in the UI:
- ML Model type (Logistic Regression / Decision Tree)
- Number of recommendations (5-20)
- Minimum match score (30-90%)
- Diversity filter (on/off)

## 📖 Documentation

- **`SETUP.md`**: Complete installation and setup guide
- **`ML_IMPLEMENTATION.md`**: Detailed ML technical documentation
- **`test_ml.py`**: Test suite with examples

## 🚀 Deployment

### Streamlit Cloud

1. Push to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Add `FRAGELLA_API_KEY` in Secrets settings
5. Deploy!

### Local Hosting

```bash
streamlit run scentify.py --server.port 8501
```

## 🧪 Example Usage

### Adding to Inventory
```python
# In the app UI:
1. Go to "Perfume Inventory"
2. Click "Add Perfume"
3. Search for perfume
4. Click "Add" on desired perfume
```

### Getting ML Recommendations
```python
# In the app UI:
1. Ensure you have 2+ perfumes in inventory
2. Go to "AI Recommendations"
3. (Optional) Adjust configuration
4. Click "Generate AI Recommendations"
5. View personalized results with match scores
```

## 🎯 ML Model Performance

The ML system typically achieves:
- **Training time**: < 1 second
- **Inference time**: < 1 second for full catalog
- **Model size**: < 100KB
- **Feature dimensionality**: 40 features
- **Scalability**: Efficient up to 10,000+ perfumes

## 🔄 Data Flow

```
User Inventory → Feature Extraction → Training Dataset
    ↓
Positive Samples + Negative Samples
    ↓
ML Model Training (Logistic Regression / Decision Tree)
    ↓
Score All Candidates → Apply Threshold → Diversity Filter
    ↓
Top N Recommendations with Explanations
```

## 🤝 Contributing

This is an academic project. For questions or improvements:
1. Review the documentation files
2. Run the test suite
3. Check implementation details in code comments

## 📄 License

Academic project for CS coursework.

## 🙏 Acknowledgments

- **Fragella API**: Perfume data source
- **Streamlit**: Web framework
- **scikit-learn**: ML algorithms
- **Plotly**: Interactive visualizations

## 📞 Support

See documentation files:
- Installation issues → `SETUP.md`
- ML questions → `ML_IMPLEMENTATION.md`
- Testing → `test_ml.py`

---

**Version**: 2.0 (with ML capabilities)  
**Last Updated**: November 2025  
**Status**: ✅ All tests passing, production ready


