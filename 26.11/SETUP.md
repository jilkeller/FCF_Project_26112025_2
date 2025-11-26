# Scentify Setup Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- streamlit (web framework)
- pandas (data manipulation)
- numpy (numerical computing)
- plotly (interactive charts)
- requests (API calls)
- python-dotenv (environment variables)
- scikit-learn (machine learning)

### 2. Set Up API Key

Create a `.env` file in the project root directory:

```bash
touch .env
```

Add your Fragella API key to the `.env` file:

```
FRAGELLA_API_KEY=your_api_key_here
```

**Note**: Replace `your_api_key_here` with your actual Fragella API key.

### 3. Create Data Directory

The app will automatically create this, but you can create it manually:

```bash
mkdir data
```

This folder will store:
- User perfume inventory
- ML models
- Interaction history
- Ranking data

## Running the Application

### Local Development

```bash
streamlit run scentify.py
```

The app will open in your browser at `http://localhost:8501`

### Streamlit Cloud Deployment

1. Push code to GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select your repository and branch
5. Set `scentify.py` as the main file
6. Add secrets in Streamlit Cloud dashboard:
   - Go to App settings → Secrets
   - Add: `FRAGELLA_API_KEY = "your_api_key_here"`
7. Deploy!

## Using the ML Features

### Minimum Requirements

- At least **2 perfumes** in your inventory to use AI recommendations
- The more perfumes you add, the better the recommendations

### How to Get Started

1. **Add Perfumes to Inventory**
   - Go to "Perfume Inventory" from home page
   - Click "Add Perfume" 
   - Search and add at least 2 perfumes you own

2. **Generate AI Recommendations**
   - Return to home page
   - Click "Get AI Recommendations"
   - Configure settings (optional)
   - Click "Generate AI Recommendations"

3. **Review Recommendations**
   - View match scores (50-100%)
   - Read AI explanations
   - View details or add to inventory

## Configuration Options

### ML Model Settings

In the AI Recommendations section, you can adjust:

- **ML Model**: Choose between:
  - Logistic Regression (default, interpretable)
  - Decision Tree (handles complex patterns)

- **Number of Recommendations**: 5-20 perfumes

- **Minimum Match Score**: 0.3-0.9 (higher = more selective)

- **Ensure Diverse Scent Types**: Toggle for variety

### Global Configuration

Edit these in `scentify.py` if needed:

```python
ML_CONFIG = {
    'negative_samples_ratio': 2,          # Training data balance
    'min_inventory_size': 2,              # Min perfumes for ML
    'model_type': 'logistic_regression',  # Default algorithm
    'random_state': 42,                    # Reproducibility
    'min_recommendation_probability': 0.5, # Default threshold
    'diversity_threshold': 0.3             # Diversity level
}
```

## Troubleshooting

### "FRAGELLA_API_KEY not found"
- Make sure `.env` file exists in project root
- Check that the file contains `FRAGELLA_API_KEY=your_key`
- Restart the Streamlit app after creating `.env`

### "You need at least 2 perfumes in your inventory"
- Add more perfumes to your inventory first
- Go to Perfume Inventory → Add Perfume

### No recommendations generated
- Lower the "Minimum Match Score" threshold
- Make sure your inventory has diverse perfumes
- Try a different ML model

### Import errors
- Run `pip install -r requirements.txt` again
- Make sure you're using Python 3.8+
- Check that scikit-learn installed correctly: `python -c "import sklearn; print(sklearn.__version__)"`

### Slow performance
- The first API load may take a few seconds
- Model training is fast (<1 second typically)
- If very slow, check your internet connection

## File Structure

```
26.11/
├── scentify.py              # Main application
├── requirements.txt         # Dependencies
├── .env                     # API key (create this)
├── SETUP.md                # This file
├── ML_IMPLEMENTATION.md    # ML technical docs
└── data/                   # Auto-created
    ├── user_perfume_inventory.json
    ├── user_interactions.json
    ├── perfume_rankings.json
    ├── ml_model.pkl
    └── ml_scaler.pkl
```

## Features Overview

### 1. Search
- Filter by brand, price, scent type, gender
- Real-time Fragella API integration
- Advanced filtering options

### 2. Questionnaire
- Personalized preference discovery
- Guided scent profile creation
- Matched recommendations

### 3. Perfume Inventory
- Personal collection management
- Detailed analytics
- Scent distribution charts
- Collection insights

### 4. AI Recommendations (NEW)
- Machine learning-powered suggestions
- Personalized to your collection
- Match scores and explanations
- Configurable algorithms
- Diversity optimization

## Support

For issues or questions:
1. Check this setup guide
2. Review ML_IMPLEMENTATION.md for technical details
3. Verify all dependencies are installed
4. Check Streamlit logs for error messages

## Academic Context

This implementation follows the machine learning specifications provided for the course project, including:
- Feature extraction from perfume characteristics
- Positive/negative training sample generation
- Multiple ML algorithms (Logistic Regression, Decision Trees)
- Personalized recommendation scoring
- Diversity filtering
- Human-readable explanations

See ML_IMPLEMENTATION.md for complete academic documentation.


