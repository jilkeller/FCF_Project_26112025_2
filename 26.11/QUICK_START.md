# Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Create `.env` File
```bash
echo "FRAGELLA_API_KEY=your_api_key_here" > .env
```
*(Replace `your_api_key_here` with your actual API key)*

### 3. Run the App
```bash
streamlit run scentify.py
```

---

## 🎯 Using AI Recommendations

1. **Add Perfumes** → Go to "Perfume Inventory" and add at least 2 perfumes
2. **Get Recommendations** → Click "Get AI Recommendations" on home page
3. **Generate** → Click "Generate AI Recommendations" button
4. **Enjoy** → View your personalized matches!

---

## 🧪 Test the Implementation

```bash
python test_ml.py
```

Expected: ✅ All 4 tests passing

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview |
| `SETUP.md` | Detailed setup |
| `ML_IMPLEMENTATION.md` | ML technical docs |
| `LECTURER_SUMMARY.md` | Academic summary |
| `IMPLEMENTATION_SUMMARY.txt` | Complete summary |
| `test_ml.py` | Test suite |

---

## ⚙️ Configuration

Adjust ML settings in the UI:
- **Model Type**: Logistic Regression or Decision Tree
- **Number of Results**: 5-20 recommendations
- **Match Threshold**: 30-90% minimum
- **Diversity**: Toggle on/off

---

## 🎓 For Submission

Give your lecturer:
1. `LECTURER_SUMMARY.md` - Academic overview
2. `ML_IMPLEMENTATION.md` - Technical details
3. Demo the working app
4. Show test results from `test_ml.py`

---

## ❓ Troubleshooting

**"API key not found"**  
→ Create `.env` file with your Fragella API key

**"Need 2+ perfumes"**  
→ Add more perfumes to your inventory first

**"No recommendations"**  
→ Lower the minimum match score threshold

**"Import errors"**  
→ Run `pip install -r requirements.txt` again

---

## 📊 Project Stats

- **Lines of Code**: ~800 new lines
- **ML Features**: 40-dimensional vectors
- **Models**: 2 (Logistic Regression, Decision Tree)
- **Tests**: 4/4 passing ✅
- **Documentation**: 6 files

---

**Ready to go!** 🎉


