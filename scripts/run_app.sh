#!/bin/bash
# California Housing Price Predictor Launch Script

echo "🏠 Starting California Housing Price Predictor..."
echo "================================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check if models exist
if [ ! -d "models/trained_models" ] || [ -z "$(ls -A models/trained_models 2>/dev/null)" ]; then
    echo "⚠️ No trained models found!"
    echo "💡 Please complete Phase 4 (Model Training) first:"
    echo "   1. Run: jupyter notebook notebooks/03_model_development_training.ipynb"
    echo "   2. Or run: PYTHONPATH=. python src/model_training.py"
    echo ""
fi

# Start Streamlit app
echo "🚀 Launching California Housing Price Predictor..."
echo "📍 App will be available at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

streamlit run app/streamlit_app.py --server.address localhost --server.port 8501

echo ""
echo "👋 California Housing Price Predictor stopped."