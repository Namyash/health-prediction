# Health Prediction System

An interactive machine learning application for predicting common health conditions based on patient data. This repository contains a user-friendly Streamlit web application and comprehensive Jupyter notebook analysis for health risk assessment.

## Overview

This project provides predictive models for three major health conditions:
- **Heart Disease** - Cardiovascular risk assessment
- **Lung Cancer** - Respiratory health screening
- **Diabetes** - Metabolic disorder prediction

The system combines an interactive web interface ([app.py](app.py)) with detailed exploratory data analysis and model development ([health.ipynb](health.ipynb)).

## Features

### 🏥 Multi-Disease Prediction
- Select from three different health prediction models
- Each model specialized for specific health conditions
- Real-time prediction results with confidence scores

### 📊 Advanced Machine Learning
- **Logistic Regression** for lung cancer prediction
- **Neural Network** for heart disease assessment
- **Random Forest** for diabetes risk evaluation

### 🔧 Data Processing
- Automated data preprocessing with scaling and encoding
- Feature normalization using scikit-learn scalers
- Input validation and error handling

### 💡 Explainable Predictions
- Simple, intuitive user interface
- Clear prediction results with risk levels
- Educational content in 'About' sections

### 📚 Educational Content
- Health condition information
- Prevention tips and lifestyle recommendations
- Medical insights for better health awareness

## Project Structure

```
health-prediction/
├── app.py                              # Main Streamlit application
├── health.ipynb                        # Jupyter notebook with analysis
├── requirements.txt                    # Python dependencies
│
├── datasets/
│   ├── heart.csv                      # Heart disease dataset
│   ├── lung.csv                       # Lung cancer dataset
│   ├── diabetes.csv                   # Diabetes dataset
│   └── kidney.csv                     # Additional kidney disease data
│
├── models/
│   ├── neural_network_heart_model.pkl       # Heart disease neural network
│   ├── logistic_regression_lung_model.pkl   # Lung cancer logistic regression
│   └── random_forest_diabetes_model.pkl     # Diabetes random forest
│
└── scalers/
    ├── heart_scaler*.pkl              # Heart disease data scalers
    ├── lung_scaler*.pkl               # Lung cancer data scalers
    └── db_scaler*.pkl                 # Diabetes data scalers
```

## Technologies Used

- **Python** - Core programming language
- **Streamlit** - Web application framework
- **scikit-learn** - Machine learning algorithms and preprocessing
- **joblib** - Model serialization and loading
- **numpy** - Numerical computations
- **pandas** - Data manipulation and analysis
- **Jupyter** - Interactive development and analysis

## Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Namyash/health-prediction.git
   cd health-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Access the app**
   - Open your web browser
   - Navigate to `http://localhost:8501`
   - Start making health predictions!

## Usage

1. **Select a Health Model**
   - Choose from Heart Disease, Lung Cancer, or Diabetes prediction
   - Each model is optimized for its specific condition

2. **Enter Patient Information**
   - Fill in the required health parameters
   - All inputs are validated for accuracy
   - Follow the helpful input guidelines

3. **Get Predictions**
   - Receive instant risk assessment
   - View confidence levels and recommendations
   - Access educational content about the condition

4. **Explore the Analysis**
   - Open [health.ipynb](health.ipynb) to see detailed model development
   - Understand the data preprocessing steps
   - Review model performance metrics

## File Links

- 🚀 **[App Source Code](app.py)** - Main Streamlit application
- 📊 **[Jupyter Analysis](health.ipynb)** - Comprehensive data analysis and model development
- 📋 **[Requirements](requirements.txt)** - Python package dependencies
- 💾 **[Heart Dataset](heart.csv)** - Heart disease training data
- 🫁 **[Lung Dataset](lung.csv)** - Lung cancer training data
- 🩺 **[Diabetes Dataset](diabetes.csv)** - Diabetes training data

## Model Performance

Each model has been trained and validated on medical datasets:

- **Heart Disease Model**: Neural network with optimized architecture
- **Lung Cancer Model**: Logistic regression with feature selection
- **Diabetes Model**: Random forest with ensemble learning

Detailed performance metrics and validation results are available in the [Jupyter notebook](health.ipynb).

## Contributing

We welcome contributions to improve the health prediction system! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Experimentation

🧪 **Try it locally!** 

This project is designed for experimentation and learning:
- Modify the models and see how predictions change
- Add new features to the dataset
- Experiment with different algorithms
- Customize the user interface
- Explore the educational content

## Disclaimer

⚠️ **Important**: This application is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

## Author

**Namyash** - [GitHub Profile](https://github.com/Namyash)

- 🔗 Project Repository: [health-prediction](https://github.com/Namyash/health-prediction)
- 📧 Feel free to reach out for questions or collaboration opportunities
- 🌟 If you found this project helpful, please consider giving it a star!

---

*Built with ❤️ for healthcare innovation and machine learning education*
