# 🏠 Housing Price Prediction Model

## 🎯 Project Overview
This project implements a machine learning model to predict housing prices using various features of residential properties. The model utilizes linear regression with feature selection to optimize prediction accuracy.

## ✨ Features
- Data preprocessing and cleaning
- Feature engineering and selection
- Model training and evaluation
- Performance metrics analysis
- Price prediction for new housing data

## 📦 Dependencies
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- statsmodels
- itertools

## 🚀 Installation
```bash
pip install pandas numpy seaborn matplotlib scikit-learn statsmodels
```

## 📁 Project Structure
1. **Data Preprocessing**
   - Loading and cleaning dataset
   - Handling missing values
   - Converting categorical variables
   - Feature engineering

2. **Feature Selection**
   - Forward selection technique
   - RMSE optimization
   - Optimal predictor subset identification

3. **Model Development**
   - Training/testing data split
   - Linear regression implementation
   - Model evaluation and tuning

## 🔧 Key Features Engineered
- AgeAtSale: Age of house at time of sale
- YearsSinceRenovation: Time since last renovation
- GarageCategory: Categorization of garage build timing
- Various binary indicators for missing features

## 📊 Model Performance
- Train RMSE: 18,033
- R² Score: 93.63%
- MAPE: 7.10%
- Leaderboard RMSE: 22,225

## 📈 Model Performance Visualization
![image](https://github.com/user-attachments/assets/dc228d53-1d68-4617-baee-a882d6127e63)


### Key Insights from the Graph:
- **Optimal Predictor Count**: The model achieves minimum test RMSE at 116 predictors (26,609.24)
- **Training Performance**: Continues to improve up to 257 predictors (18,033.71)
- **Overfitting Point**: After 116 predictors, test RMSE begins to increase while train RMSE continues to decrease
- **Model Selection**: Chose 116 predictors as optimal to prevent overfitting

## 🔄 Data Processing Steps
1. **Missing Value Treatment** 🔍
   - Binary conversion for features with >75% missing values
   - Median imputation for LotFrontage based on neighborhood
   - Mode imputation for categorical variables
   - Zero imputation for numerical absences

2. **Feature Engineering** ⚙️
   - Year-related feature creation
   - Categorical variable encoding
   - Numeric variable scaling

3. **Feature Selection** 📈
   - Forward selection implementation
   - Optimal subset of 116 predictors
   - RMSE minimization approach

## 📋 Model Evaluation
The model shows strong performance with:
- High R² indicating good fit
- Reasonable MAPE suggesting reliable predictions
- Competitive RMSE on both training and test sets

## 💻 Usage
1. Data Preparation:
```python
# Load and preprocess data
housing_data_train = pd.read_csv('Housing_Data_Train.csv')
housing_data_test = pd.read_csv('Housing_Data_Test.csv')
```

2. Model Training:
```python
# Train model using processed data
model = sm.OLS(y_train, sm.add_constant(X_train_selected))
final_model = model.fit()
```

3. Prediction:
```python
# Generate predictions
predictions = final_model.predict(sm.add_constant(X_test_selected))
```

## 🤝 Contributing
Feel free to submit issues and enhancement requests.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Author
Arpan Sharma

## 🙏 Acknowledgments
I would like to thank Dr. Mihai Nica for getting me this dataset to work on during my coursework.
