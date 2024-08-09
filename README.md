# Stock_Price_with_ML

## Project Overview
This project aims to predict stock market trends by integrating historical stock price data with sentiment analysis of financial news. The project is divided into multiple phases, covering data collection, feature engineering, model training, evaluation, and preparation of research findings.

## Project Phases and Timeline

### Phase 1: Data Collection and Preprocessing 
**Objectives:**
- Collect historical stock price data and financial news articles.
- Preprocess the data for analysis and modeling.

**Tasks:**
1. **Data Collection:**
    - **Stock Price Data:**
        - Apple (AAPL)
        - Tesla (TSLA)
        - JPMorgan (JPM)
        - Pfizer (PFE)
        - ExxonMobil (XOM)
        - Use APIs like Yahoo Finance, Alpha Vantage, or Quandl.
    - **Sentiment Data:**
        - Scrape news articles or social media posts using BeautifulSoup, Scrapy, or NewsAPI.
        - Ensure data is collected over the same period as stock prices.
2. **Data Preprocessing:**
    - **Stock Price Data:**
        - Handle missing values, normalize data, create features like moving averages and daily returns.
    - **Sentiment Data:**
        - Clean text data, perform tokenization, stemming/lemmatization.
        - Calculate sentiment scores using VADER, TextBlob, or Transformers.

**Deliverables:**
- Cleaned and preprocessed datasets for stock prices and sentiment analysis.

### Phase 2: Feature Engineering (Week 2)
**Objectives:**
- Extract meaningful features for modeling.

**Tasks:**
1. **Financial Features:**
    - Calculate technical indicators (e.g., Moving Averages, RSI, MACD).
    - Compute features like volatility and volume.
2. **Sentiment Features:**
    - Aggregate sentiment scores by day, week, or month.
    - Explore sentiment analysis techniques (e.g., VADER, BERT).
    - Consider additional NLP features like word embeddings.
3. **Temporal Features:**
    - Include time-related features (e.g., day of the week, month).

**Deliverables:**
- Feature matrix combining financial, sentiment, and temporal features.

### Phase 3: Model Implementation and Training 
**Objectives:**
- Implement and train machine learning models.

**Tasks:**
1. **Model Selection:**
    - Start with Linear Regression and Decision Trees.
    - Progress to Random Forest, XGBoost, and Neural Networks (LSTM, GRU).
2. **Model Training:**
    - Split data into training, validation, and test sets.
    - Train models, tune hyperparameters, and implement time series cross-validation.
3. **Modeling Considerations:**
    - Handle time-series data (e.g., data stationarity, sequence length for LSTM).
    - Use ensemble methods if needed.

**Deliverables:**
- Trained models and initial performance metrics (e.g., RMSE, MAE, accuracy).

### Phase 4: Model Evaluation and Optimization 
**Objectives:**
- Evaluate and optimize model performance.

**Tasks:**
1. **Evaluation Metrics:**
    - Calculate RMSE, MAE, MAPE, and accuracy.
    - Assess model generalization using the test set.
2. **Model Optimization:**
    - Fine-tune hyperparameters using Grid Search or Bayesian Optimization.
    - Perform feature selection and engineering.
    - Consider model integration or ensemble techniques.
3. **Error Analysis:**
    - Analyze model errors and investigate improvements.

**Deliverables:**
- Finalized models with optimized parameters and detailed evaluation metrics.

### Phase 5: Research Paper and Presentation 
**Objectives:**
- Document findings
**Linear Regression**
![519a9dd0-cfd9-42a0-b5f1-c00e262c5e97](https://github.com/user-attachments/assets/51666d3c-37ce-4142-b7f8-dec9e67ec436)
This provides a baseline performance measure. Since the MSE is high(98%), it indicates that linear relationships may not capture the complexity of the data.
Linear regression may not handle complex, non-linear relationships effectively. If the scatter plot shows significant deviations from the ideal line, the model might not be suitable for capturing intricate patterns in the data.
**Decision Tree Model**
Decision trees can fit complex patterns, but they may also overfit the training data, leading to poor generalization on test data. The scatter plot helps in assessing how well the model generalizes.
Decision trees provide insight into feature importance, which helps in understanding which features are most influential in predictions.
**XGBoost Model**
  ![519a9dd0-cfd9-42a0-b5f1-c00e262c5e97](https://github.com/user-attachments/assets/a24d69f5-49a5-485e-965b-618638565c01)

  Generally, XGBoost outperforms simpler models like linear regression and decision trees due to its ability to model complex relationships.
  GridSearchCV results provide the best hyperparameters, indicating how to fine-tune the model for better performance. The visualization of feature importance helps prioritize key features in the model.The MSE for this model was 0.03 after hyper-tuning.
  
**LSTM**
![7245f7c7-f69f-4b1d-a675-9b899101bd90](https://github.com/user-attachments/assets/088bf487-8232-4728-9187-ce459a31e774)
Sequential Data Handling: LSTM’s performance highlights its ability to capture temporal dependencies, which is crucial for forecasting based on historical data.
Error Analysis: Residual and error distribution plots help identify patterns in prediction errors and assess if the model captures temporal patterns effectively or if there are underlying issues. 

![e1d3bc72-dcef-4c4c-b056-507d18556e66](https://github.com/user-attachments/assets/f4f4b116-7d08-4dec-bd8a-dd3f009d87b4)
Looking at this it is evident that the model tends to overpredict more frequently than it underpredicts, as indicated by the higher concentration of negative errors. The spread of errors suggests some inconsistency in the model's performance, with predictions deviating from the actual values across a notable range.To improve the model, it may be necessary to address this bias towards overprediction, potentially by fine-tuning the model, exploring additional features, or adjusting the loss function to minimize these errors. 

This prediction using a combination of financial data and sentiment analysis, showcasing advanced skills in data science and machine learning.

## Contact
- **Email:** akinahom.wabella@mnsu.edu
- **GitHub:** [akinahomwabella](https://github.com/akinahomwabella)
