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
     - Use APIs like Yahoo Finance.
   - **Sentiment Data:**
     - Scrape news articles or social media posts using NewsAPI.
     - Ensure data is collected over the same period as stock prices.
2. **Data Preprocessing:**
   - **Stock Price Data:**
     - Handle missing values, normalize data, create features like moving averages and daily returns.
   - **Sentiment Data:**
     - Clean text data, perform tokenization, stemming/lemmatization.
     - Calculate sentiment scores using VADER and TextBlob.

**Deliverables:**
- Cleaned and preprocessed datasets for stock prices and sentiment analysis.

### Phase 2: Feature Engineering
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
- Document findings and results.
- Prepare a research paper and presentation detailing the methodology, results, and conclusions.

## How to Open and Run the Project

### Prerequisites
Before you begin, ensure you have the following installed on your system:
- **Python 3.7+**
- **Jupyter Notebook** or **JupyterLab**
- **Git**

### Setup Instructions

1. **Clone the Repository**
   - Open your terminal or command prompt.
   - Clone this repository by running the following command:
     ```bash
     git clone https://github.com/akinahomwabella/Stock_Price_with_ML.git
     ```
   - Navigate to the project directory:
     ```bash
     cd Stock_Price_with_ML
     ```

2. **Create and Activate a Virtual Environment (Optional but Recommended)**
   - Create a virtual environment:
     ```bash
     python -m venv venv
     ```
   - Activate the virtual environment:
     - On Windows:
       ```bash
       venv\Scripts\activate
       ```
     - On macOS/Linux:
       ```bash
       source venv/bin/activate
       ```

3. **Install Required Dependencies**
   - Install the required Python packages by running:
     ```bash
     pip install -r requirements.txt
     ```

4. **Run Jupyter Notebook**
   - Start Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   - In the browser window that opens, navigate to the project directory and open the notebook file(s) provided (e.g., `stock_prediction.ipynb`).

### Running the Project

1. **Data Collection:**
   - Run the cells in the notebook for data collection to fetch the historical stock prices and financial news articles.
   - Ensure you have API keys set up for any external data sources (e.g., Yahoo Finance, NewsAPI).

2. **Data Preprocessing and Feature Engineering:**
   - Follow the instructions in the notebook to preprocess the data and perform feature engineering.
   - Run each cell in sequence to ensure the datasets are prepared correctly.

3. **Model Training and Evaluation:**
   - Proceed to train the models by running the respective sections in the notebook.
   - Evaluate the model performance using the metrics provided.

4. **View Results:**
   - Examine the model outputs and visualizations generated within the notebook.
   - Make any necessary adjustments or optimizations as described.
     

![image](https://github.com/user-attachments/assets/2037d6e4-e9a8-4300-ac75-b2890a0c2ee7)

![image](https://github.com/user-attachments/assets/165be9cc-7c55-457a-ae72-2bf66c618b50)
![image](https://github.com/user-attachments/assets/8d373f2d-12f5-41a7-9025-85af10aa6a32)



![image](https://github.com/user-attachments/assets/3c69ebbf-b8f9-471e-ae3f-ec1a788c194d)

### Additional Not

- If you encounter any issues with missing dependencies, try reinstalling them using `pip install <package_name>`.
- To deactivate the virtual environment, simply run `deactivate` in your terminal.

### Contribution
If you wish to contribute to the project, feel free to create a pull request or open an issue on the GitHub repository.

