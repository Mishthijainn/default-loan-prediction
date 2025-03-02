#Credit Risk Prediction - Loan Default Probability
Overview
This project predicts credit risk by evaluating the probability of loan default using machine learning models such as Random Forest, AdaBoost, and XGBoost. It features an interactive Streamlit web app to allow easy user interaction, along with visualizations for deeper insights into the loan approval process. The project also includes a Jupyter Notebook for model training, evaluation, and comparison.

Key Features
Loan Approval Risk Prediction: Predicts the likelihood of loan defaults based on historical data.
Model Comparison: Compares the performance of different models (Random Forest, AdaBoost, XGBoost).
Hyperparameter Tuning: Optimizes the model using techniques like GridSearchCV.
Interactive Visualizations: Visual representations of data insights using pie charts, bar plots, scatter plots, and heatmaps.
Technologies Used
Machine Learning: Python, Scikit-learn, XGBoost, AdaBoost, RandomForest
Web Framework: Streamlit (for the web app)
Data Visualization: Matplotlib, Seaborn
Modeling & Evaluation: Jupyter Notebook for training and performance analysis
Project Structure
bash
Copy
Edit
/credit-risk-prediction
│
├── app.py                  # Main Streamlit app for user interaction
├── model_training.ipynb     # Jupyter notebook for model training and evaluation
├── requirements.txt         # List of required dependencies
├── data/                    # Folder containing datasets
│   └── loan_data.csv        # The dataset used for model training
├── visuals/                 # Folder for storing generated plots
│   └── plot1.png            # Example plot
└── README.md                # Project documentation
Setup Instructions
Prerequisites
Python 3.x
pip (Python package installer)
Install Dependencies
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/credit-risk-prediction.git
cd credit-risk-prediction
Create a virtual environment and activate it:

bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install the required dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Running the Streamlit Web App
Start the Streamlit app:

bash
Copy
Edit
streamlit run app.py
The app will be available at http://localhost:8501. You can use the app to interact with the model and visualize results.

Model Training & Evaluation (Jupyter Notebook)
Open the model_training.ipynb Jupyter notebook to train and evaluate the models. The notebook contains:
Preprocessing steps for the dataset.
Model training for Random Forest, AdaBoost, and XGBoost.
Hyperparameter tuning using GridSearchCV.
Model evaluation and comparison.
Visualizations
The project includes various visualizations in the visuals/ folder. You can generate additional visualizations by running the respective code in the notebook or app.
How It Works
Data Preprocessing: The dataset is cleaned and prepared for training. Missing values are imputed, and categorical features are encoded.
Model Training: Random Forest, AdaBoost, and XGBoost models are trained on the processed data.
Model Evaluation: Models are evaluated using metrics like accuracy, precision, recall, F1 score, and AUC-ROC.
Web Interface: Users can input loan application data through the Streamlit web app, and the app will return a prediction of loan default risk.
Visual Insights: Interactive visualizations help users explore the data, model performance, and feature importance.
Contributions
Feel free to fork the repository and contribute to this project. Any improvements in feature engineering, model performance, or deployment are welcome.

License
This project is licensed under the MIT License - see the LICENSE file for details.

