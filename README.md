## 📝 Project Overview

This project aims to predict whether an individual's income exceeds $50,000 per year. It leverages machine learning algorithms to classify income levels, utilizing techniques for data preprocessing, exploratory data analysis (EDA), and model optimization.

## 📂 Repository Structure

- **`main.ipynb`**: The main Jupyter Notebook containing:
  - Data preprocessing steps.
  - Feature engineering and selection.
  - Exploratory data analysis.
  - Model training, hyperparameter tuning, and evaluation.
- **`data/`**: Folder containing the dataset used in this project.
- **`KerasTuner/`**: This folder contains files and configurations generated during the hyperparameter tuning process.

## 🛠️ Tools and Libraries Used

- **Programming Language**: Python
- **Libraries for Analysis and Modeling**:
  - `pandas` and `numpy`: Data manipulation and numerical computations.
  - `matplotlib` and `seaborn`: Visualizations for insights and patterns.
  - `scikit-learn`: Machine learning model implementation and evaluation.
  - `keras_tuner`: Hyperparameter tuning for optimal model performance.
  - `tensorflow`: Deep learning framework used for building and training models.
- **Development Environment**: Jupyter Notebook

---

## 📋 Feature Dictionary

| **Feature**           | **Description**                                                              | **Type**         |
|------------------------|------------------------------------------------------------------------------|------------------|
| **Age**               | The age of the individual.                                                   | Numerical        |
| **Workclass**         | The type of work or employment sector (e.g., Private, Government, etc.).      | Categorical      |
| **Education**         | The highest level of education achieved (e.g., Bachelors, HS-grad).          | Categorical      |
| **Education-Num**     | Numeric representation of education level.                                   | Numerical        |
| **Marital-Status**    | Marital status of the individual (e.g., Married, Never-married).              | Categorical      |
| **Occupation**        | The type of job held (e.g., Tech-support, Craft-repair).                     | Categorical      |
| **Relationship**      | Relationship status relative to the household (e.g., Husband, Wife).         | Categorical      |
| **Race**              | The racial background of the individual.                                     | Categorical      |
| **Sex**               | Gender of the individual (e.g., Male, Female).                               | Categorical      |
| **Hours-per-Week**    | Average hours worked per week.                                               | Numerical        |
| **Native-Country**    | The country of origin for the individual.                                    | Categorical      |
| **Income**            | Target variable indicating if income is `<=50K` or `>50K`.                  | Categorical      |

---

## 📋 Steps Performed

### 1. Data Preprocessing
### 2. Exploratory Data Analysis (EDA)
### 3. Model Development
### 4. Evaluation Metrics
---

## 🖼️ Key Visualizations

### 1. Age Distribution 
![age](https://github.com/user-attachments/assets/41a93e2c-b5a0-4c07-b99e-abf50714172a)

### 2. Matrital Status vs. Income
![statuus](https://github.com/user-attachments/assets/60773d7a-743f-43fc-87c0-11a5bbfdd126)

### 3. Top 5 Most Important Features
![top](https://github.com/user-attachments/assets/2346fabc-d9b5-4905-add0-5fa944c77dff)

### 4. Inertia& Silhouette Score for Clustering
![clustring](https://github.com/user-attachments/assets/ccc7e89b-a72d-45e0-8887-db61d36fc506)

### 5. Roc Curve Comparison
![curve](https://github.com/user-attachments/assets/90572901-bd3e-48ff-acf5-f1fb616585a5)

---

## 💡 Insights and Results

- **Influential Factors**: Features like education, age, and hours worked were significant predictors of income level.
- **Model Performance**:
  - The dataset is `highly imbalanced`, with `76%` of instances labeled `<=50K` and only `24%` labeled `>50K`, affecting the models' ability to predict the minority class effectively.
  - `Both` models performed `better` on the majority class `(<=50K)`, with higher precision and recall compared to the minority class (>50K).
  - The `Random Forest` model `struggled` with recall for the `minority class`, missing some true positives.
  - The `tuned Neural Network` model improved performance for the `minority class` but still showed lower recall, highlighting the challenge posed by the imbalance.
  - `Handling` the `imbalance`, may lead to better results, especially for the minority class (>50K).
---

*Note: This project was developed for educational purposes and demonstrates practical applications of machine learning techniques.*
```
