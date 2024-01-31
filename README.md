**Breast Cancer Diagnosis using Logistic Regression**

This repository contains Python code for predicting breast cancer diagnosis using logistic regression. The dataset used for this project is obtained from the UCI Machine Learning Repository.

### Introduction
The aim of this project is to predict whether a breast tumor is malignant (M) or benign (B) based on various features extracted from digitized images of breast cancer biopsies.

### Requirements
- Python 3.x
- Libraries: pandas, numpy, seaborn, matplotlib, scikit-learn

### Installation
1. Clone the repository to your local machine.
2. Install the required libraries using pip:
   ```
   pip install pandas numpy seaborn matplotlib scikit-learn
   ```

### Usage
1. Import necessary libraries:
   ```python
   import pandas as pd
   import numpy as np
   import seaborn as sns
   import matplotlib.pyplot as plt
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression
   ```

2. Load and preprocess the dataset:
   - The dataset is loaded from the 'wdbc.data' file available in the repository.
   - Preprocessing steps include dropping unnecessary columns, encoding the 'Diagnosis' column, and splitting the data into training and testing sets.

3. Train the logistic regression model:
   - Fit the logistic regression model to the training data.
   - Evaluate the model's performance using the testing data.

4. Display results:
   - Print the accuracy of the logistic regression model.
   - Visualize the dataset and the model's predictions using seaborn and matplotlib.

### Files
- **wdbc.data**: Dataset file containing breast cancer diagnosis data.
- **breast_cancer_diagnosis.ipynb**: Jupyter Notebook containing the Python code.
- **README.md**: This file.

### References
- UCI Machine Learning Repository: [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

### Contributors
- [Your Name]
- [Your Email]

### License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

For any questions or feedback, please contact [Your Email].

Thank you for using Breast Cancer Diagnosis using Logistic Regression!
