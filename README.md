
# Houses Pricing Prediction

This repository contains a project focused on predicting house prices using machine learning techniques. The project includes data processing scripts, model training code, and a detailed analysis notebook.

## Project Overview

The goal of this project is to develop a predictive model for house prices based on various features. The process involves:

- **Data Collection:** Gathering relevant data on house prices and features.
- **Data Preprocessing:** Cleaning and preparing the data for analysis.
- **Exploratory Data Analysis:** Understanding the data through visualization and statistical analysis.
- **Model Training:** Building and training machine learning models to predict house prices.
- **Model Evaluation:** Assessing the performance of the models using appropriate metrics.
- **Model Explanation:** Documenting the theory behind the chosen model

For a detailed explanation of the models and their performance, refer to the `model_explanation.pdf` document.

## Repository Structure

- `data/`: Contains the dataset used for training and evaluation.
- `models/`: Stores trained models and related artifacts.
- `notebook/`: Includes 
    - `analysis.ipynb`, a Jupyter notebook for data processing, exploratory data analysis, and feature engineering. 
    - It also contains `models.ipynb`, a Jupyter notebook for model training and evaluation.
- `data_processing.py`: Script for preprocessing and cleaning the data.
- `train_models.py`: Script for training machine learning models.
- `model_explanation.pdf`: Document explaining the theory behind the chosen model


## Getting Started

To get a local copy of the project up and running, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/theodruilhe/project_maths_for_machine_learning.git
   cd project_maths_for_machine_learning
   ```

2. **Set up a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   *Note: Ensure that a `requirements.txt` file is present in the repository, listing all necessary packages.*

4. **Run the data processing script:**

   ```bash
   python data_processing.py
   ```

   This script will preprocess the raw data and prepare it for model training.

5. **Train the models:**

   ```bash
   python train_models.py
   ```

   This script will train the machine learning models using the processed data.

6. **Explore the analysis:**

   Open the Jupyter notebook(s) in the `notebook/` directory to explore data analysis, model evaluation, and results.


## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.
