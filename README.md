# CelestiaPraedicere: Predicting Celestial Patterns with Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/framework-Scikit--learn-orange" alt="Framework">
</p>

**CelestiaPraedicere** is a comprehensive machine learning project that demonstrates an end-to-end workflow for identifying celestial bodies. The core of the project is to predict the identity of planets and their moons based on their orbital position data, for which we developed and optimized a **Multiple Random Forest** model to achieve high prediction accuracy.

Beyond just the model, this repository is a complete ecosystem that includes a synthetic data generator for creating idealized training sets and detailed notebooks for experimentation. This project serves as a practical case study in data synthesis, feature engineering, and tiered modeling.

## 📝 Table of Contents
* [Key Features](#key-features)
* [Repository Structure](#repository-structure)
* [Dataset Deep Dive](#dataset-deep-dive)
  * [The Original Dataset](#the-original-dataset)
  * [The Synthetic Dataset](#the-synthetic-dataset)
* [Notebook Descriptions](#notebook-descriptions)
* [Methodology](#methodology)
* [Installation & Setup](#installation--setup)
* [Usage Instructions](#usage-instructions)
* [Further Information & References](#further-information--references)

## ✨ Key Features
* **High-Accuracy Prediction Model**: Employs a sophisticated two-step architecture (a planet-detector followed by a moon-detector) using Random Forest classifiers. This tiered approach allows the model to solve a simpler problem first, achieving a fine-tuned prediction accuracy of approximately 97% on the curated dataset.

* **Synthetic Data Generator**: Includes a custom Python script to programmatically generate clean, smooth, and uniform orbital data. This process gives us complete control over the dataset's quality and balance, creating an ideal, noise-free training environment that is crucial for model performance.

* **Advanced Feature Engineering**: Implements a `distance_r` feature, which is the calculated radial distance of a body from the Sun ($\sqrt{x^2+y^2+z^2}$). This simple yet powerful feature provides the model with critical spatial context that is not immediately apparent from the raw coordinates, significantly boosting its ability to distinguish between orbits.

## 📂 Repository Structure
The repository is organized to clearly separate data, source code, and results.
```
CelestiaPraedicere/
│
├── dataset/
│   ├── solar_system_positions_with_velocity.csv  # Original raw dataset from Kaggle.
│   └── one_revolution_paths.csv                  # The clean, generated synthetic dataset used for training.
│
├── results/
│   └── (This folder is intended to store outputs like plots, confusion matrices, model files, etc.)
│
├── synthetic_generate_data/
│   └── one-revolution_path.ipynb                 # Jupyter Notebook to generate the synthetic dataset from orbital elements.
│
└── multiple-rf.ipynb                             # The main notebook for the entire modeling and prediction workflow.
```

## 📊 Dataset Deep Dive
This project leverages two primary datasets, each serving a distinct purpose in the model development lifecycle.

### The Original Dataset
* **File**: `dataset/solar_system_positions_with_velocity.csv`
* **Source**: Adapted from the [Solar System Bodies Positions (2020-2024) on Kaggle](https://www.kaggle.com/datasets/nikitamanaenkov/solar-system-bodies-positions-2020-2024/data).
* **Description**: This dataset contains real-world observational data. It is rich with high-fidelity information, including the heliocentric position (`x_au`, `y_au`, `z_au`) and velocity (`vx`, `vy`, `vz`) of hundreds of celestial bodies at various timestamps.
* **Challenge**: The primary drawback of this dataset is the presence of "noisy" and ambiguous labels. For instance, it lists both `'1 MERCURY BARYCENTER'` and `'199 MERCURY'`. These are essentially the same object from a positional standpoint, which confuses the classification model, degrades its performance, and makes evaluation unreliable. This label ambiguity was the core motivation for synthesizing a cleaner dataset.

### The Synthetic Dataset
* **File**: `dataset/one_revolution_paths.csv`
* **Source**: Generated internally using the `synthetic_generate_data/one-revolution_path.ipynb` notebook.
* **Description**: This dataset was meticulously crafted to create the ideal training environment. The generation process uses fundamental orbital elements (from a separate source) and the principles of orbital mechanics (Kepler's laws) to produce smooth, mathematically perfect trajectories.
* **Advantage**: For each of the 23 celestial bodies chosen for this project (9 planets, 14 moons), the dataset contains **exactly 200 data points** representing one complete revolution. This standardization accomplishes two critical goals: it eliminates any class imbalance that might bias the model, and it provides a consistent, noise-free representation of each orbit, allowing the model to learn the underlying patterns more effectively.

## 📓 Notebook Descriptions
* **`multiple-rf.ipynb`**: This is the central, master notebook of the project where all modeling takes place.
  * **Purpose**: To train, evaluate, and demonstrate the final Multiple Random Forest prediction model on the synthetic dataset.
  * **Latest Version**: The logic within this notebook has been updated to reflect the high-accuracy, two-step architecture detailed in [**this Gist**](https://gist.github.com/1magines/5c68a8828fe109ce7ef7ffc46b8ae381).
  * **Workflow**: The notebook follows a clear sequence:
    1.  **Data Loading**: Loads the `one_revolution_paths.csv` synthetic data.
    2.  **Feature Engineering**: Creates the `distance_r` feature.
    3.  **Data Splitting & Balancing**: Splits the data and applies the SMOTE technique to oversample minority classes (moons with fewer data points) in the training set.
    4.  **Tiered Model Training**: Trains the two separate Random Forest models (planet-detector and moon-detector).
    5.  **Evaluation**: Assesses the model's performance using metrics like accuracy, precision, recall, and visualizes the results with a confusion matrix.

* **`synthetic_generate_data/one-revolution_path.ipynb`**:
  * **Purpose**: To programmatically generate the `one_revolution_paths.csv` file used in the main notebook.
  * **Process**: This notebook takes a table of fundamental orbital elements (e.g., semi-major axis, eccentricity, inclination) as input. It then uses a propagation function, based on celestial mechanics, to calculate 200 distinct (x, y, z) positional coordinates that trace one complete, smooth orbit for each celestial body.

## 🧠 Methodology
The core of this project is the tiered classification model (`multiple-rf.ipynb`), which is designed to solve a complex multi-class problem by breaking it down into two simpler, more manageable steps.

1.  **Planet-Detector Model**: The first Random Forest model is trained on a simplified version of the problem. Its goal is to distinguish between the 9 major planets and lump all other objects (the 14 moons) into a single "other" category. This allows the model to first master the large-scale distinctions between planetary orbits.

2.  **Moon-Detector Model**: The second Random Forest model is a specialist. It is trained *only* on the data for the 14 moons. Its sole purpose is to differentiate between these moons, a much more nuanced task given their often-close proximity to their host planets.

3.  **Combined Prediction Pipeline**: When making a final prediction, an unknown data point is first fed into the Planet-Detector.
    * If the model predicts a planet (e.g., "4 MARS BARYCENTER"), that is the final prediction.
    * If the model predicts "other", the data point is then passed to the Moon-Detector, which makes the final, specific classification (e.g., "601 MIMAS").

4.  **Optimization Techniques**:
    * **SMOTE (Synthetic Minority Over-sampling Technique)** is applied to the training sets of both models. This technique creates synthetic data points for the minority classes (especially smaller or more distant moons), ensuring the models do not become biased towards the majority classes and can learn to identify all objects effectively.
    * **Feature Engineering (`distance_r`)** provides the models with explicit information about an object's distance from the sun, a critical factor in determining its identity, which helps to significantly improve classification accuracy.

## ⚙️ Installation & Setup
To set up and run this project in your local environment, please follow these detailed steps:

1.  **Clone the Repository**
    Open your terminal or command prompt and clone the repository to your local machine.
    ```bash
    git clone [https://github.com/1magines/CelestiaPraedicere.git](https://github.com/1magines/CelestiaPraedicere.git)
    cd CelestiaPraedicere
    ```

2.  **Create and Activate a Virtual Environment (Strongly Recommended)**
    Using a virtual environment prevents conflicts with other Python projects.
    ```bash
    # Create the environment
    python -m venv venv
    
    # Activate the environment
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
    You will know it's active when you see `(venv)` at the beginning of your terminal prompt.

3.  **Install Dependencies**
    This project requires several key Python libraries. A `requirements.txt` is not provided, so you can install them directly using pip. Ensure your virtual environment is active.
    ```bash
    pip install pandas numpy scikit-learn "imblearn>=0.8" jupyterlab matplotlib seaborn
    ```

## 🚀 Usage Instructions
To see the model training process, evaluate its performance, and understand the core logic:
1.  Ensure you are in the project's root directory (`CelestiaPraedicere/`) in your terminal.
2.  Make sure your virtual environment (`venv`) is activated.
3.  Start the Jupyter Lab server:
    ```bash
    jupyter lab
    ```
4.  Your browser will open with the Jupyter interface. Use the file navigator on the left to open the **`multiple-rf.ipynb`** notebook.
5.  You can run the cells sequentially ("Run" -> "Run All Cells") or one by one to see the output of each step, including data loading, model training, and the final classification report and confusion matrix.

## ℹ️ Further Information & References
* **Project Presentation**: For a more guided and visual explanation of the project's background, methodology, and results, please see our detailed presentation hosted on the Open Science Framework (OSF).
  * **[View Presentation on OSF](https://osf.io/dazbn)**

* **Explanatory Article**: We have also authored a companion article on Medium that summarizes the project's journey, challenges, and outcomes in a narrative format.
  * **[Read the Article on Medium](https://medium.com/@azwaaliyahz/predicting-celestial-patterns-with-random-forest-ebaacc44ff5e)**

* **Primary Data Source Reference**:
  The original dataset that served as the foundation and inspiration for this project was sourced from:
  * **Manaenkov, Nikita.** (2020). *Solar System Bodies Positions (2020-2024)*. Kaggle. [https://www.kaggle.com/datasets/nikitamanaenkov/solar-system-bodies-positions-2020-2024/data](https://www.kaggle.com/datasets/nikitamanaenkov/solar-system-bodies-positions-2020-2024/data)
