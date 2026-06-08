# Spaceship Titanic Prediction 🚀

A machine learning project built on the [Kaggle Spaceship Titanic Competition](https://www.kaggle.com/competitions/spaceship-titanic) dataset. The goal is to predict whether a passenger was transported to an alternate dimension during the Spaceship Titanic's collision with a spacetime anomaly.

---

 Project Overview

This notebook covers a complete ML pipeline with a focus on thoughtful feature engineering — extracting new features from raw columns like `PassengerId` and `Cabin` — followed by preprocessing, model training, and generating a Kaggle submission file using a Random Forest classifier.

---

 Dataset

- **train.csv** — Training data with transportation labels
- **test.csv** — Test data for generating predictions
- Source: [Kaggle Spaceship Titanic Competition](https://www.kaggle.com/competitions/spaceship-titanic/data)

---

## 🔧 Steps Performed

 1. Feature Engineering
- Split `PassengerId` into `Group` and `Number` to capture passenger grouping
- Split `Cabin` into three separate features: `deck`, `num`, and `side`
- Created a new `Total Spend` feature by summing all spending columns (`RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`)
- Used domain logic: passengers in CryoSleep cannot spend money, so missing spending values were filled with 0 for those passengers

 2. Data Preprocessing
- Encoded boolean features: `VIP`, `CryoSleep`, and `Transported` (False → 0, True → 1)
- Handled missing values:
  - Spending columns → **mean imputation**
  - `Age` → **median imputation**
  - `num` → **median imputation**
- Dropped irrelevant columns: `PassengerId`, `Cabin`, `Name`
- Encoded categorical columns (`HomePlanet`, `Destination`, `deck`, `side`) using **Label Encoding**
- Scaled spending features using **StandardScaler**

 3. Model Training
- Used **Random Forest Classifier** (100 trees, entropy criterion)
- Applied **Stratified Train/Test Split** (80/20) to preserve class balance

 4. Model Evaluation
- Evaluated using **Confusion Matrix** and **Accuracy Score**

 5. Prediction & Submission
- Applied the exact same preprocessing pipeline to the test set
- Generated a `submission.csv` file with boolean `Transported` predictions for Kaggle submission

---

 Libraries Used

- `NumPy`
- `Pandas`
- `Matplotlib`
- `Scikit-learn`

---

 Results

| Model | Accuracy |
|---|---|
| Random Forest (train/test split) | Update after running |

> Run the notebook and replace the accuracy above with your actual result from the `accuracy_score` output.

---

  How to Run

1. Clone this repository
2. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/spaceship-titanic/data) and place `train.csv` and `test.csv` in the project folder
3. Open `Spaceship_titanic.ipynb` in Jupyter Notebook or VS Code
4. Run all cells in order

---

## 👤 Author

**Yahia Mohtady**
GitHub: [YahiaZeada06](https://github.com/YahiaZeada06)