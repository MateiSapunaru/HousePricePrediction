# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl as pyx
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor

# Load the dataset
dataset = pd.read_excel("venv/HousePricePrediction.xlsx")
pd.set_option('display.max_columns', None)

# Initial dataset exploration
print("First 5 records of the dataset:")
print(dataset.head(5))
print(f"Dataset shape: {dataset.shape}")

# Data Preprocessing

# Checking categorical, integer, and float columns
obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)
print(f"Number of Categorical variables: {len(object_cols)}")

int_columns = (dataset.dtypes == 'int')
int_cols = list(int_columns[int_columns].index)
print(f"Number of Integer variables: {len(int_cols)}")

float_columns = (dataset.dtypes == 'float')
fl_cols = list(float_columns[float_columns].index)
print(f"Number of Float variables: {len(fl_cols)}")

# Correlation heatmap of numerical features
numerical_dataset = dataset.select_dtypes(include=['number'])

plt.figure(figsize=(12, 6))
sns.heatmap(numerical_dataset.corr(),
            cmap='BrBG',
            fmt='.2f',
            linewidths=2,
            annot=True)
# Uncomment to display the plot
# plt.show()

# Unique value count for categorical columns
unique_values = [dataset[col].unique().size for col in object_cols]

plt.figure(figsize=(10, 6))
plt.title('Number of Unique values in Categorical Features')
plt.xticks(rotation=90)
sns.barplot(x=object_cols, y=unique_values)
# Uncomment to display the plot
# plt.show()

# Distribution of categorical features
plt.figure(figsize=(18, 36))
plt.title('Categorical Features: Distribution')
plt.xticks(rotation=90)
index = 1

for col in object_cols:
    y = dataset[col].value_counts()
    plt.subplot(11, 4, index)
    plt.xticks(rotation=90)
    sns.barplot(x=list(y.index), y=y)
    index += 1
# Uncomment to display the plot
# plt.show()

# Drop ID column and handle missing SalePrice values
dataset.drop(['Id'], axis=1, inplace=True)
dataset['SalePrice'] = dataset['SalePrice'].fillna(dataset['SalePrice'].mean())

# Drop remaining rows with missing values
new_dataset = dataset.dropna()
print("Null values in the cleaned dataset:")
print(new_dataset.isnull().sum())

# Re-encode categorical columns using One-Hot Encoding
s = (new_dataset.dtypes == 'object')
object_cols = list(s[s].index)
print(f"No. of categorical features: {len(object_cols)}")

OH_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
OH_cols.index = new_dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out()

# Combine one-hot encoded columns with the dataset
df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)

# Split the dataset into features and target
X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']

# Train-test split (80% training, 20% validation)
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)

# Model Training and Evaluation

# 1. Support Vector Regressor (SVR)
model_SVR = SVR()
model_SVR.fit(X_train, Y_train)
Y_pred_SVR = model_SVR.predict(X_valid)
print(f"MAPE for SVR: {mean_absolute_percentage_error(Y_valid, Y_pred_SVR):.4f}")

# 2. Random Forest Regressor
model_RFR = RandomForestRegressor(n_estimators=10)
model_RFR.fit(X_train, Y_train)
Y_pred_RFR = model_RFR.predict(X_valid)
print(f"MAPE for RandomForest: {mean_absolute_percentage_error(Y_valid, Y_pred_RFR):.4f}")

# 3. Linear Regression
model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred_LR = model_LR.predict(X_valid)
print(f"MAPE for Linear Regression: {mean_absolute_percentage_error(Y_valid, Y_pred_LR):.4f}")

# 4. CatBoost Regressor
cb_model = CatBoostRegressor(verbose=0)
cb_model.fit(X_train, Y_train)
Y_pred_CatBoost = cb_model.predict(X_valid)

# Evaluate CatBoost performance
cb_mae = mean_absolute_error(Y_valid, Y_pred_CatBoost)
cb_r2_score = r2_score(Y_valid, Y_pred_CatBoost)
print(f"MAE for CatBoost: {cb_mae:.4f}")
print(f"MAPE for CatBoost: {cb_r2_score:.4f}")
