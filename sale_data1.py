
# Data extracted from the sale data

import pandas as pd
df = pd.read_csv('sale_data.csv')
print(df.head())

# Have fundamental information about the data

print(df.info())
print(df.describe())

# check and process missing values in data

print(df.isnull().sum())

#We are going to proceed by deleting the rows with missing values

df_cleaned = df.dropna()
print(df_cleaned.isnull().sum())

print(df_cleaned.describe())
print(df_cleaned[df_cleaned['Sales'] < 0])

# check and process duplicate values in data

print(df_cleaned.duplicated().sum())

#We are going to proceed by deleting the duplicate values

df_cleaned = df_cleaned.drop_duplicates()
print(df_cleaned.duplicated().sum())

# we have several types of data, we need to do a conversion

print(df_cleaned.dtypes)

df_cleaned['Order Date'] = pd.to_datetime(df_cleaned['Order Date'], format="%d/%m/%Y")
df_cleaned['Ship Date'] = pd.to_datetime(df_cleaned['Ship Date'], format="%d/%m/%Y")
df_cleaned['Postal Code'] = df_cleaned['Postal Code'].astype('Int64')
df_cleaned['Sales'] = pd.to_numeric(df_cleaned['Sales'])
print(df_cleaned.dtypes)

# Save the cleaned data

df_cleaned.to_csv('sale_data_cleaned.csv', index=False)

# Exploratory Data Analysis (EDA)
#Understanding sales distribution and identifying general trends.
# We are going to analyse sales trends by different dimensions such as dates, regions, product categories

import matplotlib.pyplot as plt
import seaborn as sns

# Sales distribution

plt.hist(df_cleaned['Sales'], bins=50)
plt.title('Sales distribution')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()

# Analyse sales by product category

sales_by_category = df_cleaned.groupby('Category')['Sales'].sum()
sales_by_category.plot(kind='bar')
plt.title('Sales by product category')
plt.xlabel('Product category')
plt.ylabel('Sales')
plt.show()

# Analyse sales by region

sales_by_region = df_cleaned.groupby('Region')['Sales'].sum()
sales_by_region.plot(kind='bar')
plt.title('Sales by region')
plt.xlabel('Region')
plt.ylabel('Sales')
plt.show()

# Compare sales between customer segments (Consumer, Corporate, Home Office)

sales_by_segment = df_cleaned.groupby('Segment')['Sales'].sum()
sales_by_segment.plot(kind='bar')
plt.title('Sales by segment')
plt.xlabel('Segment')
plt.ylabel('Sales')
plt.show()

# Compare sales between different shipping modes (First Class, Second Class, Standard Class)

sales_by_ship_mode = df_cleaned.groupby('Ship Mode')['Sales'].sum()
sales_by_ship_mode.plot(kind='bar')
plt.title('Sales by ship mode')
plt.xlabel('Ship mode')
plt.ylabel('Sales')
plt.show()

# Analyse sales trends over time

sales_over_time = df_cleaned.groupby('Order Date')['Sales'].sum()
sales_over_time.plot()
plt.title('Sales trend over time')
plt.xlabel('Order date')
plt.ylabel('Sales')
plt.show()

# Analysis of sales by state

sales_by_state = df_cleaned.groupby('State')['Sales'].sum()
sales_by_state = sales_by_state.sort_values(ascending=False).head(10)
sales_by_state.plot(kind='bar', figsize=(20, 10))
plt.title('TOP 10 Sales by state')
plt.xlabel('State')
plt.ylabel('Sales')
plt.show()

# Let's take a look at the best-selling products

best_selling_products = df_cleaned.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(10)
best_selling_products.plot(kind='bar', figsize=(20, 10))
plt.title('TOP 10 best-selling products')
plt.xlabel('Product')
plt.ylabel('Sales')
plt.show()

### Noten 1 : By understanding which products are the most popular, companies can better manage their stocks to avoid shortages and minimise overstocking.
### Note 2 : Analysing the best-selling products enables us to understand price sensitivity and adjust prices to maximise profits.

# Let's take a look at the worst-selling products

worst_selling_products = df_cleaned.groupby('Product Name')['Sales'].sum().sort_values().head(10)
worst_selling_products.plot(kind='bar', figsize=(20, 10))
plt.title('TOP 10 worst-selling products')
plt.xlabel('Product')
plt.ylabel('Sales')
plt.show()

# As much analysis as we can do with these data sets, it all depends on what we want to know.

# Predictive modelling: we are now going to make forecasts

# Preparing data for modelling 
features = df_cleaned[['Category', 'Sub-Category', 'Region', 'Order Date']]
target = df_cleaned['Sales']

# Converting categorical variables into numerical variables
features = pd.get_dummies(features)

# Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Use a linear regression model to predict sales.
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluating the model's performance
from sklearn.metrics import mean_squared_error

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

### The mean squared error is a measure of the quality of the model. The lower the value, the better the model.

# Let's extract the coefficients from the model to interpret the impact of the characteristics

feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': model.coef_})
print(feature_importance.sort_values(by='Coefficient', ascending=False))

### We can also use other models such as Random Forest, Decision Tree to predict sales.

## Decision Tree method

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Create and train the model
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# Make predictions
tree_predictions = tree_model.predict(X_test)

# Evaluating the model
tree_mse = mean_squared_error(y_test, tree_predictions)
tree_r2 = r2_score(y_test, tree_predictions)
print(f"Decision Tree - Mean Squared Error: {tree_mse}, R²: {tree_r2}")


## Random Forest method

from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor(random_state=42, n_estimators=100)
forest_model.fit(X_train, y_train)

# Evaluating the model's performance
forest_predictions = forest_model.predict(X_test)
forest_mse = mean_squared_error(y_test, forest_predictions)
forest_r2 = forest_model.score(X_test, y_test)  # or forest_r2 = r2_score(y_test, forest_predictions)
print(f"Random Forest - Mean Squared Error: {forest_mse}, R²: {forest_r2}")


(tree_mse, tree_r2, forest_mse, forest_r2)

#Extract the importance of characteristics from the Random Forest model to understand which variables have the greatest influence on sales.


importances = forest_model.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)

# Note 1 : The Random Forest model is a more powerful model than the Decision Tree model.
# Note 2 : The Random Forest model provides a better explanation of sales variance
# Remark 1 : the lower the MSE of the model, the better the model, as this means it has a smaller error.
# Remark 2 : the closer the R² value is to 1, the better the model, as this means it explains a larger proportion of the variance in the data.

## Model Validation and Refinement : Cross Validation

from sklearn.model_selection import cross_val_score

# With Decision Tree model
tree_cv_scores = cross_val_score(tree_model, features, target, cv=5, scoring='neg_mean_squared_error')
tree_cv_mse = -tree_cv_scores.mean()
print(f"Decision Tree - Cross-Validated Mean Squared Error: {tree_cv_mse}")

# With Random Forest model
forest_cv_scores = cross_val_score(forest_model, features, target, cv=5, scoring='neg_mean_squared_error')
forest_cv_mse = -forest_cv_scores.mean()
print(f"Random Forest - Cross-Validated Mean Squared Error: {forest_cv_mse}")

## Model Validation and Refinement : Hyperparameter Tuning

# with decision tree model
from sklearn.model_selection import GridSearchCV
tree_param_grid = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 10, 20]}

tree_grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42), tree_param_grid, cv=5, scoring='neg_mean_squared_error')
tree_grid_search.fit(X_train, y_train)

best_tree_model = tree_grid_search.best_estimator_
best_tree_predictions = best_tree_model.predict(X_test)
best_tree_mse = mean_squared_error(y_test, best_tree_predictions)
print(f"Best Decision Tree - Mean Squared Error: {best_tree_mse}")
print(f"Best Decision Tree Parameters: {tree_grid_search.best_params_}")

# with random forest model
forest_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30]
}

forest_grid_search = GridSearchCV(RandomForestRegressor(random_state=42), forest_param_grid, cv=5, scoring='neg_mean_squared_error')
forest_grid_search.fit(X_train, y_train)

best_forest_model = forest_grid_search.best_estimator_
best_forest_predictions = best_forest_model.predict(X_test)
best_forest_mse = mean_squared_error(y_test, best_forest_predictions)
print(f"Best Random Forest - Mean Squared Error: {best_forest_mse}")
print(f"Best Random Forest Parameters: {forest_grid_search.best_params_}")

# model deployment 

pip install joblib

# Results and Recommendations










