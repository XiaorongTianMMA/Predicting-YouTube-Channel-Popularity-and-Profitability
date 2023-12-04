# Created by Hongyi Zhan, Kelly Kao, Xiaorong Tian, Siqi Wang, Jiaxuan Wang


# Part 1: CPM

# Importing the libraries 
import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 

# Loading Data
df = pd.read_csv('C:/Users/95675/OneDrive/桌面/Data Mining and Visualization/Final Project/Final Code/New Global YouTube Statistics.csv')
df.head()

# Part 1.1: Data exploration and preprocessing
# View basic statistical information about CPM
print(df['CPM'].describe())
# Check for missing values
print(df['CPM'].isnull().sum()) # No na exists

# Checking duplicates
df[df.duplicated()].shape # No duplicates exist

# Checking missing values
print(df.isnull().sum())  # No missing value

# Adding predictor: percentage of urban pop
df['urban%'] = (df['Urban_population'] / df['Population'])

# Dropping irrelevent predictors
df = df.drop(columns = ['Urban_population', 'ID', 'Title', 'Abbreviation', 'channel_type', 'Growth Potential',
                        'lowest_yearly_earnings', 'highest_yearly_earnings', 'video views',
                        'rank', 'video_views_rank', 'country_rank', 'channel_type_rank',
                        'Latitude', 'Longitude',
                        'created_month', 'created_date'])

# Outliers - Remove based on 3 std above criteria
numeric_columns = df.select_dtypes(include=['number']).columns
thresholds = df[numeric_columns].mean() + 3 * df[numeric_columns].std()
outliers = (df[numeric_columns] > thresholds).any(axis=1)
df = df[~outliers]



# Building dummy variables
# Building categorical dummy variables
df_dummies = pd.get_dummies(df[['category', 'Country']]).astype(int)
df = pd.concat([df, df_dummies], axis=1)
df = df.drop(columns = ['category', 'Country'])

# Building year-range dummies
bins = [2005, 2010, 2015, 2020, 2023]
labels = ['2005-2010', '2010-2015', '2015-2020', '2020-2023']
df['year_range'] = pd.cut(df['created_year'], bins=bins, labels=labels, right=False, include_lowest=True)
df_dummies1 = pd.get_dummies(df['year_range']).astype(int)
df = pd.concat([df, df_dummies1], axis=1)
df = df.drop(columns = ['year_range', 'created_year'])




# Correlation Matrix

# All the variables
correlation_matrix = df.corr()
plt.figure(figsize=(15, 15))
sns.heatmap(correlation_matrix, annot=False, fmt='.2f', cmap='coolwarm')
plt.title("Correlation Matrix of Variables")
plt.show()

# Without dummy variables
non_dummy_columns = [col for col in df.columns 
                     if 'Country' not in col 
                     and 'category' not in col 
                     and col not in ['2005-2010', '2010-2015', '2015-2020', '2020-2023']]
correlation_matrix_non_dummy = df[non_dummy_columns].corr()
sns.set_context('talk', font_scale=1)
plt.figure(figsize=(15, 15))
sns.heatmap(correlation_matrix_non_dummy, annot=False, fmt='.2f', cmap='coolwarm')
plt.title("Correlation Matrix of Non-Dummy Variables")
plt.show()



# Construct variables
X = df.drop('CPM', axis=1)
y = df['CPM']



# Detecting multicollinearity and drop variables
from statsmodels.tools.tools import add_constant

filtered_columns = [col for col in X.columns if 'Country' not in col 
                     and 'category' not in col 
                     and col not in ['2005-2010', '2010-2015', '2015-2020', '2020-2023']]
X_filtered = X[filtered_columns]

X1 = add_constant(X_filtered)
vif_data = pd.DataFrame()
vif_data["feature"] = X1.columns

from statsmodels.stats.outliers_influence import variance_inflation_factor
for i in range(len(X1.columns)):
    vif_data.loc[vif_data.index[i],"VIF"] = variance_inflation_factor(X1.values, i)

print(vif_data)

# VIF <= 15
features_vif_le_15 = vif_data[vif_data['VIF'] <= 15]['feature'].tolist()

# VIF > 15
features_vif_gt_15 = vif_data[vif_data['VIF'] > 15]['feature'].tolist()

print("Features with VIF <= 15:", features_vif_le_15)
print("Features with VIF > 15:", features_vif_gt_15)
# Output of features with VIF > 15: 
# ['lowest_monthly_earnings', 'highest_monthly_earnings']

# Drop variables with multicollinearity 
X = X.drop(columns = ['lowest_monthly_earnings', 'highest_monthly_earnings', 'video_views_for_the_last_30_days'])



# Standardize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)


# Part 1.2: Model selection
# Part 1.2.1: Linear Regression

# Basic Linear Regression (Keep all the predictors in it)
# This model is for testing purpose.

# Splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Creating and training the Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Evaluating the model
lr_score = lr.score(X_test, y_test)
print('Linear Regression R^2 Score:', lr_score)

y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Linear Regression MSE:', mse)


# Linear Regression of different clusters after K-means clustering
# We didn't use this model finally. Just for testing
from sklearn.cluster import KMeans

y_df = y.to_frame()
kmeans_df = pd.concat([X_scaled, y_df], axis=1)
print(kmeans_df.isnull().sum())

k = 2
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
X_scaled['Cluster'] = labels
X_scaled['Cluster'].value_counts()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

y_df = y.to_frame()
X_cluster0 = X_scaled[X_scaled['Cluster'] == 0].drop('Cluster', axis=1)
y_cluster0 = y[X_scaled['Cluster'] == 0]
X_cluster1 = X_scaled[X_scaled['Cluster'] == 1].drop('Cluster', axis=1)
y_cluster1 = y[X_scaled['Cluster'] == 1]

X0_train, X0_test, y0_train, y0_test  = train_test_split(X_cluster0, y_cluster0, test_size=0.3, random_state=10)
X1_train, X1_test, y1_train, y1_test = train_test_split(X_cluster1, y_cluster1, test_size=0.3, random_state=10)


# For Cluster 0
# Creating and training the Linear Regression model
lr = LinearRegression()
lr.fit(X0_train, y0_train)

# Evaluating the model
lr_score = lr.score(X0_test, y0_test)
print('Linear Regression R^2 Score:', lr_score)

y0_pred = lr.predict(X0_test)
mse = mean_squared_error(y0_test, y0_pred)
print('Linear Regression MSE:', mse)


# For Cluster 1
# Creating and training the Linear Regression model
lr = LinearRegression()
lr.fit(X1_train, y1_train)

# Evaluating the model
lr_score = lr.score(X1_test, y1_test)
print('Linear Regression R^2 Score:', lr_score)

y1_pred = lr.predict(X1_test)
mse = mean_squared_error(y1_test, y1_pred)
print('Linear Regression MSE:', mse)



# Linear Regression without dummy variables
# We didn't pick this model after comparison.

filtered_columns = [col for col in X.columns if 'Country' not in col 
                     and 'category' not in col 
                     and col not in ['2005-2010', '2010-2015', '2015-2020', '2020-2023']]
X_filtered = X[filtered_columns]

# Standardize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_filtered_scaled = scaler.fit_transform(X_filtered)
X_filtered_scaled = pd.DataFrame(X_scaled, columns=X_filtered.columns, index=X_filtered.index)

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_filtered_train, X_filtered_test, y_filtered_train, y_filtered_test = train_test_split(X_filtered_scaled, y, test_size=0.3, random_state=0)

# Creating and training the Linear Regression model
lr_filtered = LinearRegression()
lr_filtered.fit(X_filtered_train, y_filtered_train)

# Evaluating the model
lr_filtered_score = lr_filtered.score(X_filtered_test, y_filtered_test)
print('Linear Regression R^2 Score:', lr_filtered_score)

y_filtered_pred = lr_filtered.predict(X_filtered_test)
mse_filtered = mean_squared_error(y_filtered_test, y_filtered_pred)
print('Linear Regression MSE:', mse_filtered)




# LASSO
# We didn't pick this model after comparison.

from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error

lasso = LassoCV(cv=5, random_state=0)
lasso.fit(X_train, y_train)

# Evaluating the model
lasso_score = lasso.score(X_test, y_test)
print('LASSO R^2 Score:', lasso_score)   # 0.6825653481019915

y_pred = lasso.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('LASSO MSE:', mse)

lasso_coefs = lasso.coef_
features_coefs = [(X_train.columns[i], coef) for i, coef in enumerate(lasso_coefs) if coef != 0]

features_coefs_sorted = sorted(features_coefs, key=lambda x: abs(x[1]), reverse=True)
print("Features and Coefficients Sorted by Absolute Value:")
for feature, coef in features_coefs_sorted:
    print(f"{feature}: {coef}")




# Part 1.2.2: ANN
# We didn't pick this model after comparison.

from sklearn.neural_network import MLPRegressor

mlp_regressor = MLPRegressor(hidden_layer_sizes=(13), max_iter=1000, random_state=0, activation='logistic')
model = mlp_regressor.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

rmse = mean_squared_error(y_test, y_pred, squared=False)
print("Root Mean Squared Error:", rmse)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R^2 Score:", r2)





# Part 1.2.3: Random forest 
# We picked this model after comparison.

##########################################################
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

# Define the parameter grid
param_grid = {
    'n_estimators': [25, 50, 75, 100, 125],
    'max_depth': [10, 20, 30, 40, 50, None],
    'min_samples_split': [2, 5, 10],
}

# Create a Random Forest Regressor
rf = RandomForestRegressor(random_state=0)

# Random search of parameters
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=100, cv=3, verbose=2, random_state=0, n_jobs=-1)

# Fit the random search model
rf_random.fit(X_train, y_train)

# Print the best parameters
print("Best Parameters:", rf_random.best_params_)

optimal_rf = RandomForestRegressor(n_estimators=100, min_samples_split=10, max_depth=10, random_state=0)
optimal_rf.fit(X_train, y_train)

# Make predictions
y_pred_optimal = optimal_rf.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score

# Evaluate the performance
print('Mean Squared Error:', mean_squared_error(y_test, y_pred_optimal))
print('R^2 Score:', r2_score(y_test, y_pred_optimal))

# Get feature importances
feature_importances = optimal_rf.feature_importances_

# Convert the importances into a DataFrame
features = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})

# Sort the DataFrame by importance
features = features.sort_values(by='Importance', ascending=False)

# Display the features
print(features)

# Set a threshold for feature selection, for example, 0.005
threshold = 0.005

# Select features above the threshold
selected_features = features[features['Importance'] > threshold]['Feature']

# Create new training and testing sets with selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Retrain the model with selected features
optimal_rf_selected = RandomForestRegressor(n_estimators=100, min_samples_split=10, max_depth=10, random_state=0)
optimal_rf_selected.fit(X_train_selected, y_train)

# Make predictions
y_pred_selected = optimal_rf_selected.predict(X_test_selected)

# Evaluate the performance
print('Mean Squared Error with Selected Features:', mean_squared_error(y_test, y_pred_selected))
print('R^2 Score with Selected Features:', r2_score(y_test, y_pred_selected))
########################################################################







# Part 1.2.4: Interpreting the feature importance

# Sort the importance of the features
print("Sorted Features by Importance:")
for index, row in features.iterrows():
    print(f"{row['Feature']}: {row['Importance']}")
    
threshold = 0.003
filtered_features = features[features['Importance'] >= threshold]

data = {
    "Feature": filtered_features['Feature'].tolist(),
    "Importance": filtered_features['Importance'].tolist()
}

print(data)

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Sort the DataFrame based on the Importance
df_sorted = df.sort_values(by="Importance", ascending=False)


# Set font size for the plot
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

# Plotting the bar plot
plt.figure(figsize=(8, 11))
plt.barh(df_sorted['Feature'], df_sorted['Importance'], color='skyblue', height=0.5)
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.gca().invert_yaxis()  # To display the highest importance at the top
plt.show()

# Part 2: Growth Potential

# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
# importing the libraries 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier

# Loading dataset
df = pd.read_csv('C:/Users/95675/OneDrive/桌面/Data Mining and Visualization/Final Project/Final Code/New Global YouTube Statistics.csv')
df.head()


# Part 2.1: Data exploration and pre-processing
# Adding predictor: percentage of urban pop
df['urban%'] = (df['Urban_population'] / df['Population'])
df['Country'] = df['Abbreviation'].apply(lambda x: 'United States' if x in ['US'] else ('India' if x in ['IN'] else 'Other'))
df['avg_yearly_earning'] = (df['lowest_yearly_earnings'] + df['highest_yearly_earnings'])/2

dfglm = df

# Building dummies
df_dummies = pd.get_dummies(df[["Country",'category']]).astype(int)
df = pd.concat([df, df_dummies], axis=1)
df = df.drop(columns = ['Country', 'category', 'created_month'])


# Building year-range dummies
bins = [2005, 2010, 2015, 2020, 2023]
labels = ['2005-2010', '2010-2015', '2015-2020', '2020-2023']
df['year_range'] = pd.cut(df['created_year'], bins=bins, labels=labels, right=False, include_lowest=True)
df_dummies1 = pd.get_dummies(df['year_range']).astype(int)
df = pd.concat([df, df_dummies1], axis=1)

df = df.drop(columns = ['year_range'])

# Dropping unneeded predictors:
df = df.drop(columns = ['rank','video_views_rank','Urban_population', 'ID', 'CPM', 'channel_type', 'Title', 'Abbreviation', 
                        'subscribers_for_last_30_days', 'lowest_monthly_earnings', 'highest_monthly_earnings', 
                        'subscribers_for_last_30_days',"country_rank",'channel_type_rank','Latitude',
                       'Longitude','lowest_yearly_earnings','highest_yearly_earnings','created_year','urban%'])


# Initialize the IsolationForest model
iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)

# Fit the model
iso_forest.fit(df)

# Predict the outliers
outliers = iso_forest.predict(df)

# Adding a new column to mark outliers
df['outlier'] = outliers

# Filtering the dataset to exclude outliers
df_no_outliers = df[df['outlier'] == 1].drop(columns=['outlier'])

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Function to calculate VIF
def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    
    # Calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
    return vif_data

# Selecting only numeric features for VIF calculation
numeric_df = df_no_outliers.select_dtypes(include=[np.number])
numeric_df = df_no_outliers.iloc[:, 0:14]
# Calculate VIF
vif_df = calculate_vif(numeric_df)

# Display VIF
print(vif_df)

df = df_no_outliers


# Part 2.2: Feature importance research (random forest using Grid Search CV)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

y = df['Growth Potential']
x = df.drop(columns=['Growth Potential'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=62)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

best_rf = RandomForestRegressor(**best_params, oob_score=True,random_state=42)
best_rf.fit(x_train, y_train)
print("OOB Score:", best_rf.oob_score_)
y_pred = best_rf.predict(x_test)

# Evaluation metrics for regression
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

feature_importances = best_rf.feature_importances_
features_df = pd.DataFrame({
    'Feature': x.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Display the DataFrame
non_zero_importances = features_df[features_df['Importance'] > 0.01].sort_values(by='Importance', ascending=False)

# Print the features with importance > 0.01
print(non_zero_importances)

import shap
explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(x_train)

# Plot the summary plot
plt.figure()

# Generate the SHAP summary plot
shap.summary_plot(shap_values, x_train)

# Save the plot to a file
plt.savefig("shap_summary_plot.png", bbox_inches='tight')
plt.show()

from sklearn.tree import plot_tree
rf_regressor = RandomForestRegressor(n_estimators=3, max_depth=3, random_state=42)
rf_regressor.fit(x, y)

# Plot one of the small trees
plt.figure(figsize=(20,10))
plot_tree(rf_regressor.estimators_[0], 
          filled=True, 
          feature_names=x.columns.tolist(), 
          rounded=True,
          precision=2)
plt.savefig("tree.png", bbox_inches='tight')
plt.show()

importances = best_rf.feature_importances_
feature_names = x.columns  # Adjust this if your DataFrame doesn't have column names

# Sorting the feature importances in descending order
important_indices = np.where(importances > 0.01)[0]

# Sorting the filtered feature importances in descending order
sorted_indices = np.argsort(importances[important_indices])[::-1]

# Rearrange feature names to match the sorted, filtered feature importances
feature_names = x.columns  # Adjust this if your DataFrame doesn't have column names
important_features = [feature_names[i] for i in important_indices[sorted_indices]]
important_importances = importances[important_indices][sorted_indices]

# Creating the horizontal bar plot
plt.figure(figsize=(12, 8))
plt.barh(range(len(important_indices)), important_importances, color='skyblue', edgecolor='black')
plt.yticks(range(len(important_indices)), important_features, fontsize=12)
plt.gca().invert_yaxis()  # Invert y-axis to have the most important at the top
plt.xlabel('Relative Importance', fontsize=14)
plt.title('Feature Importance (Importance > 0.01)', fontsize=16)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()


# Part 2.3: Feature selection (Lasso)
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

alphas = np.linspace(1e-5, 1, 100)
best_score = float('inf')

for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(x_train, y_train)
    y_pred = lasso.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)

    if mse < best_score:
        best_score = mse
        best_alpha = alpha

print("Best alpha:", best_alpha)

lasso_best = Lasso(alpha=best_alpha)
lasso_best.fit(x_train, y_train)

selected_features = [feature for feature, coef in zip(x.columns, lasso_best.coef_) if coef != 0]
print("Selected features:", selected_features)

x1 = x[non_zero_importances['Feature'].tolist()]
x1
x = x[selected_features]


# Part 2.4: Model selection


# Part 2.4.1: Random forest
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=62)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'max_features': [1,3,5,7,10]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

best_rf = RandomForestRegressor(**best_params, oob_score=True,random_state=42)
best_rf.fit(x_train, y_train)
print("OOB Score:", best_rf.oob_score_)
y_pred = best_rf.predict(x_test)

# Evaluation metrics for regression
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

feature_importances = best_rf.feature_importances_
features_df = pd.DataFrame({
    'Feature': x.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Display the DataFrame
non_zero_importances = features_df[features_df['Importance'] > 0.01].sort_values(by='Importance', ascending=False)

# Print the features with importance > 0.01
print(non_zero_importances)


# Part 2.4.2: ElasticNet
from sklearn.linear_model import ElasticNet
X_train, X_test, y_train, y_test = train_test_split(x1, y, test_size=0.3,random_state=62)

param_grid = {
    'alpha': np.logspace(-4, 4, 50),
    'l1_ratio': np.linspace(0.1, 1, 10)
}

# Create GridSearchCV object
model = ElasticNet()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the GridSearchCV object
grid_search.fit(X_train, y_train)

# Best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", -grid_search.best_score_)

from sklearn.metrics import mean_squared_error

# Retrieve the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Predict on the test data
y_pred = best_model.predict(X_test)

# Calculate the Mean Squared Error on test data
mse_test = mean_squared_error(y_test, y_pred)

print(f"MSE on Test Data: {mse_test}")
r_squared = r2_score(y_test, y_pred)
print(f'R-squared: {r_squared}')


# Part 2.4.3: Linear mixed model
# This is a basic example formula
import statsmodels.api as sm
import statsmodels.formula.api as smf

dfglm['Country'] = dfglm['Country'].astype('category')
dfglm['category'] = dfglm['category'].astype('category')
dfglm = dfglm.rename(columns={'video views': 'video_views'})
dfglm = dfglm.rename(columns={'Unemployment rate': 'Unemployment_rate'})
dfglm = dfglm.rename(columns={'Gross tertiary education enrollment (%)': 'education_enrollment'})
dfglm = dfglm.rename(columns={'Growth Potential': 'Growth_Potential'})

model_formula = 'Growth_Potential ~ avg_yearly_earning + video_views + subscribers + video_views_for_the_last_30_days+ elapsedtime+ VideoCommentCount + videoLikeCount+ videoDislikeCount + created_date + Unemployment_rate + education_enrollment +Population'

# Create the model
mixedlm_model = smf.mixedlm(model_formula, dfglm, groups=dfglm['Country'])

mixedlm_result = mixedlm_model.fit()
print(mixedlm_result.summary())

predictions = mixedlm_result.predict(dfglm)

from sklearn.metrics import mean_squared_error

# Assuming 'actual' contains the actual values of the dependent variable
mse = mean_squared_error(dfglm['Growth_Potential'], predictions)
print(f'Mean Squared Error: {mse}')