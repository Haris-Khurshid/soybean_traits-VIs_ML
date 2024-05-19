#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
print(os.getcwd())


# In[2]:


os.chdir(r"D:\Genetics\R\ML")


# In[3]:


print(os.getcwd())


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


from sklearn.linear_model import LinearRegression


# In[7]:


from sklearn.tree import DecisionTreeRegressor


# In[8]:


from sklearn.ensemble import RandomForestRegressor


# In[9]:


import xgboost as xgb


# In[10]:


data = pd.read_csv("soybean_data.csv")


# In[11]:


data.describe()


# In[12]:


print(data)


# In[13]:


# Define all features except PlotY as independent variables
features = data.columns.tolist() 
features.remove("Yield")
features.remove("Gen")
features.remove("Rep")
features.remove("FC")
features.remove("PlotY")


# In[14]:


# Define the target variable
target = "PlotY"


# In[15]:


# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)


# In[16]:


print(features)


# In[17]:


# Create and train the Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)


# In[18]:


# Create and train the Decision Tree Regressor model
dt_reg = DecisionTreeRegressor()
dt_reg.fit(X_train, y_train)


# In[19]:


# Create and train the Random Forest Regressor model
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)


# In[20]:


# Create and train the XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
xgb_model.fit(X_train, y_train)


# In[21]:


# Make predictions on the test set for Linear Regression model
y_pred_lin = lin_reg.predict(X_test)


# In[22]:


print(y_pred_lin)


# In[23]:


# Make predictions on the test set for decision tree model
y_pred_dt = dt_reg.predict(X_test)


# In[24]:


print(y_pred_dt)


# In[25]:


# Make predictions on the test set for random forest model
y_pred_rf = rf_reg.predict(X_test)


# In[26]:


print(y_pred_rf)


# In[27]:


# Make predictions on the test set for xgb model
y_pred_xgb = xgb_model.predict(X_test)


# In[28]:


print(y_pred_xgb)


# In[29]:


# Evaluate the performance of each model using mean squared error (MSE)
from sklearn.metrics import mean_squared_error

mse_lin = mean_squared_error(y_test, y_pred_lin)
mse_dt = mean_squared_error(y_test, y_pred_dt)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)


# In[30]:


# Print the performance metrics for each model
print("Linear Regression MSE:", mse_lin)
print("Decision Tree Regressor MSE:", mse_dt)
print("Random Forest Regressor MSE:", mse_rf)
print("XGBoost Regressor MSE:", mse_xgb)


# In[31]:


# Evaluate the performance of each model using RMSE and R-squared
from sklearn.metrics import mean_squared_error, r2_score

mse_lin = mean_squared_error(y_test, y_pred_lin)
rmse_lin = np.sqrt(mse_lin)  # Calculate RMSE from MSE
r2_lin = r2_score(y_test, y_pred_lin)

mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
r2_dt = r2_score(y_test, y_pred_dt)

mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

# Print the performance metrics for each model
print("Model\tRMSE\tR-squared")
print("-------\t-------\t---------")
print("Linear Regression:", round(rmse_lin, 2), "\t", round(r2_lin, 2))
print("Decision Tree:", round(rmse_dt, 2), "\t", round(r2_dt, 2))
print("Random Forest:", round(rmse_rf, 2), "\t", round(r2_rf, 2))
print("XGBoost:", round(rmse_xgb, 2), "\t", round(r2_xgb, 2))


# In[51]:


plt.scatter(y_test, y_pred_lin)
plt.plot([0.5, 3],
        [0.5, 3],
        color= 'r',
        linestyle = '-',
        linewidth = 2)
plt.xlabel("Observed")
plt.ylabel("Predicted")


# In[52]:


plt.scatter(y_test, y_pred_dt)
plt.plot([0.5, 3],
        [0.5, 3],
        color= 'r',
        linestyle = '-',
        linewidth = 2)
plt.xlabel("Observed")
plt.ylabel("Predicted")


# In[53]:


plt.scatter(y_test, y_pred_rf)
plt.plot([0.5, 3],
        [0.5, 3],
        color= 'r',
        linestyle = '-',
        linewidth = 2)
plt.xlabel("Observed")
plt.ylabel("Predicted")


# In[54]:


plt.scatter(y_test, y_pred_xgb)
plt.plot([0.5, 3],
        [0.5, 3],
        color= 'r',
        linestyle = '-',
        linewidth = 2)
plt.xlabel("Observed")
plt.ylabel("Predicted")


# In[36]:


# Explain model predictions using SHAP
# Explanation objects for each model
explainer_lin = shap.LinearExplainer(lin_reg, X_train)
explainer_dt = shap.TreeExplainer(dt_reg)
explainer_rf = shap.TreeExplainer(rf_reg)
explainer_xgb = shap.TreeExplainer(xgb_model)


# In[37]:


# SHAP values for a single data point (example)
shap_values_lin = explainer_lin.shap_values(X_test)
shap_values_dt = explainer_dt.shap_values(X_test)
shap_values_rf = explainer_rf.shap_values(X_test)
shap_values_xgb = explainer_xgb.shap_values(X_test)


# In[38]:


shap_values_lin


# In[39]:


shap_values_lin = explainer_lin.shap_values(X_test.iloc[0])  # For the first test data point


# In[40]:


shap_values_lin = explainer_lin.shap_values(X_test)


# In[41]:


shap_values_lin


# In[42]:


shap.summary_plot(shap_values_lin, X_test, plot_type="bar")


# In[ ]:





# In[49]:


shap.summary_plot(shap_values_dt, X_test, plot_type="bar")


# In[50]:


shap.summary_plot(shap_values_rf, X_test, plot_type="bar")


# In[43]:


shap.summary_plot(shap_values_xgb, X_test, plot_type="bar")


# In[55]:


# Dependence plot - individual feature dependence on model prediction
shap.dependence_plot("VR_0311", shap_values_lin, X_test)  # Dependence plot for 'VR_0311' feature (Linear Regression)


# In[56]:


shap.dependence_plot("StmDia", shap_values_dt, X_test)  # Dependence plot for 'StmDia' feature (Decision Tree)


# In[57]:


shap.dependence_plot("HSW", shap_values_rf, X_test)  # Dependence plot for 'HSW' feature (Random Forest)


# In[58]:


shap.dependence_plot("PPP", shap_values_xgb, X_test)  # Dependence plot for 'PPP' feature (XGBoost)


# In[ ]:




