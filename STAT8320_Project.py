#!/usr/bin/env python
# coding: utf-8

# # Group Project Wilson & Mills

# In[4]:


#Libraries Used
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from sklearn.cluster import KMeans
import missingno as msno
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.express as px


# ## Import Data and Review

# In[6]:


#Import Dataset
original_df = pd.read_csv("US_Accidents_March23.csv")
original_df.columns.tolist


# In[7]:


#Data Characteristics
original_df.head(5)


# In[8]:


#Data Characteristics
original_df.describe(include = 'all')


# In[9]:


#Data Characteristics
original_df.isna().sum()


# # Clean Dataset for Experiment

# In[11]:


#Remove fields to not be modeled and make values useable for multivariate analysis
GA_DF = original_df.loc[original_df['State'] == 'GA'].drop(columns =['Source',
                                                                     'ID',
                                                                     'Description',
                                                                     'State',
                                                                     'Street',
                                                                     'End_Lat',
                                                                     'End_Lng',
                                                                     'End_Time',
                                                                     'City',
                                                                     'County',
                                                                     'Country',
                                                                     'Timezone',
                                                                     'Zipcode',
                                                                     'Bump',
                                                                     'Weather_Condition',
                                                                     'Airport_Code',
                                                                     'Wind_Direction',
                                                                     'Weather_Timestamp',
                                                                     'Civil_Twilight',
                                                                     'Nautical_Twilight',
                                                                     'Traffic_Calming',
                                                                     'Roundabout',
                                                                     'Turning_Loop',
                                                                     'Astronomical_Twilight',
                                                                    'Sunrise_Sunset',
                                                                    'Start_Time'],
                                                           axis=1
                                                          ).rename(
    columns ={'Distance(mi)' : 'Distance'}
)
with pd.option_context('future.no_silent_downcasting', True):
    GA_DF.loc[:,'Amenity':'Traffic_Signal'] = GA_DF.loc[:,'Amenity':'Traffic_Signal'].astype(int)


# In[12]:


#Find missingness relationship
msno.heatmap(GA_DF, cmap='YlGnBu')


# In[13]:


#Remove Missing
GA_DF = GA_DF.dropna()


# In[14]:


#Verify all missing values no longer present
msno.heatmap(GA_DF, cmap='YlGnBu')


# In[15]:


#Experiment Dataset characteristics
GA_DF.describe(include = 'all')


# In[16]:


#Experiment Dataset characteristics
GA_DF.head(20)


# In[17]:


#Location Overview
levels, categories = pd.factorize(sorted(GA_DF['Severity'], reverse = False)) 
scatter = plt.scatter(GA_DF['Start_Lng'],GA_DF['Start_Lat'], s=1, c=levels)
plt.legend(scatter.legend_elements()[0], categories, title='Severity')
plt.gca().set(xlabel='Longitude', ylabel='Distance of Road Impacted (mi)', title='Georgias Most Impactful Accidents')


# In[18]:


#Create dependent and Independent variables
X = GA_DF.loc[:, ~GA_DF.columns.isin(['Severity','Distance'])]
Y = GA_DF[['Severity','Distance']]


# In[19]:


#Independent variable characteristics
X.describe()


# # Assumption Tests

# In[21]:


#Correlation map
sb.heatmap(X.corr(), vmax=1., square=True)


# In[22]:


#All independents Q-Q Plot
fig = sm.qqplot(X, line='45')
plt.show()


# In[23]:


#Test Multicollinearity
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(X.shape[1])]

vif_data


# # Baseline Model

# In[97]:


#Split test and train data
X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size=0.3, random_state=1) 


# In[99]:


#Build basic multivariate model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
regression_predictions = regr.predict(X_test) 


# In[101]:


regr.intercept_


# In[103]:


regr.coef_


# In[105]:


#MSE
mean_squared_error(y_test, regression_predictions) 


# In[107]:


#MAE
mean_absolute_error(y_test, regression_predictions) 


# In[109]:


#R2
r2_score(y_test, regression_predictions) 


# # PCA

# In[112]:


#Scaler Transformation
scaler = StandardScaler()


# In[114]:


scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[116]:


pca = PCA(.95).fit(X_train)
pca.n_components_


# In[118]:


plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Factors')
plt.ylabel('Variance (%)')
plt.title('Pre-PCA Transformation Explained Variance')
plt.show()


# In[120]:


#PCA Transformation
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)


# In[122]:


#PCA Model
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(X_train,y_train['Severity'].to_numpy())
predictions = logisticRegr.predict(X_test) 


# In[124]:


#MSE
mean_squared_error(y_test['Severity'], predictions) 


# In[126]:


#MAE
mean_absolute_error(y_test['Severity'], predictions) 


# In[128]:


#R2
r2_score(y_test['Severity'], predictions) 


# In[130]:


#Model coefficients
pca = PCA()
pca.fit(X)
explained_variance_ratio = pca.explained_variance_ratio_
explained_variance_ratio


# In[132]:


#Scree Plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--')
plt.title('PCA Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.xticks(np.arange(1, len(explained_variance_ratio) + 1, 1))
plt.show()


# In[134]:


#Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')
plt.title('Explained Variance')
plt.show()


# In[154]:


#Store components 
pca = PCA(n_components=3)
components = pca.fit_transform(X)


# In[178]:


#MANOVA
data_pca = pd.concat([Y.reset_index(drop=True), pd.DataFrame(components,columns = ['PC1','PC2','PC3']).reset_index(drop=True)],axis=1)
formula = 'Severity + Distance ~ PC1 + PC2 +PC3'
manova_pca = MANOVA.from_formula(formula, data = data_pca)
results = manova_pca.mv_test()
print(results)


# In[140]:


#PC1 vs PC2
fig = px.scatter(components, x=0, y=1, color=GA_DF['Severity'],labels={
                     "0": "PC1",
                     "1": "PC2",
                     "color": "Severity"
                 },
                title="PCA Scatterplot (PC1 vs. PC2)")
fig.show()


# In[141]:


#PC1 vs PC3
fig = px.scatter(components, x=0, y=2, color=GA_DF['Severity'],labels={
                     "0": "PC1",
                     "2": "PC3",
                     "color": "Severity"
                 },
                title="PCA Scatterplot (PC1 vs. PC3)")
fig.show()


# In[143]:


#PC2 vs PC3
fig = px.scatter(components, x=1, y=2, color=GA_DF['Severity'],labels={
                     "1": "PC2",
                     "2": "PC3",
                     "color": "Severity"
                 },
                title="PCA Scatterplot (PC2 vs. PC3)")
fig.show()


# In[146]:


# 3D Plot
total_var = pca.explained_variance_ratio_.sum() * 100

fig = px.scatter_3d(
    components, x=0, y=1, z=2, color=GA_DF['Severity'],
    title=f'Total Explained Variance: {total_var:.2f}%',
    labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
)
fig.show()


# In[148]:


#Loadings Plot
loadings = pca.components_  
for i in range(3):
    plt.plot(loadings[i], label=f'PC {i+1}', marker='o')

plt.title('Loading Plot')
plt.xlabel('Features')
plt.ylabel('Loading Value')
plt.ylim(-1, 1)
plt.legend()
plt.grid(True)
plt.show()


# In[150]:


# Sum of loadings 
loadings[0]+loadings[1]+loadings[2]


# In[152]:


# Q-Q plots against normal distribution
get_ipython().run_line_magic('matplotlib', 'tk')
fig, axes = plt.subplots(6, 3, figsize=(15, 5), layout = 'constrained')
axes = axes.flatten()
for i, col in enumerate(X.columns):
    stats.probplot(GA_DF[col], dist="norm", plot=axes[i])
    axes[i].set_title(f"Q-Q Plot of {col}")
plt.show()


# # LASSO Model

# In[ ]:


#LASSO train and test scaler transform
LassoX_train = scaler.fit_transform(X_train)
LassoX_test = scaler.transform(X_test)


# In[ ]:


#Lasso model
model = Lasso()
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
model = MultiTaskLassoCV(alphas=np.arange(0, 1, 0.01), cv=cv, n_jobs=-1).fit(LassoX_train, y_train)
print('alpha: %f' % model.alpha_)


# In[ ]:


#Prediction to test MSE et all
predictions = model.predict(LassoX_test) 


# In[ ]:


#MSE
mean_squared_error(y_test, predictions)


# In[ ]:


#MAE
mean_absolute_error(y_test, predictions) 


# In[ ]:


#R2
r2_score(y_test, predictions) 


# # All of United States for Comparison

# In[181]:


#Apply same data cleaning as GA dataset
US_DF = original_df.loc[original_df['State'] != 'GA'].dropna().drop(columns =['Source',
                                                                     'ID',
                                                                     'Description',
                                                                     'State',
                                                                     'Street',
                                                                     'End_Lat',
                                                                     'End_Lng',
                                                                     'End_Time',
                                                                     'City',
                                                                     'County',
                                                                     'Country',
                                                                     'Timezone',
                                                                     'Zipcode',
                                                                     'Bump',
                                                                     'Weather_Condition',
                                                                     'Airport_Code',
                                                                     'Wind_Direction',
                                                                     'Weather_Timestamp',
                                                                     'Civil_Twilight',
                                                                     'Nautical_Twilight',
                                                                     'Traffic_Calming',
                                                                     'Roundabout',
                                                                     'Turning_Loop',
                                                                     'Astronomical_Twilight',
                                                                    'Sunrise_Sunset',
                                                                    'Start_Time'],
                                                           axis=1
                                                          ).rename(
    columns ={'Distance(mi)' : 'Distance'}
)
with pd.option_context('future.no_silent_downcasting', True):
    US_DF.loc[:,'Amenity':'Traffic_Signal'] = US_DF.loc[:,'Amenity':'Traffic_Signal'].astype(int)


# In[183]:


#Split test and training
X_all = US_DF.loc[:, ~US_DF.columns.isin(['Severity','Distance'])]
Y_all = US_DF[['Severity','Distance']]


# In[185]:


#Predictions to test MSE et all
all_predictions = regr.predict(X_all) 


# In[187]:


#MSE
mean_squared_error(Y_all, all_predictions)


# In[189]:


#MAE
mean_absolute_error(Y_all, all_predictions) 


# In[191]:


#R2
r2_score(Y_all, all_predictions) 


# In[ ]:




