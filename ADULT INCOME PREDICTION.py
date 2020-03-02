#!/usr/bin/env python
# coding: utf-8

# # IMPORTING LIBRARIES FOR THE ANALYSIS

# Numpy library is used for scientific computing operations & mathematical functions purpose.
# Pandas library is used to manipulate and analyse the data.
# Matplotlib library is used for data visualization in 2D plotting.
# Seaborn library is used for data visualization of high level interface & informative statistical graphs.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # IMPORTING DATASET

# Importing the data for analysis.

# In[2]:


df_ad = pd.read_csv('C:\\Users\\SAITEJA\\Downloads\\adult dataset.csv')
df_ad


# Removing the unnecessary columns in the dataset.

# In[3]:


df = df_ad.drop(['fnlwgt'], axis = 1)
df


# Checking the shape of the dataset i.e., rows & columns.

# In[4]:


df.shape


# Checking the data types

# In[5]:


df.info()


# Checking the datatypes.

# In[6]:


df.dtypes, df.columns


# Describe the data is doing Exploratory Data Analysis for the dataset.

# In[7]:


df.describe()


# # PREPROCESSING 

# Replacing '?' with nan values to find the null values.

# In[8]:


df = df.replace('?',np.nan)
df


# Checking the null values.

# In[9]:


df.isnull().sum()


# Replacing null values using mode function.

# In[10]:


for col in ['workclass','occupation','native.country']:
    df[col].fillna(df[col].mode()[0], inplace =True)


# In[11]:


df


# In[12]:


df.isnull().sum()


# # VISUALIZATION

# # HISTOGRAM

# A histogram is one of the most frequently used data visualization techniques in machine learning. It represents the distribution of a continuous variable over a given interval or period of time. Histograms plot the data by dividing it into intervals called ‘bins’. It is used to inspect the underlying frequency distribution (eg. Normal distribution), outliers, skewness, etc.

# In[14]:


df.hist(figsize= (12,12))
plt.show()


# # PAIRPLOT

# Pairplot is used to understand the best set of features to explain a relationship between two variables or to form the most separated clusters. It also helps to form some simple classification models by drawing some simple lines or make linear separation in our dataset.

# In[15]:


sns.pairplot(df)


# # BOXPLOT

# A boxplot is a standardized way of displaying the distribution of data based on a five number summary (“minimum”, first quartile (Q1), median, third quartile (Q3), and “maximum”). It can tell you about your outliers and what their values are. It can also tell you if your data is symmetrical, how tightly your data is grouped, and if and how your data is skewed.

# In[16]:


df.boxplot(figsize = (15,15))
plt.show()


# # BARPLOT

# A bar plot is a plot that presents categorical data with rectangular bars with lengths proportional to the values that they represent. A bar plot shows comparisons among discrete categories. One axis of the plot shows the specific categories being compared, and the other axis represents a measured value.

# In[17]:


sns.barplot(x="income", y="age", data = df)
plt.show()


# Comparing Income with the age variable. income earning >50k salary has more percentage than the earning <50k salary.

# In[27]:


sns.barplot(x="income", y="education", data = df)
plt.show()


# Comparing Income with the education variable, Income earning >50k salary has more percentage than the earning <50k salary.

# In[ ]:


sns.barplot(x="income", y="education.num", data = df)
plt.show()


# Comparing Income with the education.num variable. income earning >50k salary has more percentage than the earning <50k salary.

# In[ ]:


sns.barplot(x='income', y='workclass', data =df)
plt.show()


# Comparing Income with the workclass variable. Income earning >50k salary has equal percentage to the earning <50k salary.

# In[ ]:


sns.barplot(x="income", y="occupation", data = df)
plt.show()


# Comparing Income with the occupation variable. income earning >50k salary has more percentage than the earning <50k salary.

# In[ ]:


sns.barplot(x="income", y="relationship", data = df)
plt.show()


# Comparing Income with the relationship variable. Income earning >50k salary has less percentage than the earning <50k salary.

# In[ ]:


sns.barplot(x="income", y="marital.status", data = df)
plt.show()


# Comparing Income with the marital.status variable. Income earning >50k salary has less percentage than the earning <50k salary.

# In[ ]:


sns.barplot(x="income", y="race", data = df)
plt.show()


# Comparing Income with the race variable. Income earning >50k salary has more percentage than the earning <50k salary.

# In[ ]:


sns.barplot(x="income", y="sex", data = df)
plt.show()


# Comparing Income with the male variable. Income earning >50k salary has more percentage than the earning <50k salary.

# In[ ]:


sns.barplot(x="income", y="hours.per.week", data = df)
plt.show()


# Comparing Income with the hours per week variable. Income earning >50k salary has less percentage than the earning <50k salary.

# In[ ]:


sns.barplot(x="income", y="native.country", data = df)
plt.show()


# Comparing Income with the native country variable. Income earning >50k salary has equal percentage than the earning <50k salary.

# # DISTPLOT

# A distplot plots a univariate distribution of observations. The distplot() function combines the matplotlib hist function with the seaborn kdeplot() and rugplot() functions.

# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(df['income'])
plt.show()


# Checking the variable distribution of income.

# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(df['age'])
plt.show()


# Checking the variable distribution of age.

# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(df['education'])
plt.show()


# Checking the variable distribution of education.

# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(df['education.num'])
plt.show()


# Checking the variable distribution of education.num.

# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(df['occupation'])
plt.show()


# Checking the variable distribution of occupation.

# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(df['workclass'])
plt.show()


# Checking the variable distribution of workclass.

# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(df['marital.status'])
plt.show()


# Checking the variable distribution of marital status.

# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(df['relationship'])
plt.show()


# Checking the variable distribution of relationship.

# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(df['race'])
plt.show()


# Checking the variable distribution of race.

# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(df['sex'])
plt.show()


# Checking the variable distribution of sex.

# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(df['hours.per.week'])
plt.show()


# Checking the variable distribution of hours per week.

# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(df['native.country'])
plt.show()


# Checking the variable distribution of native country.

# # KDEPLOT

# KDE Plot described as Kernel Density Estimate is used for visualizing the Probability Density of a continuous variable. It depicts the probability density at different values in a continuous variable. We can also plot a single graph for multiple samples which helps in more efficient data visualization.

# In[ ]:


plt.figure(figsize=(10,5))
sns.kdeplot(df['income'])
plt.show()


# Checking probability density of income variable.

# In[ ]:


plt.figure(figsize=(10,5))
sns.kdeplot(df['age'])
plt.show()


# Checking probability density of age variable.

# In[ ]:


plt.figure(figsize=(10,5))
sns.kdeplot(df['workclass'])
plt.show()


# Checking probability density of workclass variable.

# In[ ]:


plt.figure(figsize=(10,5))
sns.kdeplot(df['education'])
plt.show()


# Checking probability density of education variable.

# In[ ]:


plt.figure(figsize=(10,5))
sns.kdeplot(df['education.num'])
plt.show()


# Checking probability density of education num variable.

# In[ ]:


plt.figure(figsize=(10,5))
sns.kdeplot(df['occupation'])
plt.show()


# Checking probability density of income variable.

# In[ ]:


plt.figure(figsize=(10,5))
sns.kdeplot(df['relationship'])
plt.show()


# Checking probability density of relationship variable.

# In[ ]:


plt.figure(figsize=(10,5))
sns.kdeplot(df['marital.status'])
plt.show()


# Checking probability density of marital status variable.

# In[ ]:


plt.figure(figsize=(10,5))
sns.kdeplot(df['race'])
plt.show()


# Checking probability density of race variable.

# In[ ]:


plt.figure(figsize=(10,5))
sns.kdeplot(df['sex'])
plt.show()


# Checking probability density of sex variable.

# In[ ]:


plt.figure(figsize=(10,5))
sns.kdeplot(df['hours.per.week'])
plt.show()


# Checking probability density of hours per week variable.

# In[ ]:


plt.figure(figsize=(10,5))
sns.kdeplot(df['native.country'])
plt.show()


# Checking probability density of native country variable.

# # VIOLIN PLOT

# A violin plot is a method of plotting numeric data. It is similar to a box plot, with the addition of a rotated kernel density plot on each side.Typically a violin plot will include all the data that is in a box plot: a marker for the median of the data; a box or marker indicating the interquartile range; and possibly all sample points, if the number of samples is not too high.

# In[ ]:


sns.violinplot(x='income',y='age',data=df)


# Checking the higher density between income and the age , here mean lies in between 30-40 in earning <50K salary, earning >50k salary mean lies between 40-50 age.

# In[ ]:


sns.violinplot(x='income',y='workclass',data=df)


# Checking the higher density between income and the workclass , here mean lies in between 3 in earning <50K salary, earning >50k salary mean lies between 3 workclass.

# In[ ]:


sns.violinplot(x='income',y='education',data=df)


# Checking the higher density between income and the education , here mean lies in between 10-12.5 in earning <50K salary, earning >50k salary mean lies between 10-12.5 education.

# In[ ]:


sns.violinplot(x='income',y='education.num',data=df)


# Checking the higher density between income and the education num , here mean lies in between 7.5-10.0 in earning <50K salary, earning >50k salary mean lies between 10.0-12.5 education num.

# In[ ]:


sns.violinplot(x='income',y='occupation',data=df)


# Checking the higher density between income and the occupation , here mean lies in between 6 in earning <50K salary, earning >50k salary mean lies between 6-8 occupation.

# In[ ]:


sns.violinplot(x='income',y='marital.status',data=df)


# Checking the higher density between income and the marital status , here mean lies in between 3 in earning <50K salary, earning >50k salary mean lies between 2 marital status.

# In[ ]:


sns.violinplot(x='income',y='relationship',data=df)


# Checking the higher density between income and the relationship , here mean lies in between 1 in earning <50K salary, earning >50k salary mean lies between is 0 relationship.

# In[ ]:


sns.violinplot(x='income',y='race',data=df)


# Checking the higher density between income and the race , here mean lies in between 4 in earning <50K salary, earning >50k salary mean lies between 4 race.

# In[ ]:


sns.violinplot(x='income',y='sex',data=df)


# Checking the higher density between income and the sex , here mean lies in between 1 in earning <50K salary, earning >50k salary mean lies between 1 sex.

# In[ ]:


sns.violinplot(x='income',y='native.country',data=df)


# Checking the higher density between income and the native country , here mean lies in between 40 in earning <50K salary, earning >50k salary mean lies between 40 race.

# In[ ]:


sns.violinplot(x='income',y='hours.per.week',data=df)


# Checking the higher density between income and the hours per week , here mean lies in between 40 in earning <50K salary, earning >50k salary mean lies between 40 race.

# # JOINT PLOT

# Jointplot is seaborn library specific and can be used to quickly visualize and analyze the relationship between two variables and describe their individual distributions on the same plot.

# In[ ]:


sns.jointplot(x='income',y='age',data =df, kind = 'hex', gridsize = 20)


# Checking the relationship between income and age, describing the individual distribution. 

# In[ ]:


sns.jointplot(x='income',y='workclass',data =df, kind = 'hex', gridsize = 20)


# Checking the relationship between income and workclass, describing the individual distribution. 

# In[ ]:


sns.jointplot(x='income',y='education',data =df, kind = 'hex', gridsize = 20)


# Checking the relationship between income and education, describing the individual distribution. 

# In[ ]:


sns.jointplot(x='income',y='education.num',data =df, kind = 'hex', gridsize = 20)


# Checking the relationship between income and education num, describing the individual distribution. 

# In[ ]:


sns.jointplot(x='income',y='occupation',data =df, kind = 'hex', gridsize = 20)


# Checking the relationship between income and occupation, describing the individual distribution. 

# In[ ]:


sns.jointplot(x='income',y='marital.status',data =df, kind = 'hex', gridsize = 20)


# Checking the relationship between income and marital status, describing the individual distribution. 

# In[ ]:


sns.jointplot(x='income',y='relationship',data =df, kind = 'hex', gridsize = 20)


# Checking the relationship between income and relationship, describing the individual distribution. 

# In[ ]:


sns.jointplot(x='income',y='race',data =df, kind = 'hex', gridsize = 20)


# Checking the relationship between income and race, describing the individual distribution. 

# In[ ]:


sns.jointplot(x='income',y='sex',data =df, kind = 'hex', gridsize = 20)


# Checking the relationship between income and sex, describing the individual distribution. 

# In[ ]:


sns.jointplot(x='income',y='hours.per.week',data =df, kind = 'hex', gridsize = 20)


# Checking the relationship between income and hours per week, describing the individual distribution. 

# In[ ]:


sns.jointplot(x='income',y='native.country',data =df, kind = 'hex', gridsize = 20)


# Checking the relationship between income and native country, describing the individual distribution. 

# # COUNTPLOT

# seaborn.countplot is a barplot where the dependent variable is the number of instances of each instance of the independent variable.

# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(df['income'], hue = df['age'], palette = 'rainbow', edgecolor = [(0,0,0), (0,0,0)])
plt.show()


# Checking the countplot of income and age, age 23 has more percentage of earning <50k salary, where as 35 and 44 are earning >50k salary.

# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(df['income'], hue = df['workclass'], palette = 'rainbow', edgecolor = [(0,0,0), (0,0,0)])
plt.show()


# Checking the countplot of income and the workclass, here both the private employee not inc are earning <50K salary and >50k salary, Slef emp not inc has more percentage of earning <50k salary than earning the >50k salary.

# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(df['income'], hue = df['education'], palette = 'rainbow', edgecolor = [(0,0,0), (0,0,0)])
plt.show()


# Checking the countplot of income and the education, bachelors are earning >50k salary and HS-grad has more percentage of earning <50k salary.

# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(df['income'], hue = df['education.num'], palette = 'rainbow', edgecolor = [(0,0,0), (0,0,0)])
plt.show()


# Checking the countplot of income and the education.num,13 (bachelors) are earning >50k salary and 9 (HS-grad) has more percentage of earning <50k salary.

# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(df['income'], hue = df['occupation'], palette = 'rainbow', edgecolor = [(0,0,0), (0,0,0)])
plt.show()


# Checking the countplot of income and occupation,prof speciality are earning >50k salary & <50k salary. Other than that exec-managerial is also has earning of <50k and >50k salary.

# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(df['income'], hue = df['marital.status'], palette = 'rainbow', edgecolor = [(0,0,0), (0,0,0)])
plt.show()


# Checking the countplot of income and the marital status, married civ spouse has earning <50k salary & >50k salary, and never married, married civ-spouse has more percentage than the other variables of earning <50k salary.

# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(df['income'], hue = df['race'], palette = 'rainbow', edgecolor = [(0,0,0), (0,0,0)])
plt.show()


# Checking the countplot of income and race, White's are earning <50k salary & >50k salary, after that black's are earning <50k salary & >50k salary. 

# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(df['income'], hue = df['sex'], palette = 'rainbow', edgecolor = [(0,0,0), (0,0,0)])
plt.show()


# Checking the countplot of income and sex(gender), male has more percentage of earning >50k salary and <50k salary. And female earning >50k salary has less percentage.

# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(df['income'], hue = df['native.country'], palette = 'rainbow', edgecolor = [(0,0,0), (0,0,0)])
plt.show()


# Checking the countplot of income and native country, united states has more percentage of earning >50k salary.

# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(df['income'], hue = df['hours.per.week'], palette = 'rainbow', edgecolor = [(0,0,0), (0,0,0)])
plt.show()


# Checking the countplot of income and hours per week. Average Hours per week is 45.

# # FACTORPLOT

# A factor plot is simply the same plot generated for different response and factor variables and arranged on a single page. The underlying plot generated can be any univariate or bivariate plot. The scatter plot is the most common application.

# In[32]:


s = sns.factorplot(x="age",y="income",data=df,kind="bar",size = 6,palette = "muted")
s.despine(left=True)
s= s.set_ylabels(">50K probability")


# Checking the factorplot of income and age.

# In[33]:


s = sns.factorplot(x="workclass",y="income",data=df,kind="bar",size = 6,palette = "muted")
s.despine(left=True)
s= s.set_ylabels(">50K probability")


# Checking the factorplot of income and workclass.

# In[34]:


s = sns.factorplot(x="education",y="income",data=df,kind="bar",size = 6,palette = "muted")
s.despine(left=True)
s= s.set_ylabels(">50K probability")


# Checking the factorplot of income and education.

# In[35]:


s = sns.factorplot(x="education.num",y="income",data=df,kind="bar",size = 6,palette = "muted")
s.despine(left=True)
s= s.set_ylabels(">50K probability")


# Checking the factorplot of income and education.num.

# In[36]:


s = sns.factorplot(x="occupation",y="income",data=df,kind="bar",size = 6,palette = "muted")
s.despine(left=True)
s= s.set_ylabels(">50K probability")


# Checking the factorplot of income and occupation.

# In[37]:


s = sns.factorplot(x="relationship",y="income",data=df,kind="bar",size = 6,palette = "muted")
s.despine(left=True)
s= s.set_ylabels(">50K probability")


# Checking the factorplot of income and relationship.

# In[38]:


s = sns.factorplot(x="marital.status",y="income",data=df,kind="bar",size = 6,palette = "muted")
s.despine(left=True)
s= s.set_ylabels(">50K probability")


# Checking the factorplot of income and marital status.

# In[39]:


s = sns.factorplot(x="race",y="income",data=df,kind="bar",size = 6,palette = "muted")
s.despine(left=True)
s= s.set_ylabels(">50K probability")


# Checking the factorplot of income and race.

# In[40]:


s = sns.factorplot(x="sex",y="income",data=df,kind="bar",size = 6,palette = "muted")
s.despine(left=True)
s= s.set_ylabels(">50K probability")


# Checking the factorplot of income and sex(gender).

# In[41]:


s = sns.factorplot(x="native.country",y="income",data=df,kind="bar",size = 6,palette = "muted")
s.despine(left=True)
s= s.set_ylabels(">50K probability")


# Checking the factorplot of income and native country.

# In[42]:


s = sns.factorplot(x="hours.per.week",y="income",data=df,kind="bar",size = 6,palette = "muted")
s.despine(left=True)
s= s.set_ylabels(">50K probability")


# Checking the factorplot of income and hours per week.

# # FACETGRID 

# A useful approach to explore medium-dimensional data, is by drawing multiple instances of the same plot on different subsets of your dataset. This technique is commonly called as lattice, or trellis plotting, and it is related to the idea of small multiples.

# In[22]:


g = sns.FacetGrid(df, col='income')
g = g.map(sns.distplot, "age")
plt.show()


# Checking the facetgrid of income and age. Graph goes through different subsets of dataset. Age 20-40 has max peak dimension distance of earning <50k salary and age 40-60 has max peak dimension distance of earning >50k salary.

# In[31]:


g = sns.FacetGrid(df, col='income')
g = g.map(sns.distplot, "workclass")
plt.show()


# Checking the facetgrid of income and workclass. Graph goes through different subsets of dataset. Workclass of 3 is having max dimension distance of earning <50k salary & >50k salary.

# In[ ]:


g = sns.FacetGrid(df, col='income')
g = g.map(sns.distplot, "education")
plt.show()


# Checking the facetgrid of income and education. Graph goes through different subsets of dataset. education of 11 is having max dimension distance of earning <50k salary & education 9 is having max dimension distance of earning >50k salary.

# In[24]:


g = sns.FacetGrid(df, col='income')
g = g.map(sns.distplot, "education.num")
plt.show()


# Checking the facetgrid of income and education num. Graph goes through different subsets of dataset. education num of 9 is having max dimension distance of earning <50k salary & education 13 is having max dimension distance of earning >50k salary.

# In[114]:


g = sns.FacetGrid(df, col='income')
g = g.map(sns.distplot, "occupation")
plt.show()


# Checking the facetgrid of income and occupation. Graph goes through different subsets of dataset. occupation of 9 is having max dimension distance of earning <50k salary & occupation 3 is having max dimension distance of earning >50k salary.

# In[116]:


g = sns.FacetGrid(df, col='income')
g = g.map(sns.distplot, "relationship")
plt.show()


# Checking the facetgrid of income and relationship. Graph goes through different subsets of dataset. relationship of 1 is having max dimension distance of earning <50k salary & relationship of 5 is having max dimension distance of earning >50k salary.

# In[117]:


g = sns.FacetGrid(df, col='income')
g = g.map(sns.distplot, "marital.status")
plt.show()


# Checking the facetgrid of income and marital status. Graph goes through different subsets of dataset. marital status of 4 is having max dimension distance of earning <50k salary & marital status of 2 is having max dimension distance of earning >50k salary.

# In[118]:


g = sns.FacetGrid(df, col='income')
g = g.map(sns.distplot, "race")
plt.show()


# Checking the facetgrid of income and race. Graph goes through different subsets of dataset. race of 4 is having max dimension distance of earning <50k salary & >50k salary.

# In[122]:


g = sns.FacetGrid(df, col='income')
g = g.map(sns.distplot, "sex")
plt.show()


# Checking the facetgrid of income and sex(gender). Graph goes through different subsets of dataset. sex of 0 is having max dimension distance of earning <50k salary & sex of 1 is having max dimension distance of earning >50k salary.

# In[121]:


g = sns.FacetGrid(df, col='income')
g = g.map(sns.distplot, "hours.per.week")
plt.show()


# Checking the facetgrid of income and hours per week. Graph goes through different subsets of dataset. Hours per week of 40 is having max dimension distance of earning <50k salary & >50k salary.

# In[119]:


g = sns.FacetGrid(df, col='income')
g = g.map(sns.distplot, "native.country")
plt.show()


# Checking the facetgrid of income and native country. Graph goes through different subsets of dataset. native country of 38 is having max dimension distance of earning <50k salary & >50k salary.

# # LABEL ENCODING

# Label Encoding refers to converting the labels into numeric form so as to convert it into the machine-readable form. Machine learning algorithms can then decide in a better way on how those labels must be operated. It is an important pre-processing step for the structured dataset in supervised learning.Label encoding convert the data in machine readable form, but it assigns a unique number(starting from 0) to each class of data. 

# In[25]:


from sklearn.preprocessing import LabelEncoder
lr= LabelEncoder()
df['workclass']=lr.fit_transform(df['workclass'])
df['education']=lr.fit_transform(df['education'])
df['marital.status']=lr.fit_transform(df['marital.status'])
df['occupation']=lr.fit_transform(df['occupation'])
df['relationship']=lr.fit_transform(df['relationship'])
df['race']=lr.fit_transform(df['race'])
df['sex']=lr.fit_transform(df['sex'])
df['income']=lr.fit_transform(df['income'])
df['native.country']=lr.fit_transform(df['native.country'])
df['income']=lr.fit_transform(df['income'])


# In[26]:


df


# # MIN-MAX SCALER

# The technique re-scales a feature or observation value with distribution value between 0 and 1 or  -1 to 1 if there are negative values.

# In[19]:


from sklearn.preprocessing import minmax_scale

df[['capital.loss', 'capital.gain']]=minmax_scale(df[['capital.loss','capital.gain']])
df


# # CORRELATION

# Correlation is a statistical technique that can show whether and how strongly pairs of variables are related Correlation values range between -1 and 1.

# In[20]:


df.corr()


# # HEATMAP

# A heatmap is a graphical representation of data that uses a system of color-coding to represent different values.

# In[135]:


hmap = df.corr()
plt.subplots(figsize=(12, 9))
sns.heatmap(hmap, vmax=.8,annot=True,cmap="BrBG", square=True);


# # INPUT & OUTPUT VARIABLE SEPARATION

# In[27]:


x= df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]]
x.head()

y = df.iloc[:,[13]]
y.head()


# # SPLITTING THE DATASET INTO TRAINING & TESTING.

# Splitting the dataset into training and testing dataset to find the accuracy of the models.

# In[72]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


# In[73]:


x_train


# In[30]:


y_train


# In[31]:


y_train


# In[32]:


y_test


# # IMPORTING LIBRARIES TO GET THE TP & TN,FP & FN, ACCURACY.

# Importing libraries to find the confusion matrix, accuracy score.

# In[33]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# # LOGISTIC REGRESSION CLASSIFICATION

# In[ ]:


#The target variable(or output), y, can take only discrete values for given set of features(or inputs), X.


# In[34]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train, y_train)
y_pred =lr.predict(x_test)


# In[35]:


cm = confusion_matrix(y_pred, y_test)
cm


# In[36]:


a_s = accuracy_score(y_pred,y_test)
a_s


# # DECISION TREE CLASSIFICATION

# Decision tree uses the tree representation to solve the problem in which each leaf node corresponds to a class label and attributes are represented on the internal node of the tree.
# 
# Decision tree is the most powerful and popular tool for classification and prediction.
# 
# A tree can be “learned” by splitting the source set into subsets based on an attribute value test. This process is repeated on each derived subset in a recursive manner called recursive partitioning. The recursion is completed when the subset at a node all has the same value of the target variable, or when splitting no longer adds value to the predictions.

# In[37]:


from sklearn.tree import DecisionTreeClassifier

DTclassifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
DTclassifier.fit(x_train, y_train)
y_pred = DTclassifier.predict(x_test)
y_pred


# In[38]:


cm = confusion_matrix(y_pred, y_test)
cm


# In[39]:


a_s = accuracy_score(y_pred,y_test)
a_s


# # K NEAREST NEIGHBOUR CLASSIFICATION

# KNN classifier stores all the values and classifies new cases by the majority vote by the k neighbour. it does not make any underlying assumptions about the distribution of data (as opposed to other algorithms such as GMM, which assume a Gaussian distribution of the given data).
# 
# We give some training data, which classifies coordinates into groups identified by an attribute.
# K can be kept as an odd number so that we can calculate a clear majority in the case where only two groups are possible. With increasing K, we get smoother, more defined boundaries across different classifications. Also, the accuracy of the above classifier increases as we increase the number of data points in the training set.

# In[40]:


from sklearn.neighbors import KNeighborsClassifier

KNclassifier = KNeighborsClassifier(n_neighbors=5)  
KNclassifier.fit(x_train, y_train)
y_pred = KNclassifier.predict(x_test)
y_pred


# In[41]:


cm = confusion_matrix(y_test,y_pred)
cm


# In[42]:


a_s = accuracy_score(y_pred,y_test)
a_s


# # NAIVE BAYES CLASSIFICATION

# Bayes Theorem finds the probability of an event occurring given the probability of another event that has already occurred.Naive Bayes classifiers are a collection of classification algorithms based on Bayes’ Theorem. It is not a single algorithm but a family of algorithms where all of them share a common principle, i.e. every pair of features being classified is independent of each other.The assumptions made by Naive Bayes are not generally correct in real-world situations. In-fact, the independence assumption is never correct but often works well in practice.

# In[43]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train,y_train)
y_pred= nb.predict(x_test)
y_pred


# In[44]:


cm = confusion_matrix(y_test,y_pred)
cm


# In[45]:


a_s = accuracy_score(y_pred,y_test)
a_s


# # RANDOM FOREST CLASSIFICATION

# Ensemble learning helps improve machine learning results by combining several models. This approach allows the production of better predictive performance compared to a single model. Basic idea is to learn a set of classifiers (experts) and to allow them to vote.Random Forest is an extension over bagging. Each classifier in the ensemble is a decision tree classifier and is generated using a random selection of attributes at each node to determine the split. During classification, each tree votes and the most popular class is returned.

# In[46]:


from sklearn.ensemble import RandomForestClassifier

rfclassifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rfclassifier.fit(x_train, y_train)
y_pred = rfclassifier.predict(x_test)
y_pred


# In[47]:


cm = confusion_matrix(y_test,y_pred)
cm


# In[48]:


a_s = accuracy_score(y_pred,y_test)
a_s


# # SUPPORT VECTOR MACHINE CLASSIFICATION

# A Support Vector Machine (SVM) is a discriminative classifier formally defined by a separating hyperplane. In other words, given labeled training data (supervised learning), the algorithm outputs an optimal hyperplane which categorizes new examples.In addition to performing linear classification, SVMs can efficiently perform a non-linear classification, implicitly mapping their inputs into high-dimensional feature spaces.

# In[49]:


from sklearn.svm import SVC

SVclassifier = SVC(kernel = 'rbf', random_state = 0)
SVclassifier.fit(x_train, y_train)
y_pred = SVclassifier.predict(x_test)
y_pred


# In[50]:


cm = confusion_matrix(y_test,y_pred)
cm


# In[51]:


a_s = accuracy_score(y_pred,y_test)
a_s


# # ADABOOST CLASSIFICATION

# Boosting algorithms seek to improve the prediction power by training a sequence of weak models, each compensating the weaknesses of its predecessors.

# In[52]:


from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(random_state=1)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
y_pred


# In[53]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[54]:


a_s = accuracy_score(y_pred,y_test)
a_s


# # XG BOOST CLASSIFICATION

# XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. It is a perfect combination of software and hardware optimization techniques to yield superior results using less computing resources in the shortest amount of time.

# In[55]:


from xgboost import XGBClassifier

XGBclf = XGBClassifier()
XGBclf.fit(x_train, y_train)
y_pred = XGBclf.predict(x_test)
y_pred


# In[56]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[57]:


a_s = accuracy_score(y_test, y_pred)
a_s


# # BAGGING CLASSIFICATION

# A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions to form a final prediction. Bagging (Bootstrap Aggregation) is used to reduce the variance of a decision tree.

# In[58]:


from sklearn.ensemble import BaggingClassifier

bclassifier = BaggingClassifier(random_state=1)
bclassifier.fit(x_train,y_train)
y_pred = bclassifier.predict(x_test)
y_pred


# In[59]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[60]:


a_s = accuracy_score(y_test, y_pred)
a_s


# # EXTRA TREES CLASSIFICATION

# It is a type of ensemble learning technique which aggregates the results of multiple de-correlated decision trees collected in a “forest” to output it’s classification result.

# In[61]:


from sklearn.ensemble import ExtraTreesClassifier

etclassifier = ExtraTreesClassifier(random_state=1)
etclassifier.fit(x_train, y_train)
y_pred = bclassifier.predict(x_test)
y_pred


# In[62]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[63]:


a_s = accuracy_score(y_test, y_pred)
a_s


# # LINEAR DISCRIMINANT ANALYSIS

# Discriminant Function Analysis is a dimensionality reduction technique which is commonly used for the supervised classification problems. It is used for modeling differences in groups i.e. separating two or more classes. It is used to project the features in higher dimension space into a lower dimension space.

# In[66]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=5)
lda.fit(x_train,y_train)
y_pred = lda.predict(x_test)
y_pred


# In[67]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[68]:


a_s = accuracy_score(y_test, y_pred)
a_s


# # CONCLUSION

# The result XGBoost classifier gives the 86.41% prediction comparing to other algorithms.
