#!/usr/bin/env python
# coding: utf-8

# # <u>Step 1</u>
# ## Problem Statement:
# #### To predict if a customer will pay their premium on time or not.

# # <u>Step 2</u>
# ## Hypothesis Generation
# Following can be the factors that can be used to predict if a customer will pay their premium on time:
# 1. Whether previous premium is payed.
# 2. Time of previous payment
# 3. Type of job
# 4. Area of residence
# 5. Age

# # <u>Step 3</u>
# ## Data Extraction
# Dataset was provided beforehand. 

# # <u>Step 4</u>
# ## Data Exploration
# 
# 1) Reading the data. <br>
# 2) Variable Identification. <br>
# 3) Univariate Analysis.<br>
# 4) Bivariate Analysis.<br>
# 5) Missing Value Treatment.<br>
# 6) Outlier Treatment.<br>
# 7) Variable Transformation.<br>

# In[170]:


# Importing all the required libraries and modules

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
from sklearn import *


# #### 1) Reading the Dataset

# In[134]:


##---Reading the Data---

df= pd.read_csv("C:\\Users\\devesh\\Desktop\\Git Repositories\\Predicting-if-a-customer-will-default-their-next-premium\\train.csv")


# In[135]:


df.head()


# In[136]:


df.tail()


# In[5]:


# Application Scores are a type of score used by lenders to help them accept suitable customers
# based purely on the information given in the credit application form

# Insurance companies can sell their policies through numerous distribution channels, such platforms are known as sourcing channels.


# In[6]:


df.shape


# In[7]:


df.columns


# #### 2) Variable Identification
# 
# ###### -Dependepent Variables: The target variable <br>
# 
# ###### -Independent Variables:
# 1) id <br>
# 2) perc_premium_paid_by_cash_credit <br>
# 3) age_in_days <br>
# 4) Income <br>
# 5) Count_3-6_months_late <br>
# 6) Count_6-12_months_late <br>
# 7) Count_more_than_12_months_late <br>
# 8) application_underwriting_score <br>
# 9) no_of_premiums_paid <br>
# 10) sourcing_channel <br>
# 11) residence_area_type <br>

# In[8]:


df.dtypes


# #### D-types:
# 
# ##### Categorical Variables:
# sourcing_channel, residence_area_type
# 
# ##### Continuous Variables:
# id, perc_premium_paid_by_cash_credit, age_in_days, Income, Count_3-6_months_late, Count_6-12_months_late, Count_more_than_12_months_late, application_underwriting_score, no_of_premiums_paid, target
# 
# 
# ##### Since our target variable is dependant we will go forward with <u>Supervised Learning</u>

# #### 3) Univariate Analysis
# Univariate analysis explores each variable in a data set, separately. It looks at the range of values, as well as the central tendency of the values.

# In[9]:


df.describe()


# In[10]:


# Here we can see that count_3-6_months_late, count_6-12_months_late, count_more_than_12_months_late, 
# and application_underwriting_score have missing values. We will fill these missing values later in this
# segment. First, we will draw histograms and box-plots for each independent continuous variable to see their distributions and check if they have outliers. 


# In[11]:


df['perc_premium_paid_by_cash_credit'].plot.hist()


# In[12]:


# here we can see that majority of people have not paid their premiums in cash


# In[13]:


# using a boxplot to check the presence of outliers

df['perc_premium_paid_by_cash_credit'].plot.box()


# In[14]:


# We will be using mean to find the central tendency of columns with no outliers and median for columns with outliers.


# In[15]:


print(df["perc_premium_paid_by_cash_credit"].mean())


# In[16]:


# This means that on an average, people have paid 31.42% of their premium in cash.


# In[17]:


df["age_in_days"].plot.hist()


# In[18]:


# We can see a majority of our customers lie in the age range of 54-68 years (20,000-25,000 days).


# In[19]:


df["age_in_days"].plot.box()


# In[20]:


# the box plot here depicts that the column 'age_in_days' has outliers
# we will make a separate list for the coulmns having outliers for future reference.


# In[3]:


outliers_list = []
outliers_list.append("age_in_days")
outliers_list


# In[22]:


print((df["age_in_days"].min())/365)
print((df["age_in_days"].max())/365)
print((df["age_in_days"].median())/365)


# In[23]:


# Our youngest customer is approx. 21 years.
# oldest customer is approx. 103 years.
# and the average age of our customers approx. 51 years.


# In[24]:


df['Income'].plot.hist()


# In[25]:


df['Income'].plot.box()


# In[26]:


outliers_list.append("Income")
outliers_list


# In[27]:


print(df["Income"].min())
print(df["Income"].max())
print(df["Income"].median())


# In[28]:


# The lowest income of among our customers is Rs.24,030 ,
# while the highest being Rs.9,02,62,600
# with an average of Rs.1,66,560
# Note that customers with lower income are less likely to pay premium on time.


# In[29]:


df["application_underwriting_score"].plot.hist()


# In[30]:


# This is a left-skewed distribution telling us that approximately 40,000 customers have a good (near 100) application underwriting score.


# In[31]:


df["application_underwriting_score"].plot.box()


# In[32]:


outliers_list.append("application_underwriting_score")
outliers_list


# In[33]:


print(df["application_underwriting_score"].min())
print(df["application_underwriting_score"].max())
print(df["application_underwriting_score"].median())


# In[34]:


# Maximum A.U.S. = 99.89
# Minimum A.U.S. = 91.9
# Average A.U.S. = 99.21 


# In[35]:


df["no_of_premiums_paid"].plot.hist()


# In[36]:


# This is a right skewed distribution telling us that approximately 
# 35,000 customers have paid atleast 10 premiums. 
# Note that the number of customers decrease as the number of premiums paid increases, 
# which means they both have a negative correlation.


# In[37]:


df['no_of_premiums_paid'].plot.box()


# In[38]:


outliers_list.append('no_of_premiums_paid')
outliers_list


# In[39]:


print(df["no_of_premiums_paid"].min())
print(df["no_of_premiums_paid"].max())
print(df["no_of_premiums_paid"].median())


# In[40]:


# Lease number of premiums paid by a customer = 2
# Most number of premiums paid by a customer = 60
# Average premiums paid = 10


# In[41]:


df["target"].plot.hist()


# In[42]:


# This shows that a high number of customers are likely to pay their premiums on time.


# In[43]:


# I have treated Count_3-6_months_late, Count_6-12_months_late, and Count_more_than_12_months_late as categorical variables as there is not much variation in their values.


# In[44]:


df["Count_3-6_months_late"].value_counts()


# In[45]:


df["Count_3-6_months_late"].value_counts().plot.bar()


# In[46]:


df["Count_6-12_months_late"].value_counts()


# In[47]:


df["Count_6-12_months_late"].value_counts().plot.bar()


# In[48]:


df["Count_more_than_12_months_late"].value_counts()


# In[49]:


df["Count_more_than_12_months_late"].value_counts().plot.bar()


# In[50]:


# we can observe that alot of our customers have paid their premiums on time.


# In[51]:


df["sourcing_channel"].value_counts()


# In[52]:


df["sourcing_channel"].value_counts().plot.bar()


# In[53]:


df["residence_area_type"].value_counts()


# In[54]:


df["residence_area_type"].value_counts().plot.bar()


# #### 4) Bivariate Analysis
# Analysis of any concurrent relation between two variables or attributes.

# In[55]:


df.corr()


# In[56]:


df.groupby('target')['perc_premium_paid_by_cash_credit'].mean().plot.bar()


# In[57]:


df.groupby('target')['age_in_days'].mean().plot.bar()


# In[58]:


df.groupby('target')['Income'].mean().plot.bar()


# In[59]:


df.groupby('target')['Count_3-6_months_late'].mean().plot.bar()


# In[60]:


df.groupby('target')['Count_6-12_months_late'].mean().plot.bar()


# In[61]:


df.groupby('target')['Count_more_than_12_months_late'].mean().plot.bar()


# In[62]:


df.groupby('target')['application_underwriting_score'].mean().plot.bar()


# In[63]:


df.groupby('target')['no_of_premiums_paid'].mean().plot.bar()


# In[64]:


pd.crosstab(df['target'],df['sourcing_channel'])


# In[65]:


from scipy.stats import chi2_contingency

chi2_contingency(pd.crosstab(df['target'],df['sourcing_channel']))


# In[66]:


# here we can see that the p-value= 1.390061884429808e-29, which is way less than 0.05
# a p-value less than 0.05 is statistically significant.


# In[67]:


pd.crosstab(df['target'],df['residence_area_type'])


# In[68]:


chi2_contingency(pd.crosstab(df['target'],df['residence_area_type']))


# In[69]:


# here we can see that the p-value is approx. 0.65 which is greater than 0.05
# this signifies that urban population is more likely to default the premium


# #### 5) Missing Value Treatment
# <u> Reasons for missing values</u>
# - non-response 
# - error in data collection 
# - error in reading data 

# In[70]:


df.isnull().sum()


# In[71]:


#  Count_3-6_months_late, Count_6-12_months_late, Count_more_than_12_months_late have 97 missing values
#  application_underwriting_score has 2974 missing values.


# In[72]:


# Since we are treating the Count_3-6_months_late, Count_6-12_months_late, and Count_more_than_12_months_late as categorical values, we will be filling their missing values using mode, 
# while application_writing_underscore's missing values will be filled by it's mean


# In[73]:


df_copy= df.copy()


# In[74]:


df_copy.head()


# In[75]:


df_copy.dropna(inplace= False).corr()


# ##### Dropping the rows with missing values doesn't seem like a good option

# In[201]:


# filling the missing vales with the 0

df['target'].corr(df['application_underwriting_score'].fillna(0,inplace= False))


# In[202]:


median= df['application_underwriting_score'].median()


# In[203]:


# filling the missing vales with the median

df['target'].corr(df['application_underwriting_score'].fillna(median,inplace=False))


# In[296]:


# filling the missing vales with the mean

df['target'].corr(df['application_underwriting_score'].fillna(df['application_underwriting_score'].mean(),inplace=False))


# In[76]:


# here we can see that Imputing the missing values with Mean is 
# giving higher correlation, so we will use that !!


# In[5]:


def null(df):
    df['application_underwriting_score'].fillna(df['application_underwriting_score'].mean(),inplace=True)
    df['Count_3-6_months_late'].fillna(0,inplace=True)
    df['Count_6-12_months_late'].fillna(0,inplace=True)
    df['Count_more_than_12_months_late'].fillna(0,inplace=True)
    return df


# In[137]:


null(df)


# In[79]:


df.isnull().sum()


# #### 6) Outlier Treatmeant
# <u> Reasons for outliers</u>
# - data entry errors
# - measurement errors
# - processing errors
# - change in underlying population 

# Let's recall our list containing all columns containing outliers.

# In[80]:


outliers_list


# In[7]:


# start by calculating quantiles and IQRs for each column having outliers


# In[138]:


q1 = int(df.age_in_days.quantile([0.25]))
q3 = int(df.age_in_days.quantile([0.75]))
IQR = q3 - q1
ul = int(q3+ 1.5 * IQR)
ll = int(q1 - 1.5 * IQR)
print(f"Upper limit is {ul} and lower limit is {ll}.")


# In[139]:


# The loc() function helps us to retrieve data values from a dataset at an ease.

df.loc[df["age_in_days"]>ul, "age_in_days"] = np.mean(df["age_in_days"])
df.loc[df["age_in_days"]<ll, "age_in_days"] = np.mean(df["age_in_days"])


# In[140]:


df["age_in_days"].plot.box()


# In[141]:


# we can conclude from the above figure, 
# that we have successfully removed the outliers from the "age_in_days" feature of the dataframe.


# In[142]:


q1 = int(df.Income.quantile(0.25))
q3 = int(df.Income.quantile(0.75))
IQR = q3 - q1
ul = int(q3 + 1.5 * IQR)
ll = int(q1 - 1.5 * IQR)
print(f"Upper limit is {ul} and lower limit is {ll}.")


# In[143]:


df.loc[df["Income"]>ul, "Income"] = np.mean(df["Income"])
df.loc[df["Income"]<ll, "Income"] = np.mean(df["Income"])


# In[144]:


df["Income"].plot.box()


# In[145]:


q1 = int(df.application_underwriting_score.quantile(0.25))
q3 = int(df.application_underwriting_score.quantile(0.75))
IQR = q3 - q1
ul = int(q3 + 1.5 * IQR)
ll = int(q1 - 1.5 * IQR)
print(f"Upper limit is {ul} and lower limit is {ll}.")


# In[146]:


# The data in the column "application_underwriting_score" seems consistent, it's better to leave it like that only.


# In[147]:


q1 = int(df.no_of_premiums_paid.quantile(0.25))
q3 = int(df.no_of_premiums_paid.quantile(0.75))
IQR = q3 - q1
ul = int(q3 + 1.5 * IQR)
ll = int(q1 - 1.5 * IQR)
print(f"Upper limit is {ul} and lower limit is {ll}.")


# In[148]:


df.loc[df["no_of_premiums_paid"]>ul, "no_of_premiums_paid"] = np.mean(df["no_of_premiums_paid"])
df.loc[df["no_of_premiums_paid"]<ll, "no_of_premiums_paid"] = np.mean(df["no_of_premiums_paid"])


# In[149]:


df["no_of_premiums_paid"].plot.box()


# In[150]:


# we can still see two outlying points
# lets fix that by setting ul=22


# In[151]:


ul= 22
df.loc[df["no_of_premiums_paid"]>ul, "no_of_premiums_paid"] = np.mean(df["no_of_premiums_paid"])


# In[152]:


df["no_of_premiums_paid"].plot.box()


# In[153]:


# voila


# #### 7) Variable Transformation
# It is meant to change the scale of values and/or to adjust the skewed data distribution.

# In[33]:


df["age_in_days"].plot.hist()


# In[34]:


np.log(df["age_in_days"]).plot.hist()


# Log transformation gives us a more symmteric curve.

# In[35]:


df["age_in_days"].plot.hist()


# In[36]:


bins = [0,6500, 21900,33960]
group = ["Teenager", "Adult", "Old"]
df["Age Group"] = pd.cut(df["age_in_days"], bins, labels=group)


# In[37]:


df["Age Group"].value_counts()


# In[38]:


df["Income"].plot.hist()


# In[118]:


np.log(df["Income"]).plot.hist()


# In[39]:


# our regular Income feature is just fine, i dont think we need to transform it
# let's just group it in sub-categories


# In[40]:


bins = [0, 100000, 468140]
group = ["Poor", "Rich"]
df["Fianancial Status"] = pd.cut(df["Income"], bins, labels=group)


# In[41]:


df["Fianancial Status"].value_counts()


# In[42]:


df.head()


# In[95]:


def transformation(df):
    df['age_in_days']= np.log(df['age_in_days'])
    
    bins = [0,6500, 21900,33960]
    group = ["Teenager", "Adult", "Old"]
    df["Age Group"] = pd.cut(df["age_in_days"], bins, labels=group)
    
    bins_2 = [0, 100000, 468140]
    group_2 = ["Poor", "Rich"]
    df["Fianancial Status"] = pd.cut(df["Income"], bins_2, labels=group_2)
    
    return df
    


# In[154]:


transformation(df)


# In[98]:


df.describe()


# # <u>Step 5</u>
# ## Predictive Modeling

# ### 1) Logistic Regression Model

# In[155]:


df=pd.get_dummies(df)
train,test = train_test_split(df,test_size=0.2,random_state=112)

# random_state is used for initializing the internal random number generator,
# which will decide the splitting of data into train and test indices in our case.


# In[156]:


x_train=train.drop('target',axis=1)
y_train=train['target']
x_test=test.drop('target',axis=1)
y_test=test['target']
x_train1=scaler.fit_transform(x_train)
x_test1=scaler.fit_transform(x_test)


# In[157]:


logreg.fit(x_train1, y_train)


# In[158]:


logreg.score(x_train1, y_train)


# In[159]:


logreg.score(x_test1,y_test)


# In[106]:


# we can see that our Logistic Regression is working just fine with an accuracy of approx. 94%
# but we will still check for other models if we can get more accurate results


# ### 2) Decision Tree

# In[51]:


df=pd.get_dummies(df)


# In[52]:


x=df.drop('target',axis=1)
y=df['target']


# In[53]:


train_x,test_x,train_y,test_y=train_test_split(x,y,random_state=101,stratify=y)

## so what stratify does is, it makes the proportions of each target value almost similar in both train and test dataset


# In[54]:


train_x1=scaler.fit_transform(train_x)
test_x1=scaler.fit_transform(test_x)


# In[55]:


train_y.value_counts()/len(train_y)


# In[56]:


test_y.value_counts()/len(test_y)


# In[57]:


# notice how the proportion of both 1 and 0 is almost similar in both test and train datasets


# In[58]:


dtc.fit(train_x1,train_y)


# In[59]:


dtc.score(train_x1,train_y)


# In[60]:


dtc.score(test_x1,test_y)


# In[61]:


dtc.predict(test_x1)


# In[63]:


# we are getting an accuracy of 89.45% which is not as good as LogReg


# # <u>Step 6</u>
# ## Model Deployment / Implementation

# In[160]:


test= pd.read_csv("C:\\Users\\devesh\\Desktop\\Git Repositories\\Predicting-if-a-customer-will-default-their-next-premium\\test.csv")


# In[161]:


null(test)


# In[162]:


transformation(test)


# In[164]:


test= pd.get_dummies(test)


# In[165]:


pred= logreg.predict(test)


# In[166]:


print(pred)


# In[167]:


one=0
zero=0
for x in pred:
    if x==1:
        one=one+1
    elif x==0:
        zero= zero+1
        


# In[168]:


one


# In[169]:


zero


# In[171]:


# we can see that about 99% people will pay their premium and about 1% of customers are likely to default the premium 


# ## Last Step : Pickling
# Pickle is used in serializing and deserializing a Python object structure. It's the process of converting a Python object into a byte stream to store it in a file/database, maintain program state across sessions, or transport data over the network.

# In[172]:


import pickle
filename = 'model.pkl'
pickle.dump(logreg, open(filename, 'wb'))


# In[ ]:




