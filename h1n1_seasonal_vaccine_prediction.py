H1N1 and Seasonal Flu Prediction

Author Ranjith Ramaswamy
"""

#Loading the important libraries for Data manipulation and Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""Loading the training and testing data"""

df_f = pd.read_csv("training_set_features.csv")
df_l = pd.read_csv("training_set_labels.csv")
df = pd.merge(df_f, df_l, on='respondent_id', how='inner')
df_test = pd.read_csv("test_set_features.csv")

df.describe()

#missing values of each column were sorted in Descending order
df.isna().sum().sort_values(ascending = False)

"""H1N1 Data manipulation and analysis"""

#the dataset is copied to make changes and analyse
df_h_Ranj = df.copy()

#The H1N1 vaccination with reagrd to age group is visualised using bar chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
h1n1_vaccinated = df_h_Ranj[df_h_Ranj["h1n1_vaccine"] ==1]
value_age = h1n1_vaccinated["age_group"].value_counts()
ax1.bar(value_age.index, value_age.values)
ax1.set_ylabel('Count')
ax1.set_title("H1N1 vaccination")
ax1.tick_params(axis='x', rotation=45)

#The Seasonal vaccination with reagrd to age group is visualised using bar chart
seas_vaccinated = df_h_Ranj[df_h_Ranj["seasonal_vaccine"] ==1]
value_age = seas_vaccinated["age_group"].value_counts()
ax2.bar(value_age.index, value_age.values)
ax2.set_ylabel('Count')
ax2.set_title("Seasonal vaccination")
ax2.tick_params(axis='x', rotation=45)
plt.show()

"""The above figure shows that there is significant difference between H1N1 and seasonal flu vaccination. Therefore the data should be preprocessed separately for H1N1 and seasonal Flu

The missing values may contain some useful information therefore the missing values are replaced with -1 for column with datatype integer and for column with datatype string "unknown" value is imputed.
"""

df_h_Ranj["education"].fillna("unknown", inplace=True)
df_h_Ranj["income_poverty"].fillna("unknown", inplace=True)
df_h_Ranj["employment_status"].fillna("unknown", inplace=True)
df_h_Ranj["household_adults"].fillna(-1, inplace=True)
df_h_Ranj["marital_status"].fillna("unknown", inplace=True)
df_h_Ranj["h1n1_concern"].fillna(-1, inplace=True)
df_h_Ranj["h1n1_knowledge"].fillna(-1, inplace=True)
df_h_Ranj["opinion_h1n1_vacc_effective"].fillna(-1, inplace=True)
df_h_Ranj["opinion_h1n1_risk"].fillna(-1, inplace=True)
df_h_Ranj["opinion_h1n1_sick_from_vacc"].fillna(-1, inplace=True)
df_h_Ranj["opinion_seas_vacc_effective"].fillna(-1, inplace=True)
df_h_Ranj["opinion_seas_risk"].fillna(-1, inplace=True)
df_h_Ranj["behavioral_antiviral_meds"].fillna(-1, inplace=True)
df_h_Ranj["behavioral_avoidance"].fillna(-1, inplace=True)
df_h_Ranj["behavioral_face_mask"].fillna(-1, inplace=True)
df_h_Ranj["behavioral_wash_hands"].fillna(-1, inplace=True)
df_h_Ranj["behavioral_large_gatherings"].fillna(-1, inplace=True)
df_h_Ranj["behavioral_outside_home"].fillna(-1, inplace=True)
df_h_Ranj["behavioral_touch_face"].fillna(-1, inplace=True)
df_h_Ranj["doctor_recc_h1n1"].fillna(-1, inplace=True)
df_h_Ranj["doctor_recc_seasonal"].fillna(-1, inplace=True)
df_h_Ranj["chronic_med_condition"].fillna(-1, inplace=True)
df_h_Ranj["child_under_6_months"].fillna(-1, inplace=True)
df_h_Ranj["health_worker"].fillna(-1, inplace=True)
df_h_Ranj["health_insurance"].fillna(-1, inplace=True)
df_h_Ranj["household_children"].fillna(-1, inplace=True)
df_h_Ranj["rent_or_own"].fillna("unknown", inplace=True)
df_h_Ranj["opinion_seas_sick_from_vacc"].fillna(5, inplace=True)

#no of unique values and its count for geological region
df_h_Ranj["hhs_geo_region"].value_counts()

#no of unique values and its count for Employment Industry
df_h_Ranj["employment_industry"].value_counts()

#no of unique values and its count for Employment Occupation
df_h_Ranj["employment_occupation"].value_counts()

"""The number of unique categorical values is more in columns such as  hhs_geo_region, employment_industry, employment_occupation if we encode these columns with oneHotEncoding we will have more number of columns which will be difficult for machine learning algorithm to train and achieve global minimum. Therefore inorder to reduce the dimension we replace the column values with the percentage of h1n1 vaccination with respect to each value in the column"""

#the attribute called geo region percentage is created in which the values are the percentage of vaccination
#taken in that region
aggreg = df_h_Ranj.groupby(['hhs_geo_region', 'h1n1_vaccine']).size().reset_index(name='count')
aggreg['percentage'] = aggreg.groupby('hhs_geo_region')['count'].apply(lambda x: 100 * x / x.sum())
seasonal_vaccine_percentage = {}
for d in aggreg.loc[aggreg["h1n1_vaccine"] == 1].iterrows():
    seasonal_vaccine_percentage[d[1][0]] = round(d[1][3] /100, 4)
df_h_Ranj['hhs_geo_region_percentage'] = df_h_Ranj["hhs_geo_region"].map(seasonal_vaccine_percentage)

#the attribute called employment industry percentage is created in which the values are the percentage of people
#in that industry taken vaccination
df_h_Ranj["employment_industry"].fillna("unknown", inplace=True)
aggreg = df_h_Ranj.groupby(['employment_industry', 'h1n1_vaccine']).size().reset_index(name='count')
aggreg['percentage'] = aggreg.groupby('employment_industry')['count'].apply(lambda x: 100 * x / x.sum())
employment_industry_percentage = {}
for d in aggreg.loc[aggreg["h1n1_vaccine"] == 1].iterrows():
    employment_industry_percentage[d[1][0]] = round(d[1][3] /100, 4)
df_h_Ranj['employment_industry_percentage'] = df_h_Ranj["employment_industry"].map(employment_industry_percentage)

#the attribute called employment occupation percentage is created in which the values are the percentage of people
#in that occupation taken vaccination

df_h_Ranj["employment_occupation"].fillna("unknown", inplace=True)
aggreg = df_h_Ranj.groupby(['employment_occupation', 'h1n1_vaccine']).size().reset_index(name='count')
aggreg['percentage'] = aggreg.groupby('employment_occupation')['count'].apply(lambda x: 100 * x / x.sum())
employment_occupation_percentage = {}
for d in aggreg.loc[aggreg["h1n1_vaccine"] == 1].iterrows():
    employment_occupation_percentage[d[1][0]] = round(d[1][3] /100,4)
df_h_Ranj['employment_occupation_percentage'] = df_h_Ranj["employment_occupation"].map(employment_occupation_percentage)

#The dataset is copied for analysis
df_h_Lok = df.copy()

#by replacing missing value we observe the missing value proportion is equal to the not married value
df_h_Lok["marital_status"].fillna("missing",inplace=True)
grouped = df_h_Lok.groupby(['marital_status', 'h1n1_vaccine']).size().reset_index(name='count')
grouped['percentage'] = grouped.groupby('marital_status')['count'].apply(lambda x: 100 * x / x.sum())
sns.histplot(data=grouped, x='marital_status', hue='h1n1_vaccine', weights='percentage', multiple='stack')
plt.xticks(rotation = 45)

"""From the above chart we can observe missing values in the marital_status column have the same proportion of "Not Married" value therefore the missing values were replaced with "Not Married" value.
The same procedure is followed for other columns. If the missing value proportion doesn't match with any other values in that column we can replace the value with -1 or "unknown"
"""

#The columns and missing values for the column were mentioned, These arrays will help us to
#visualise the value in further  analysis
columns = ["education", "income_poverty", "employment_status",
            "household_adults", "employment_industry",
           "employment_occupation", "marital_status", "h1n1_concern",
           "h1n1_knowledge", "opinion_h1n1_vacc_effective", "opinion_h1n1_risk",
           "opinion_h1n1_sick_from_vacc", "opinion_seas_vacc_effective",
           "behavioral_antiviral_meds", "behavioral_avoidance", "behavioral_face_mask",
           "behavioral_wash_hands", "behavioral_large_gatherings","behavioral_outside_home",
           "behavioral_touch_face", "doctor_recc_h1n1","doctor_recc_seasonal",
           "chronic_med_condition", "child_under_6_months", "health_worker"
          ]
missing_values = ["unknown", "unknown", "unknown", -1,
                  "unknown", "unknown", "unknown", -1, -1, -1,-1,
                  -1, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
index = 0;

#proportion of values with missing values
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
for i in range(2):
    for j in range(2):
        df_h_Lok[columns[index]].fillna(missing_values[index],inplace=True)
        grouped = df_h_Lok.groupby([columns[index], 'h1n1_vaccine']).size().reset_index(name='count')
        grouped[columns[index]] = grouped[columns[index]].astype(str)
        grouped['percentage'] = grouped.groupby(columns[index])['count'].apply(lambda x: 100 * x / x.sum())
        labels = grouped[columns[index]].unique()
        sns.histplot(data=grouped, x=columns[index], hue='h1n1_vaccine', weights='percentage', multiple='stack',ax=axes[i,j])
        axes[i,j].set_ylabel('Percentage')
        axes[i,j].tick_params(axis='x', rotation=45)
        index = index + 1

plt.tight_layout()
plt.show()

#proportion of values with missing values
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
for i in range(2):
    for j in range(2):
        df_h_Lok[columns[index]].fillna(missing_values[index],inplace=True)
        grouped = df_h_Lok.groupby([columns[index], 'h1n1_vaccine']).size().reset_index(name='count')
        grouped[columns[index]] = grouped[columns[index]].astype(str)
        grouped['percentage'] = grouped.groupby(columns[index])['count'].apply(lambda x: 100 * x / x.sum())
        labels = grouped[columns[index]].unique()
        sns.histplot(data=grouped, x=columns[index], hue='h1n1_vaccine', weights='percentage', multiple='stack',ax=axes[i,j])
        axes[i,j].set_ylabel('Percentage')
        axes[i,j].tick_params(axis='x', rotation=45)
        index = index + 1

plt.tight_layout()
plt.show()

#proportion of values with missing values
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
for i in range(2):
    for j in range(2):
        df_h_Lok[columns[index]].fillna(missing_values[index],inplace=True)
        grouped = df_h_Lok.groupby([columns[index], 'h1n1_vaccine']).size().reset_index(name='count')
        grouped[columns[index]] = grouped[columns[index]].astype(str)
        grouped['percentage'] = grouped.groupby(columns[index])['count'].apply(lambda x: 100 * x / x.sum())
        labels = grouped[columns[index]].unique()
        sns.histplot(data=grouped, x=columns[index], hue='h1n1_vaccine', weights='percentage', multiple='stack',ax=axes[i,j])
        axes[i,j].set_ylabel('Percentage')
        axes[i,j].tick_params(axis='x', rotation=45)
        index = index + 1

plt.tight_layout()
plt.show()

#proportion of values with missing values
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
for i in range(2):
    for j in range(2):
        df_h_Lok[columns[index]].fillna(missing_values[index],inplace=True)
        grouped = df_h_Lok.groupby([columns[index], 'h1n1_vaccine']).size().reset_index(name='count')
        grouped[columns[index]] = grouped[columns[index]].astype(str)
        grouped['percentage'] = grouped.groupby(columns[index])['count'].apply(lambda x: 100 * x / x.sum())
        labels = grouped[columns[index]].unique()
        sns.histplot(data=grouped, x=columns[index], hue='h1n1_vaccine', weights='percentage', multiple='stack',ax=axes[i,j])
        axes[i,j].set_ylabel('Percentage')
        axes[i,j].tick_params(axis='x', rotation=45)
        index = index + 1

plt.tight_layout()
plt.show()

#proportion of values with missing values
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
for i in range(2):
    for j in range(2):
        df_h_Lok[columns[index]].fillna(missing_values[index],inplace=True)
        grouped = df_h_Lok.groupby([columns[index], 'h1n1_vaccine']).size().reset_index(name='count')
        grouped[columns[index]] = grouped[columns[index]].astype(str)
        grouped['percentage'] = grouped.groupby(columns[index])['count'].apply(lambda x: 100 * x / x.sum())
        labels = grouped[columns[index]].unique()
        sns.histplot(data=grouped, x=columns[index], hue='h1n1_vaccine', weights='percentage', multiple='stack',ax=axes[i,j])
        axes[i,j].set_ylabel('Percentage')
        axes[i,j].tick_params(axis='x', rotation=45)
        index = index + 1

plt.tight_layout()
plt.show()

#proportion of values with missing values
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
for i in range(2):
    for j in range(2):
        df_h_Lok[columns[index]].fillna(missing_values[index],inplace=True)
        grouped = df_h_Lok.groupby([columns[index], 'h1n1_vaccine']).size().reset_index(name='count')
        grouped[columns[index]] = grouped[columns[index]].astype(str)
        grouped['percentage'] = grouped.groupby(columns[index])['count'].apply(lambda x: 100 * x / x.sum())
        labels = grouped[columns[index]].unique()
        sns.histplot(data=grouped, x=columns[index], hue='h1n1_vaccine', weights='percentage', multiple='stack',ax=axes[i,j])
        axes[i,j].set_ylabel('Percentage')
        axes[i,j].tick_params(axis='x', rotation=45)
        index = index + 1

plt.tight_layout()
plt.show()

#proportion of values with missing values
df_h_Lok[columns[index]].fillna(missing_values[index],inplace=True)
grouped = df_h_Lok.groupby([columns[index], 'h1n1_vaccine']).size().reset_index(name='count')
grouped[columns[index]] = grouped[columns[index]].astype(str)
grouped['percentage'] = grouped.groupby(columns[index])['count'].apply(lambda x: 100 * x / x.sum())
labels = grouped[columns[index]].unique()
sns.histplot(data=grouped, x=columns[index], hue='h1n1_vaccine', weights='percentage', multiple='stack')
plt.tight_layout()
plt.show()

"""The above charts were compared and missing values were replaced with the values in the column that have same proportion. proportional imputation"""

#missing values are replaced with the values which have same proportion
df_h_Lok["education"].fillna("< 12 Years", inplace=True)
df_h_Lok["income_poverty"].fillna("<= $75,000, Above Poverty", inplace=True)
df_h_Lok["employment_status"].fillna("Employed", inplace=True)
df_h_Lok["household_adults"].fillna(-1, inplace=True)
df_h_Lok["marital_status"].fillna("unknown", inplace=True)
df_h_Lok["h1n1_concern"].fillna(1, inplace=True)
df_h_Lok["h1n1_knowledge"].fillna(1, inplace=True)
df_h_Lok["opinion_h1n1_vacc_effective"].fillna(4, inplace=True)
df_h_Lok["opinion_h1n1_risk"].fillna(3, inplace=True)
df_h_Lok["opinion_h1n1_sick_from_vacc"].fillna(2, inplace=True)
df_h_Lok["opinion_seas_vacc_effective"].fillna(4, inplace=True)
df_h_Lok["opinion_seas_risk"].fillna(3, inplace=True)
df_h_Lok["behavioral_antiviral_meds"].fillna(2, inplace=True)
df_h_Lok["behavioral_avoidance"].fillna(0, inplace=True)
df_h_Lok["behavioral_face_mask"].fillna(1, inplace=True)
df_h_Lok["behavioral_wash_hands"].fillna(0, inplace=True)
df_h_Lok["behavioral_large_gatherings"].fillna(-1, inplace=True)
df_h_Lok["behavioral_outside_home"].fillna(-1, inplace=True)
df_h_Lok["behavioral_touch_face"].fillna(-1, inplace=True)
df_h_Lok["doctor_recc_h1n1"].fillna(-1, inplace=True)
df_h_Lok["doctor_recc_seasonal"].fillna(-1, inplace=True)
df_h_Lok["chronic_med_condition"].fillna(0, inplace=True)
df_h_Lok["child_under_6_months"].fillna(0, inplace=True)
df_h_Lok["health_worker"].fillna(0, inplace=True)
df_h_Lok["health_insurance"].fillna(-1, inplace=True)
df_h_Lok["household_children"].fillna(-1, inplace=True)
df_h_Lok["rent_or_own"].fillna("Rent", inplace=True)
df_h_Lok["opinion_seas_sick_from_vacc"].fillna(5, inplace=True)

#H1N1 vaccine with regards to race visualised
sns.histplot(data=df_h_Lok, x='race', hue='h1n1_vaccine', multiple='stack')
plt.show()

"""Some of the column values have some inherent order which can be useful for machine learning algorithm to find the relationship based on the values which have order.
for example, race values can be ordered based on vaccination percentage
white -> Other or Multiple -> Hispanic -> Black
important note : Here the values of the race are ordered based only on the vaccination percentage of the data.
"""

#The order for ordinal encoding, order were inspired by natural order or by proportion of vaccination
age_order = ["18 - 34 Years", "35 - 44 Years", "45 - 54 Years", "55 - 64 Years", "65+ Years"]
education_order = ["< 12 Years", "12 Years", "Some College", "College Graduate"]
povery_order = ["Below Poverty", "<= $75,000, Above Poverty", "> $75,000"]
employment_order = ["Not in Labor Force","Employed","Unemployed"]
race_order = ["White", "Other or Multiple", "Hispanic", "Black"]
marital_order = ["Married", "Not Married", "unknown"]

"""Comparision :
    The values in the column varies for H1n1 and seasonal flu vaccination the data were handled seperately and to reduce dimensionality the columns with more catgorical values were replaced with percentage of vaccination with respect to each value.
    Handling missing values with proportional imputation gave better results than replacing with new unique values such as -1 or "unknown". However, if we could'nt find same proportion equal to missing values we can replace with new unique values. The columns which have inherent order are ordinally encoded instead of One-hot encoding.
    
The best ideas were considered and the data is preprocessed for further analysis
"""

df_h = df.copy()

df_h["education"].fillna("< 12 Years", inplace=True)
df_h["income_poverty"].fillna("<= $75,000, Above Poverty", inplace=True)
df_h["employment_status"].fillna("Employed", inplace=True)
df_h["household_adults"].fillna(-1, inplace=True)
df_h["marital_status"].fillna("unknown", inplace=True)
df_h["h1n1_concern"].fillna(1, inplace=True)
df_h["h1n1_knowledge"].fillna(1, inplace=True)
df_h["opinion_h1n1_vacc_effective"].fillna(4, inplace=True)
df_h["opinion_h1n1_risk"].fillna(3, inplace=True)
df_h["opinion_h1n1_sick_from_vacc"].fillna(2, inplace=True)
df_h["opinion_seas_vacc_effective"].fillna(4, inplace=True)
df_h["opinion_seas_risk"].fillna(3, inplace=True)
df_h["behavioral_antiviral_meds"].fillna(2, inplace=True)
df_h["behavioral_avoidance"].fillna(0, inplace=True)
df_h["behavioral_face_mask"].fillna(1, inplace=True)
df_h["behavioral_wash_hands"].fillna(0, inplace=True)
df_h["behavioral_large_gatherings"].fillna(-1, inplace=True)
df_h["behavioral_outside_home"].fillna(-1, inplace=True)
df_h["behavioral_touch_face"].fillna(-1, inplace=True)
df_h["doctor_recc_h1n1"].fillna(-1, inplace=True)
df_h["doctor_recc_seasonal"].fillna(-1, inplace=True)
df_h["chronic_med_condition"].fillna(0, inplace=True)
df_h["child_under_6_months"].fillna(0, inplace=True)
df_h["health_worker"].fillna(0, inplace=True)
df_h["health_insurance"].fillna(-1, inplace=True)
df_h["household_children"].fillna(-1, inplace=True)
df_h["rent_or_own"].fillna("Rent", inplace=True)
df_h["opinion_seas_sick_from_vacc"].fillna(5, inplace=True)

aggreg = df_h.groupby(['hhs_geo_region', 'h1n1_vaccine']).size().reset_index(name='count')
aggreg['percentage'] = aggreg.groupby('hhs_geo_region')['count'].apply(lambda x: 100 * x / x.sum())
seasonal_vaccine_percentage = {}
for d in aggreg.loc[aggreg["h1n1_vaccine"] == 1].iterrows():
    seasonal_vaccine_percentage[d[1][0]] = round(d[1][3] /100, 4)
df_h['hhs_geo_region_percentage'] = df_h["hhs_geo_region"].map(seasonal_vaccine_percentage)

df_h["employment_industry"].fillna("unknown", inplace=True)
aggreg = df_h.groupby(['employment_industry', 'h1n1_vaccine']).size().reset_index(name='count')
aggreg['percentage'] = aggreg.groupby('employment_industry')['count'].apply(lambda x: 100 * x / x.sum())
employment_industry_percentage = {}
for d in aggreg.loc[aggreg["h1n1_vaccine"] == 1].iterrows():
    employment_industry_percentage[d[1][0]] = round(d[1][3] /100, 4)
df_h['employment_industry_percentage'] = df_h["employment_industry"].map(employment_industry_percentage)

df_h["employment_occupation"].fillna("unknown", inplace=True)
aggreg = df_h.groupby(['employment_occupation', 'h1n1_vaccine']).size().reset_index(name='count')
aggreg['percentage'] = aggreg.groupby('employment_occupation')['count'].apply(lambda x: 100 * x / x.sum())
employment_occupation_percentage = {}
for d in aggreg.loc[aggreg["h1n1_vaccine"] == 1].iterrows():
    employment_occupation_percentage[d[1][0]] = round(d[1][3] /100,4)
df_h['employment_occupation_percentage'] = df_h["employment_occupation"].map(employment_occupation_percentage)

df_h = df_h.drop(["hhs_geo_region", "employment_industry", "employment_occupation"], axis = 1)

#Dummy estimator which is used in pipeline
from sklearn.base import BaseEstimator, TransformerMixin
class PassthroughTransformer(BaseEstimator):
  def fit(self, X, y = None):
    self.cols = X.columns
    return self

  def transform(self, X, y = None):
    self.cols = X.columns
    return X.values

  def get_feature_names(self):
    return self.cols

#attributes based on type of data were seperated
binary_attributes_n = ["behavioral_avoidance", "behavioral_face_mask", "behavioral_wash_hands",
                       "behavioral_large_gatherings", "behavioral_outside_home", "behavioral_touch_face",
                       "doctor_recc_h1n1", "doctor_recc_seasonal", "chronic_med_condition",
                       "child_under_6_months", "health_worker", "health_insurance"]

binary_attribute_s = ["sex", "rent_or_own"]

ordinal_attribute_n = ["opinion_h1n1_vacc_effective", "opinion_h1n1_risk", "opinion_h1n1_sick_from_vacc",
                       "opinion_seas_vacc_effective", "opinion_seas_risk", "opinion_seas_sick_from_vacc",
                       "household_adults", "household_children","behavioral_antiviral_meds",
                       "hhs_geo_region_percentage", "employment_industry_percentage",
                       "employment_occupation_percentage","h1n1_concern","h1n1_knowledge"]

ordinal_attribute_s = ["age_group", "education", "income_poverty", "employment_status","race","marital_status"]

nominal_attribute = ["census_msa"]

#using pipeline and column transfer the data were processed parallely
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def do_preprocess(data):

    binary_s_pipeline =  make_pipeline(OrdinalEncoder())

    binary_n_pipeline = make_pipeline(PassthroughTransformer())
    nominal_s = make_pipeline(OneHotEncoder(sparse=False))

    age_order = ["18 - 34 Years", "35 - 44 Years", "45 - 54 Years", "55 - 64 Years", "65+ Years"]
    education_order = ["< 12 Years", "12 Years", "Some College", "College Graduate"]
    povery_order = ["Below Poverty", "<= $75,000, Above Poverty", "> $75,000"]
    employment_order = ["Not in Labor Force","Employed","Unemployed"]
    race_order = ["White", "Other or Multiple", "Hispanic", "Black"]
    marital_order = ["Married", "Not Married", "unknown"]

    ordinal_attribute_s_pipeline = make_pipeline(
        OrdinalEncoder(categories=[age_order, education_order, povery_order,employment_order, race_order, marital_order])
    )
    pass_col = ["h1n1_vaccine", "seasonal_vaccine"]
    passthrough_pipeline = make_pipeline(PassthroughTransformer())
    ordinal_attribute_n_pipeline = make_pipeline(PassthroughTransformer())
    preprocessing = ColumnTransformer(transformers=[
                    ("bin_s", binary_s_pipeline, binary_attribute_s),
                    ("bin_n", binary_n_pipeline, binary_attributes_n),
                    ("nominal", nominal_s, nominal_attribute),
                    ("ordinal_s", ordinal_attribute_s_pipeline, ordinal_attribute_s),
                    ("ordinal_n", ordinal_attribute_n_pipeline, ordinal_attribute_n),
                    ("pass", passthrough_pipeline, pass_col)
                    ], remainder="drop")
    return preprocessing.fit_transform(data)

#processing the data
X = do_preprocess(df_h)

#column heading
columns_label = ['bin_s__sex',
 'bin_s__rent_or_own',
 'behavioral_avoidance',
 'behavioral_face_mask',
 'behavioral_wash_hands',
 'behavioral_large_gatherings',
 'behavioral_outside_home',
 'behavioral_touch_face',
 'doctor_recc_h1n1',
 'doctor_recc_seasonal',
 'chronic_med_condition',
 'child_under_6_months',
 'health_worker',
 'health_insurance',
 'MSA, Not Principle  City',
 'MSA, Principle City',
 'Non-MSA',
 'age_group',
 'education',
 'income_poverty',
 'employment_status',
 'race',
 'marital_status',
 'opinion_h1n1_vacc_effective',
 'opinion_h1n1_risk',
 'opinion_h1n1_sick_from_vacc',
 'opinion_seas_vacc_effective',
 'opinion_seas_risk',
 'opinion_seas_sick_from_vacc',
 'household_adults',
 'household_children',
 'behavioral_antiviral_meds',
 'hhs_geo_region_percentage',
 'employment_industry_percentage',
 'employment_occupation_percentage',
 'h1n1_concern',
 'h1n1_knowledge',
 'h1n1_vaccine',
 'seasonal_vaccine']

#processed data is converted into dataframe
df_h_p = pd.DataFrame(X, columns=columns_label)

df_h_p

correlation_matrix = df_h_p.corr()

#columns which have highest correlation were ordered
correlation_matrix["h1n1_vaccine"].abs().sort_values(ascending=False).index

#based on correlation opinion and doctor recommendation columns were taken into consideration
important_correlation = df_h_p[["h1n1_vaccine","opinion_h1n1_vacc_effective", "opinion_h1n1_risk",
                              "opinion_h1n1_sick_from_vacc", "opinion_seas_vacc_effective",
                              "opinion_seas_risk", "opinion_seas_sick_from_vacc",
                               "doctor_recc_h1n1", "doctor_recc_seasonal", "chronic_med_condition"]].corr()

#heat map is used to visualize the correlation
sns.heatmap(important_correlation,
            annot=True, cmap="coolwarm")

#target and remaining columns were divided
train_h = df_h_p.iloc[:,:-2]
target_h = df_h_p.iloc[:,-2]

#Test and the training data were splitted
from sklearn.model_selection import train_test_split
X_tr, X_tt, Y_tr, Y_tt =  train_test_split(train_h, target_h, test_size = 0.2,shuffle=True)

#using randomizedCV with K-fold cross validation 3 optimal parameter for XGBosst classifier is searched
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

xgb = XGBClassifier(learning_rate = 0.01, objective = 'binary:logistic', eval_metric="auc")
param_xgb = {
    "max_depth": [5,8,10,15],
    "subsample" : [1, 0.9,0.8, 0.7],
    "n_estimators" : [100, 120, 200, 250],
    "gamma" : [0, 0.25, 0.5],
    "alpha" : [0, 0.7, 0.9, 1]

}
xgb_m = RandomizedSearchCV(estimator=xgb, param_distributions=param_xgb, cv=3,n_jobs=-1, n_iter=30)
xgb_m.fit(X_tr, Y_tr)

#Best parameters according to randomized search
xgb_m.best_params_

#XGBoost classifier using parameters found by randomized search
xgb_best = XGBClassifier(learning_rate = 0.01, objective = 'binary:logistic', eval_metric="auc",
                         gamma = 0.5, alpha=0,max_depth = 8, subsample = 0.8)

#using grid search for n_estimator best value is found
from sklearn.model_selection import GridSearchCV
param = {
    "n_estimators" : [250,280,300,320,340]
}
grid_xgb = GridSearchCV(estimator=xgb_best, param_grid=param, cv=5, n_jobs=-1)
grid_xgb.fit(X_tr, Y_tr)

#best param for n_estimator
grid_xgb.best_params_

#XGBoost classifier is updated by n_estimator param found in previous grid search
xgb_best = XGBClassifier(n_estimators = 320, objective = 'binary:logistic', eval_metric="auc",
                         gamma = 0.5, alpha=0, max_depth = 8, subsample = 0.8)

#Learning rate is searched using Grid Search
param = {
    "learning_rate" : [0.001, 0.01, 0.1]
}
grid_xgb = GridSearchCV(estimator=xgb_best, param_grid=param, cv=5, n_jobs=-1)
grid_xgb.fit(X_tr, Y_tr)
grid_xgb.best_params_

#XGBoost classifier updated with learning rate
xgb_best = XGBClassifier(n_estimators = 320, objective = 'binary:logistic', eval_metric="auc",
                          alpha=0, max_depth = 8, subsample = 0.8, learning_rate = 0.01)

#reguarization parameter gamma value is searched using grid search
param = {
    "gamma" : [0.5, 0.6, 0.7, 0.8]
}
grid_xgb = GridSearchCV(estimator=xgb_best, param_grid=param, cv=5, n_jobs=-1)
grid_xgb.fit(X_tr, Y_tr)
grid_xgb.best_params_

#regularization param gamma is updated
xgb_best = XGBClassifier(n_estimators = 320, objective = 'binary:logistic', eval_metric="auc",
                          gamma = 0.8, max_depth = 8, subsample = 0.8, learning_rate = 0.01)

#L1 regularization term alpha value is searched using grid search
param = {
    "alpha" : [0.6, 0.7, 0.8, 0.9, 1]
}
grid_xgb = GridSearchCV(estimator=xgb_best, param_grid=param, cv=5, n_jobs=-1)
grid_xgb.fit(X_tr, Y_tr)
grid_xgb.best_params_

#alpha value is updated in XGBoost classifier
xgb_best = XGBClassifier(n_estimators = 320, objective = 'binary:logistic', eval_metric="auc",
                          gamma = 0.5, alpha = 0.6, subsample = 0.7, learning_rate = 0.01)

#Max tree depth param is searched using grid search
param = {
    "max_depth" : [8, 10, 12, 14, 16]
}
grid_xgb = GridSearchCV(estimator=xgb_best, param_grid=param, cv=5, n_jobs=-1)
grid_xgb.fit(X_tr, Y_tr)
grid_xgb.best_params_

#max tree depth param is updated
xgb_best = XGBClassifier(n_estimators = 320, objective = 'binary:logistic', eval_metric="auc",
                          gamma = 0.5, alpha = 1, subsample = 0.7, learning_rate = 0.01, max_depth = 8)

#ROC AUC score is found using K fold cross validation where k = 5
from sklearn.model_selection import cross_val_score
scores = cross_val_score(xgb_best, X_tr, Y_tr, cv=5, n_jobs=-1, scoring='roc_auc')
scores

#Randomized search is used to find optimal parameters for Random Forest Classifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
dcls = RandomForestClassifier(n_jobs=-1, class_weight = "balanced")
param_dist = {
    'n_estimators': np.arange(360, 370, 3),
    'max_depth': [22,23,24,25],
    'max_features': ['sqrt', 'log2','auto'],
    "min_samples_leaf" : [2, 3, 4],

    }

random_search = RandomizedSearchCV(estimator=dcls, param_distributions=param_dist, cv=5, n_jobs=-1, n_iter=30)
random_search.fit(X_tr, Y_tr)
random_search.best_params_

#using best params found in Random search random forest classifier is created
rf_best = RandomForestClassifier(n_estimators = 360, min_samples_leaf = 2, max_features = "log2",
                                 class_weight = "balanced", max_depth = 22, n_jobs=-1)

#Maximum samples used to train a tree is found using grid search
param = {
    "max_samples" : [0.7, 0.8, 0.9, 1]
}
grid_rfc = GridSearchCV(estimator=rf_best, param_grid=param, cv=5, n_jobs=-1)
grid_rfc.fit(X_tr, Y_tr)
grid_rfc.best_params_

#max sample size param is updated in Random forest classifier
rf_best = RandomForestClassifier(n_estimators = 360, min_samples_leaf = 2, max_features = "auto",
                                 class_weight = "balanced", max_samples = 0.7)

#maximum depth of tree param is found using grid search
param = {
    "max_depth" : [12, 20, 24, 26, 28]
}
grid_rfc = GridSearchCV(estimator=rf_best, param_grid=param, cv=5, n_jobs=-1)
grid_rfc.fit(X_tr, Y_tr)
grid_rfc.best_params_

#maximum tree depth param is updated
rf_best = RandomForestClassifier(n_estimators = 360, min_samples_leaf = 2,
                                 class_weight = "balanced", max_samples = 0.7, max_depth = 28)

#size of random subset param is found using grid search
param = {
    "max_features" : ["sqrt", "log2", "auto"]
}
grid_rfc = GridSearchCV(estimator=rf_best, param_grid=param, cv=5, n_jobs=-1)
grid_rfc.fit(X_tr, Y_tr)
grid_rfc.best_params_

#max feature param is updated in random forest classifier
rf_best = RandomForestClassifier(n_estimators = 360, min_samples_leaf = 2,max_features = "sqrt",
                                 class_weight = "balanced", max_samples = 0.7, max_depth = 28)

#random forest classifier ROC AUC score is obtained using cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(rf_best, X_tr, Y_tr, cv=5, n_jobs=-1, scoring='roc_auc')
scores

#The best models of Random forest and XGBosst classifier is fitted in training sets and values predicted probability for test sets

xgb_best = XGBClassifier(n_estimators = 320, objective = 'binary:logistic', eval_metric="auc",
                         gamma = 0.5, alpha = 1, subsample = 0.7, learning_rate = 0.01, max_depth = 8)
xgb_best.fit(X_tr, Y_tr)
Y_xgp = xgb_best.predict_proba(X_tt)[:, 1]


rf_best = RandomForestClassifier(n_estimators = 360, min_samples_leaf = 2,max_features = "sqrt",
                                 class_weight = "balanced", max_samples = 0.7, max_depth = 28)
rf_best.fit(X_tr, Y_tr)
Y_rfp = rf_best.predict_proba(X_tt)[:, 1]

from sklearn.metrics import roc_curve
#Fals poitive and true positive rates are obtained compared to predicted test sets

fpr_xg, tpr_xg, thresholds = roc_curve(Y_tt, Y_xgp)

fpr_rf, tpr_rf, thresholds = roc_curve(Y_tt, Y_rfp)

plt.plot(fpr_xg, tpr_xg, label="XGBoost classifier")
plt.plot(fpr_rf, tpr_rf, label="Random Forest Classifier")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("H1N1 Vaccine")
plt.legend(loc="lower right")
plt.axis([0, 1, 0, 1])
plt.plot([0, 1], [0, 1], color='r', linestyle='--')
plt.show()

#predicted classes for test sets

Y_xgp = xgb_best.predict(X_tt)

Y_rfp = rf_best.predict(X_tt)

#precision, recall and f1score for test set is obtained
from sklearn.metrics import precision_score, recall_score,f1_score
print(precision_score(Y_tt, Y_xgp))
print(recall_score(Y_tt, Y_xgp))
print(f1_score(Y_tt, Y_xgp))

#precision, recall and f1score for test set is obtained
from sklearn.metrics import precision_score, recall_score,f1_score
print(precision_score(Y_tt, Y_rfp))
print(recall_score(Y_tt, Y_rfp))
print(f1_score(Y_tt, Y_rfp))

#ROC AUC score for test set is obtained
from sklearn.metrics import roc_auc_score
Y_xgprob = xgb_best.predict_proba(X_tt)[: ,1]
roc_auc_score(Y_tt, Y_xgprob)

#ROC AUC score for test set is obtained
from sklearn.metrics import roc_auc_score
Y_rfproba = rf_best.predict_proba(X_tt)[:, 1]
roc_auc_score(Y_tt, Y_rfproba)

#XGBoost predicted probability of training set using cross validation
from sklearn.model_selection import cross_val_predict
xgb_best = XGBClassifier(n_estimators = 320, objective = 'binary:logistic', eval_metric="auc",
                         gamma = 0.5, alpha = 1, subsample = 0.7, learning_rate = 0.01, max_depth = 8)
crs_rf = cross_val_predict(xgb_best, train_h, target_h, cv = 3, method = "predict_proba")

#ROC AUC score of XGBoost on training set using cross validation
from sklearn.metrics import roc_auc_score
roc_auc_score(target_h, crs[:, 1])

#random forest predicted probability of training set using cross validation
rf_best = RandomForestClassifier(n_estimators = 360, min_samples_leaf = 2,max_features = "sqrt",
                                 class_weight = "balanced", max_samples = 0.7, max_depth = 28)

crs_rf = cross_val_predict(rf_best, train_h, target_h, cv = 3, method = "predict_proba")

#ROC AUC score of Random Forest on training set using cross validation
roc_auc_score(target_h, crs_rf[:, 1])

#Random Forest Classifier with depth 3 to find correlation among opinion and doctor recommendation
rf_temp = RandomForestClassifier(n_estimators = 30, min_samples_leaf = 2,
                                 class_weight = "balanced", max_depth = 3, random_state=123)

#columns that have high correlation such as opinion and doctor recommendation column were used to fit the model
rf_temp.fit(train_h[["opinion_h1n1_vacc_effective", "opinion_h1n1_risk",
                    "opinion_h1n1_sick_from_vacc", "opinion_seas_vacc_effective",
                    "opinion_seas_risk", "opinion_seas_sick_from_vacc", "doctor_recc_h1n1",
                    "doctor_recc_seasonal", "chronic_med_condition"]],
                    target_h)

#Random forest estimator tree plotted
from sklearn.tree import plot_tree
tree = rf_temp.estimators_[0]
plt.figure(figsize=(80,40))
plot_tree(tree, feature_names = ["opinion_h1n1_vacc_effective", "opinion_h1n1_risk",
                    "opinion_h1n1_sick_from_vacc", "opinion_seas_vacc_effective",
                    "opinion_seas_risk", "opinion_seas_sick_from_vacc", "doctor_recc_h1n1",
                    "doctor_recc_seasonal", "chronic_med_condition"],
          class_names=['not vaccinated', "vaccinated"],filled=True);

#contigency table is created for doctor recommendation on h1n1 and h1n1 vaccination to perform chisq test
contingency_table = pd.crosstab(target_h, train_h['doctor_recc_h1n1'])
contingency_table

"""Null Hypothesis : There is no relationship between doctor recommendation of H1n1 vaccine and H1N1 vaccination

Alternate Hypothesis : There is relationship between doctor recommendation of H1n1 vaccine and H1N1 vaccination
"""

#Chi square test is created for doctor recommendation on h1n1 and h1n1 vaccination.
from scipy.stats import chi2_contingency
from scipy.stats import chi2
alpha = 0.05
chisq, p, df, expected = chi2_contingency(contingency_table)
critical_chisq_value = chi2.ppf(1 - alpha, df)
print("Chi-square critical value:" , critical_chisq_value)
print("Chi-square value:", chisq)
print("Probability:", p)
print("Degrees of freedom:", df)
print("Expected frequencies:", expected)

if p < alpha:
    print("we reject the Null hypothesis")
    print("There is 95% confidence that there is relationship between doctor recommendation of H1n1 vaccine and H1N1 vaccination")
else:
    print("we accept the Null hypothesis")
    print("There is 95% confidence that there is no relationship between doctor recommendation of H1n1 vaccine and H1N1 vaccination")

#Chi square test is created for opinion h1n1 vaccination effective and h1n1 vaccination.
contingency_table = pd.crosstab(target_h, train_h['opinion_h1n1_vacc_effective'])
contingency_table

"""Null Hypothesis : There is no relationship between opinion of H1N1 vaccine effective and H1N1 vaccination

Alternate Hypothesis : There is relationship between opinion of H1N1 vaccine effective and H1N1 vaccination
"""

#Chi square test is created for H1N1 vaccination effective and h1n1 vaccination.
chisq, prob, df, expected = chi2_contingency(contingency_table)
alpha = 0.05
critical_chisq_value = chi2.ppf(1 - alpha, df)
print("Chi-square critical value:" , critical_chisq_value)
print("Chi-square value:", chisq)
print("Probability:", prob)
print("Degrees of freedom:", df)
print("Expected frequencies:", expected)

if prob < alpha:
    print("we reject the Null hypothesis")
    print("There is 95% confidence that there is relationship between opinion on H1N1 vaccine effective and H1N1 vaccination")
else:
    print("we accept the Null hypothesis")
    print("There is 95% confidence that there is no relationship between opinion on H1N1 vaccine effective and H1N1 vaccination")

#Chi square test is created for opinion seasonal vaccination effective and h1n1 vaccination.
contingency_table = pd.crosstab(target_h, train_h['opinion_seas_vacc_effective'])
contingency_table

"""Null Hypothesis : There is no relationship between opinion of seasonal vaccine effective and H1N1 vaccination

Alternate Hypothesis : There is relationship between opinion of seasonal vaccine effective and H1N1 vaccination
"""

#Chi square test is created for opinion of seasonal vaccine effective and h1n1 vaccination.
chisq, prob, df, expected = chi2_contingency(contingency_table)
alpha = 0.05
critical_chisq_value = chi2.ppf(1 - alpha, df)
print("Chi-square critical value:" , critical_chisq_value)
print("Chi-square value:", chisq)
print("Probability:", prob)
print("Degrees of freedom:", df)
print("Expected frequencies:", expected)

if prob < alpha:
    print("we reject the Null hypothesis")
    print("There is 95% confidence that there is relationship between opinion on seasonal vaccine effective and H1N1 vaccination")
else:
    print("we accept the Null hypothesis")
    print("There is 95% confidence that there is no relationship between opinion on seasonal vaccine effective and H1N1 vaccination")

"""Seasonal flu preprocessing and Analysis

The same data preprocessing technique like H1N1 followed
"""

df_s = df.copy()

#columns and missing value for analyses on seasonal vaccine

columns = ["education", "income_poverty", "employment_status",
            "household_adults", "employment_industry",
           "employment_occupation", "marital_status", "h1n1_concern",
           "h1n1_knowledge", "opinion_h1n1_vacc_effective", "opinion_h1n1_risk",
           "opinion_h1n1_sick_from_vacc", "opinion_seas_vacc_effective",
           "behavioral_antiviral_meds", "behavioral_avoidance", "behavioral_face_mask",
           "behavioral_wash_hands", "behavioral_large_gatherings","behavioral_outside_home",
           "behavioral_touch_face", "doctor_recc_h1n1","doctor_recc_seasonal",
           "chronic_med_condition", "child_under_6_months", "health_worker"
          ]
missing_values = ["unknown", "unknown", "unknown", -1,
                  "unknown", "unknown", "unknown", -1, -1, -1,-1,
                  -1, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
index = 0;

#proportion of values with missing values with respect to seasonal vaccine
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
for i in range(2):
    for j in range(2):
        df_h_Lok[columns[index]].fillna(missing_values[index],inplace=True)
        grouped = df_h_Lok.groupby([columns[index], 'seasonal_vaccine']).size().reset_index(name='count')
        grouped[columns[index]] = grouped[columns[index]].astype(str)
        grouped['percentage'] = grouped.groupby(columns[index])['count'].apply(lambda x: 100 * x / x.sum())
        labels = grouped[columns[index]].unique()
        sns.histplot(data=grouped, x=columns[index], hue='seasonal_vaccine', weights='percentage', multiple='stack',ax=axes[i,j])
        axes[i,j].tick_params(axis='x', rotation=45)
        index = index + 1

plt.tight_layout()
plt.show()

#proportion of values with missing values with respect to seasonal vaccine
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
for i in range(2):
    for j in range(2):
        df_h_Lok[columns[index]].fillna(missing_values[index],inplace=True)
        grouped = df_h_Lok.groupby([columns[index], 'seasonal_vaccine']).size().reset_index(name='count')
        grouped[columns[index]] = grouped[columns[index]].astype(str)
        grouped['percentage'] = grouped.groupby(columns[index])['count'].apply(lambda x: 100 * x / x.sum())
        labels = grouped[columns[index]].unique()
        sns.histplot(data=grouped, x=columns[index], hue='seasonal_vaccine', weights='percentage', multiple='stack',ax=axes[i,j])
        axes[i,j].tick_params(axis='x', rotation=45)
        index = index + 1

plt.tight_layout()
plt.show()

#proportion of values with missing values with respect to seasonal vaccine
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
for i in range(2):
    for j in range(2):
        df_h_Lok[columns[index]].fillna(missing_values[index],inplace=True)
        grouped = df_h_Lok.groupby([columns[index], 'seasonal_vaccine']).size().reset_index(name='count')
        grouped[columns[index]] = grouped[columns[index]].astype(str)
        grouped['percentage'] = grouped.groupby(columns[index])['count'].apply(lambda x: 100 * x / x.sum())
        labels = grouped[columns[index]].unique()
        sns.histplot(data=grouped, x=columns[index], hue='seasonal_vaccine', weights='percentage', multiple='stack',ax=axes[i,j])
        axes[i,j].tick_params(axis='x', rotation=45)
        index = index + 1

plt.tight_layout()
plt.show()

#proportion of values with missing values with respect to seasonal vaccine
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
for i in range(2):
    for j in range(2):
        df_h_Lok[columns[index]].fillna(missing_values[index],inplace=True)
        grouped = df_h_Lok.groupby([columns[index], 'seasonal_vaccine']).size().reset_index(name='count')
        grouped[columns[index]] = grouped[columns[index]].astype(str)
        grouped['percentage'] = grouped.groupby(columns[index])['count'].apply(lambda x: 100 * x / x.sum())
        labels = grouped[columns[index]].unique()
        sns.histplot(data=grouped, x=columns[index], hue='seasonal_vaccine', weights='percentage', multiple='stack',ax=axes[i,j])
        axes[i,j].tick_params(axis='x', rotation=45)
        index = index + 1

plt.tight_layout()
plt.show()

#proportion of values with missing values with respect to seasonal vaccine
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
for i in range(2):
    for j in range(2):
        df_h_Lok[columns[index]].fillna(missing_values[index],inplace=True)
        grouped = df_h_Lok.groupby([columns[index], 'seasonal_vaccine']).size().reset_index(name='count')
        grouped[columns[index]] = grouped[columns[index]].astype(str)
        grouped['percentage'] = grouped.groupby(columns[index])['count'].apply(lambda x: 100 * x / x.sum())
        labels = grouped[columns[index]].unique()
        sns.histplot(data=grouped, x=columns[index], hue='seasonal_vaccine', weights='percentage', multiple='stack',ax=axes[i,j])
        axes[i,j].tick_params(axis='x', rotation=45)
        index = index + 1

plt.tight_layout()
plt.show()

#proportion of values with missing values with respect to seasonal vaccine
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
for i in range(2):
    for j in range(2):
        df_h_Lok[columns[index]].fillna(missing_values[index],inplace=True)
        grouped = df_h_Lok.groupby([columns[index], 'seasonal_vaccine']).size().reset_index(name='count')
        grouped[columns[index]] = grouped[columns[index]].astype(str)
        grouped['percentage'] = grouped.groupby(columns[index])['count'].apply(lambda x: 100 * x / x.sum())
        labels = grouped[columns[index]].unique()
        sns.histplot(data=grouped, x=columns[index], hue='seasonal_vaccine', weights='percentage', multiple='stack',ax=axes[i,j])
        axes[i,j].tick_params(axis='x', rotation=45)
        index = index + 1

plt.tight_layout()
plt.show()

#proportion of values with missing values with respect to seasonal vaccine
df_h_Lok[columns[index]].fillna(missing_values[index],inplace=True)
grouped = df_h_Lok.groupby([columns[index], 'seasonal_vaccine']).size().reset_index(name='count')
grouped[columns[index]] = grouped[columns[index]].astype(str)
grouped['percentage'] = grouped.groupby(columns[index])['count'].apply(lambda x: 100 * x / x.sum())
labels = grouped[columns[index]].unique()
sns.histplot(data=grouped, x=columns[index], hue='seasonal_vaccine', weights='percentage', multiple='stack')
plt.tight_layout()
plt.show()

#missing values are replaced with the values which have same proportion
df_s["education"].fillna("< 12 Years", inplace=True)
df_s["income_poverty"].fillna("<= $75,000, Above Poverty", inplace=True)
df_s["employment_status"].fillna("Employed", inplace=True)
df_s["household_adults"].fillna(-1, inplace=True)
df_s["marital_status"].fillna("unknown", inplace=True)
df_s["h1n1_concern"].fillna(1, inplace=True)
df_s["h1n1_knowledge"].fillna(1, inplace=True)
df_s["opinion_h1n1_vacc_effective"].fillna(3, inplace=True)
df_s["opinion_h1n1_risk"].fillna(2, inplace=True)
df_s["opinion_h1n1_sick_from_vacc"].fillna(3, inplace=True)
df_s["opinion_seas_vacc_effective"].fillna(4, inplace=True)
df_s["opinion_seas_risk"].fillna(2, inplace=True)
df_s["behavioral_antiviral_meds"].fillna(2, inplace=True)
df_s["behavioral_avoidance"].fillna(1, inplace=True)
df_s["behavioral_face_mask"].fillna(1, inplace=True)
df_s["behavioral_wash_hands"].fillna(1, inplace=True)
df_s["behavioral_large_gatherings"].fillna(1, inplace=True)
df_s["behavioral_outside_home"].fillna(1, inplace=True)
df_s["behavioral_touch_face"].fillna(1, inplace=True)
df_s["doctor_recc_h1n1"].fillna(0, inplace=True)
df_s["doctor_recc_seasonal"].fillna(0, inplace=True)
df_s["chronic_med_condition"].fillna(0, inplace=True)
df_s["child_under_6_months"].fillna(-1, inplace=True)
df_s["health_worker"].fillna(0, inplace=True)
df_s["health_insurance"].fillna(1, inplace=True)
df_s["household_children"].fillna(-1, inplace=True)
df_s["rent_or_own"].fillna("Own", inplace=True)
df_s["opinion_seas_sick_from_vacc"].fillna(5, inplace=True)

#the attribute called geo region percentage is created in which the values are the percentage of vaccination
#taken in that region
aggreg = df_s.groupby(['hhs_geo_region', 'seasonal_vaccine']).size().reset_index(name='count')
aggreg['percentage'] = aggreg.groupby('hhs_geo_region')['count'].apply(lambda x: 100 * x / x.sum())
seasonal_vaccine_percentage = {}
for d in aggreg.loc[aggreg["seasonal_vaccine"] == 1].iterrows():
    seasonal_vaccine_percentage[d[1][0]] = round(d[1][3] /100, 4)
df_s['hhs_geo_region_percentage'] = df_s["hhs_geo_region"].map(seasonal_vaccine_percentage)

#the attribute called employment industry percentage is created in which the values are the percentage of people
#in that industry taken vaccination
df_s["employment_industry"].fillna("unknown", inplace=True)
aggreg = df_s.groupby(['employment_industry', 'seasonal_vaccine']).size().reset_index(name='count')
aggreg['percentage'] = aggreg.groupby('employment_industry')['count'].apply(lambda x: 100 * x / x.sum())
employment_industry_percentage = {}
for d in aggreg.loc[aggreg["seasonal_vaccine"] == 1].iterrows():
    employment_industry_percentage[d[1][0]] = round(d[1][3] /100, 4)
df_s['employment_industry_percentage'] = df_s["employment_industry"].map(employment_industry_percentage)

#the attribute called employment occupation percentage is created in which the values are the percentage of people
#in that occupation taken vaccination
df_s["employment_occupation"].fillna("unknown", inplace=True)
aggreg = df_s.groupby(['employment_occupation', 'seasonal_vaccine']).size().reset_index(name='count')
aggreg['percentage'] = aggreg.groupby('employment_occupation')['count'].apply(lambda x: 100 * x / x.sum())
employment_occupation_percentage = {}
for d in aggreg.loc[aggreg["seasonal_vaccine"] == 1].iterrows():
    employment_occupation_percentage[d[1][0]] = round(d[1][3] /100,4)
df_s['employment_occupation_percentage'] = df_s["employment_occupation"].map(employment_occupation_percentage)

#the data is processed using pipeline and column transfer
X = do_preprocess(df_s)
df_s_p = pd.DataFrame(X, columns=columns_label)

#target and other attributes were seperated
train_s = df_s_p.iloc[:,:-2]
target_s = df_s_p.iloc[:,-1]

#test and training set are separated
from sklearn.model_selection import train_test_split
X_tr, X_tt, Y_tr, Y_tt =  train_test_split(train_s, target_s, test_size = 0.2,random_state=42,shuffle=True)

#using randomizedCV with K-fold cross validation 3 optimal parameter for XGBosst classifier is searched
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

xgb = XGBClassifier(learning_rate = 0.01, objective = 'binary:logistic', eval_metric="auc")
param_xgb = {
    "max_depth": [5,8,10,15],
    "subsample" : [1, 0.9,0.8, 0.7],
    "n_estimators" : [100, 120, 200, 250],
    "gamma" : [0, 0.25, 0.5],
    "alpha" : [0, 0.7, 0.9, 1]

}
xgb_m = RandomizedSearchCV(estimator=xgb, param_distributions=param_xgb, cv=3,n_jobs=-1, n_iter=30)
xgb_m.fit(X_tr, Y_tr)

#best parameter for xgboost classifer
xgb_m.best_params_

#Random forest classifier is used to find best parameters to predict seasonal vaccine
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
dcls = RandomForestClassifier(n_jobs=-1, class_weight = "balanced")
param_dist = {
    'n_estimators': np.arange(360, 370, 3),
    'max_depth': [22,23,24,25],
    'max_features': ['sqrt', 'log2','auto'],
    "min_samples_leaf" : [2, 3, 4]
    }

random_search = RandomizedSearchCV(estimator=dcls, param_distributions=param_dist, cv=5, n_jobs=-1, n_iter=30)
random_search.fit(X_tr, Y_tr)
random_search.best_params_

#Random forest classifier is obtained using best parameter obtained
rf_best_s = RandomForestClassifier(class_weight = "balanced", n_estimators = 369, min_samples_leaf = 4,
                                   max_features = 'sqrt', max_depth = 25, n_jobs=-1)

#ROC AUC score using cross validation for random forest classifier
crs_rf_s = cross_val_predict(rf_best_s, X_tr, Y_tr, cv = 3, method = "predict_proba")
roc_auc_score(Y_tr, crs_rf_s[:, 1])

#ROC AUC score of XGBoost classifier is obtained using cross validation
xgb_best_s = XGBClassifier(learning_rate = 0.01, objective = 'binary:logistic', eval_metric="auc", subsample = 0.8,
                           n_estimators = 250, max_depth = 8, gamma = 0.5, alpha = 1)
crs_xgb_s = cross_val_predict(xgb_best_s, X_tr, Y_tr, cv = 3, method = "predict_proba")
roc_auc_score(Y_tr, crs_xgb_s[:, 1])

#models were fitted on training set and prediction for test set is obtained


xgb_best_s = XGBClassifier(learning_rate = 0.01, objective = 'binary:logistic', eval_metric="auc", subsample = 0.9,
                         n_estimators = 250, max_depth = 8, gamma = 0.25, alpha = 0)
xgb_best_s.fit(X_tr, Y_tr)
xgb_best_s_p = xgb_best_s.predict_proba(X_tt)[:, 1]
xgb_best_p = xgb_best_s.predict(X_tt)


rf_best_s = RandomForestClassifier(class_weight = "balanced", n_estimators = 366, min_samples_leaf = 4,
                                   max_features = 'sqrt', max_depth = 25, n_jobs=-1)
rf_best_s.fit(X_tr, Y_tr)
rf_best_s_p = rf_best_s.predict_proba(X_tt)[:, 1]
rf_best_p = rf_best_s.predict(X_tt)

#precision, recall, f1score, rocauc score for test set is obtained
from sklearn.metrics import precision_score, recall_score,f1_score
print(precision_score(Y_tt, xgb_best_p))
print(recall_score(Y_tt, xgb_best_p))
print(f1_score(Y_tt, xgb_best_p))
print(roc_auc_score(Y_tt, xgb_best_s_p))

#precision, recall, f1score, rocauc score for test set is obtained
from sklearn.metrics import precision_score, recall_score,f1_score
print(precision_score(Y_tt, rf_best_p))
print(recall_score(Y_tt, rf_best_p))
print(f1_score(Y_tt, rf_best_p))
print(roc_auc_score(Y_tt, rf_best_s_p))

#false positive and true positive rate obtained for test set
from sklearn.metrics import roc_curve

fpr_xg, tpr_xg, thresholds = roc_curve(Y_tt, xgb_best_s_p)

fpr_rf, tpr_rf, thresholds = roc_curve(Y_tt, rf_best_s_p)

plt.plot(fpr_xg, tpr_xg, label="XGBoost classifier")
plt.plot(fpr_rf, tpr_rf, label="Random Forest Classifier")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("Seasonal Vaccine")
plt.legend(loc="lower right")
plt.axis([0, 1, 0, 1])
plt.plot([0, 1], [0, 1], color='r', linestyle='--')
plt.show()

correlation_matrix = df_s_p.corr()

correlation_matrix["seasonal_vaccine"].abs().sort_values(ascending=False).index

important_correlation = df_h_p[["seasonal_vaccine","opinion_h1n1_vacc_effective", "opinion_h1n1_risk",
                              "opinion_h1n1_sick_from_vacc", "opinion_seas_vacc_effective",
                              "opinion_seas_risk", "opinion_seas_sick_from_vacc",
                               "doctor_recc_h1n1", "doctor_recc_seasonal", "chronic_med_condition"]].corr()

sns.heatmap(important_correlation,
            annot=True, cmap="coolwarm")

#random forest classifier with depth 3 using columns which have high correlation
rf_sub_question = RandomForestClassifier(n_estimators = 30, min_samples_leaf = 2,
                                 class_weight = "balanced",  max_depth = 3, random_state=123)

#column with high correlation such as opinion and doctor recommendation
rf_sub_question.fit(train_s[["opinion_h1n1_vacc_effective", "opinion_h1n1_risk",
                    "opinion_h1n1_sick_from_vacc", "opinion_seas_vacc_effective",
                    "opinion_seas_risk", "opinion_seas_sick_from_vacc", "doctor_recc_h1n1",
                    "doctor_recc_seasonal", "chronic_med_condition"]],
                    target_s)

#from sklearn.tree import plot_tree
tree = rf_sub_question.estimators_[0]
plt.figure(figsize=(80,40))
plot_tree(best_tree, feature_names = ["opinion_h1n1_vacc_effective", "opinion_h1n1_risk",
                    "opinion_h1n1_sick_from_vacc", "opinion_seas_vacc_effective",
                    "opinion_seas_risk", "opinion_seas_sick_from_vacc", "doctor_recc_h1n1",
                    "doctor_recc_seasonal", "chronic_med_condition"],
          class_names=['not vaccinated', "vaccinated"],filled=True);
plt.show()

#contigency table for doctor recommendation and seasonal vaccine
contingency_table = pd.crosstab(target_s, train_s['doctor_recc_h1n1'])
contingency_table

"""Null Hypothesis : There is no relationship between doctor recommendation of H1n1 vaccine and seasonal vaccination
Alternate Hypothesis : There is relationship between doctor recommendation of H1n1 vaccine and seasonal vaccination
"""

#chi square test
from scipy.stats import chi2_contingency
from scipy.stats import chi2
alpha = 0.05
chisq, p, df, expected = chi2_contingency(contingency_table)
critical_chisq_value = chi2.ppf(1 - alpha, df)
print("Chi-square critical value:" , critical_chisq_value)
print("Chi-square value:", chisq)
print("Probability:", p)
print("Degrees of freedom:", df)
print("Expected frequencies:", expected)

if p < alpha:
    print("we reject the Null hypothesis")
    print("There is 95% confidence that there is relationship between doctor recommendation of H1n1 vaccine and seasonal vaccination")
else:
    print("we accept the Null hypothesis")
    print("There is 95% confidence that there is no relationship between doctor recommendation of H1n1 vaccine and seasonal vaccination")

"""Null Hypothesis : There is no relationship between people opinion on seasonal flu risk and seasonal vaccination
Alternate Hypothesis : There is relationship between opinion on seasonal flu risk and seasonal vaccination
"""

#contigency table for opinion of seasonal flu risk and seasonal vaccine
contingency_table = pd.crosstab(target_s, train_s['opinion_seas_risk'])
contingency_table

#chi sq test
chisq, prob, df, expected = chi2_contingency(contingency_table)
alpha = 0.05
critical_chisq_value = chi2.ppf(1 - alpha, df)
print("Chi-square critical value:" , critical_chisq_value)
print("Chi-square value:", chisq)
print("Probability:", prob)
print("Degrees of freedom:", df)
print("Expected frequencies:", expected)

if prob < alpha:
    print("we reject the Null hypothesis")
    print("There is 95% confidence that there is relationship between opinion on seasonal flu risk and seasonal vaccination")
else:
    print("we accept the Null hypothesis")
    print("There is 95% confidence that there is no relationship between opinion on seasonal flu risk and seasonal vaccination")
