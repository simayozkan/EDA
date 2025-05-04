#EDA
#The dataset is not shared because it's exclusive to Miuul Data Science Bootcamp.
#Game company dataset / Company wants to estimate how much income that new customers can bring to the company on average.
#It is important to create segments in this analysis.

#Variables :
# Price: Customer spending
# Source: Type of device the customer is connected
# Sex: Gender of customer
# Country: Country of customer
# Age: Age of customer



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("datasets")

#First look to dataframe
df.head()

df.shape

df.info()

#How many unique "Source" in the dataset?
df["SOURCE"].nunique()

df["SOURCE"].value_counts()

#How many unique "PRICE" in the dataset?
df["PRICE"].nunique()

df["PRICE"].value_counts()

#How many sales were made from which country?
df["COUNTRY"].value_counts()

df.groupby("COUNTRY").agg({"PRICE":"count"})

df.pivot_table(values="PRICE",index="COUNTRY",aggfunc="count")

#How much was earned from sales in total by country?
df.groupby("COUNTRY").agg({"PRICE": "sum"})

df.pivot_table(values="PRICE",index="COUNTRY",aggfunc="sum")

#What are the sales numbers by "SOURCE" types?
df["SOURCE"].value_counts()

#What are the "PRICE" averages by country?
df.groupby(by=['COUNTRY']).agg({"PRICE": "mean"})

#What are "PRICE" averages relative to SOURCE??
df.groupby(by=['SOURCE']).agg({"PRICE": "mean"})

#What are the "PRICE" averages in the "COUNTRY"-"SOURCE" breakdown?
df.groupby(by=["COUNTRY", 'SOURCE']).agg({"PRICE": "mean"})

#What are the average earnings in the "COUNTRY", "SOURCE", "SEX", "AGE" breakdown?
df.groupby(["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).head()

#Sort the output by "PRICE"
agg_df = df.groupby(by=["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)

agg_df.head(10)

#Convert the names in the index to variable names
agg_df = agg_df.reset_index()

agg_df.head()

agg_df.columns

#Convert the "AGE" variable to a categorical variable and add it to agg_df
df["AGE"]

bins = [0, 18, 23, 30, 40, agg_df["AGE"].max()]

#Add label to them
mylabels = ['0_18', '19_23', '24_30', '31_40', '41_' + str(agg_df["AGE"].max())]

#Divide "AGE"
agg_df["age_cat"] = pd.cut(agg_df["AGE"], bins, labels=mylabels)
agg_df.head()

agg_df["NEW_AGE"] = pd.cut(agg_df["AGE"], bins=[0,18,23,30,40,agg_df["AGE"].max()],
                           labels= ["0_18", "19_23", "24_30", "31_40", "41_"+ str(agg_df["AGE"].max())])

del agg_df["NEW_AGE"]

#Define new level based customers and add them to the data set as a variable
#Define a variable called customers level based and add this variable to the data set.
#Variable Names
agg_df.columns

#Put the values of the variables "COUNTRY", "SOURCE", "SEX" and "age_cat" next to each other and combine them with an underscore
agg_df['customers_level_based'] = agg_df[['COUNTRY', 'SOURCE', 'SEX', 'age_cat']].apply(lambda x: '_'.join(x).upper(), axis=1)

#In order to deduplicate the segments, take groupby
agg_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"})

# customers_level_based included in index. Convert it to variable
agg_df = agg_df.reset_index()
agg_df.head()

#Check for duplicate variables
agg_df["customers_level_based"].value_counts()

agg_df.head()

a=agg_df["customers_level_based"].value_counts()

a[a>1].count()

agg_df["CUSTOMER_LEVEL_BASED1"] = agg_df["customers_level_based"].drop_duplicates()


#Divide new customers into segments (USA_ANDROID_MALE_0_18)
#Segment by "PRICE"
#Give new name to segments as "SEGMENT" and add them to agg_df
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])

agg_df.head(30)

agg_df.groupby("SEGMENT").agg({"PRICE": ["mean","max","sum","count"]})

del agg_df["CUSTOMER_LEVEL_BASED1"]

#Classify new customers and estimate how much revenue they can bring.
#To which segment does a 33-year-old Turkish woman using ANDROID belong.
#On average, how much income is the woman expected to bring to the company?
new_user = "TUR_ANDROID_FEMALE_31_40"

agg_df[agg_df["customers_level_based"] == new_user]

agg_df[agg_df["customers_level_based"] =="TUR_ANDROID_FEMALE_31_40"]

#To which segment does a 35-year-old French woman using IOS belong.
#On average, how much income is the woman expected to bring to the company?
new_user1 = "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user1]
