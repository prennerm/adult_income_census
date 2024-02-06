# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# create unified dataframe
df_bank_a = pd.read_csv("data/BankA.csv")
df_bank_b = pd.read_csv("data/BankB.csv")
df_bank_c = pd.read_csv("data/BankC.csv")
df_all = pd.concat([df_bank_a, df_bank_b, df_bank_c])

# repeat Leo's steps of preprocessing the data

# strip all string values from the dataset
df = df_all.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
# combine Never-worked and Without-pay into one category
df["workclass"] = df["workclass"].replace(
    ["Never-worked", "Without-pay"], "Not-working"
)
df["workclass"] = df["workclass"].replace(["?", "*"], "unknown")
df["workclass"].unique()

# combine Married-civ-spouse and Married-AF-spouse into one category
df["marital-status"] = df["marital-status"].replace(
    ["Married-civ-spouse", "Married-AF-spouse"], "Married"
)
df["marital-status"].unique()
# combine Husband and Wife into one category
df["relationship"] = df["relationship"].replace(["Husband", "Wife"], "Parent")
df["relationship"].unique()

# replace income by 0 and 1
df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})

# replace occupation by 4 categories (low, medium, high, unknown) based on the mean of income
df["occupation"] = df["occupation"].replace(
    ["Exec-managerial", "Prof-specialty"], "high"
)
df["occupation"] = df["occupation"].replace(
    [
        "Armed-Forces",
        "Protective-serv",
        "Tech-support",
        "Sales",
        "Craft-repair",
        "Transport-moving",
    ],
    "medium",
)
df["occupation"] = df["occupation"].replace(
    [
        "Adm-clerical",
        "Machine-op-inspct",
        "Farming-fishing",
        "Handlers-cleaners",
        "Other-service",
        "Priv-house-serv",
    ],
    "low",
)
df["occupation"] = df["occupation"].replace(["?"], "unknown")
df["occupation"].unique()

# map native-country to continents
df["native-country"] = df["native-country"].str.strip()
df["native-country"] = df["native-country"].replace(
    [
        "United-States",
        "Puerto-Rico",
        "Canada",
        "Outlying-US(Guam-USVI-etc)",
        "Cuba",
        "Jamaica",
        "Mexico",
        "Dominican-Republic",
        "El-Salvador",
        "Guatemala",
        "Haiti",
        "Honduras",
        "Nicaragua",
        "Trinadad&Tobago",
        "Peru",
        "Ecuador",
        "Columbia",
        "Honduras",
        "Haiti",
        "Guatemala",
        "El-Salvador",
        "Dominican-Republic",
        "Columbia",
        "Ecuador",
        "Peru",
        "Jamaica",
        "Mexico",
        "Puerto-Rico",
        "Cuba",
        "Outlying-US(Guam-USVI-etc)",
        "Canada",
        "United-States",
    ],
    "North-America",
)
df["native-country"] = df["native-country"].replace(
    [
        "Germany",
        "England",
        "Italy",
        "Poland",
        "Portugal",
        "Ireland",
        "France",
        "Yugoslavia",
        "Scotland",
        "Greece",
        "Hungary",
        "Holand-Netherlands",
    ],
    "Europe",
)
df["native-country"] = df["native-country"].replace(
    [
        "Philippines",
        "India",
        "China",
        "Japan",
        "Vietnam",
        "Taiwan",
        "Iran",
        "Thailand",
        "Hong",
        "Cambodia",
        "Laos",
    ],
    "Asia",
)
df["native-country"] = df["native-country"].replace(
    ["South", "Columbia", "Ecuador", "Peru"], "South-America"
)
df["native-country"] = df["native-country"].replace(
    [
        "Trinadad&Tobago",
        "Honduras",
        "Haiti",
        "Guatemala",
        "El-Salvador",
        "Dominican-Republic",
        "Columbia",
        "Ecuador",
        "Peru",
    ],
    "Central-America",
)
df["native-country"] = df["native-country"].replace(["?", "*"], "Unknown")
df["native-country"].unique()

df["education"] = df["education"].replace(
    ["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th"],
    "school",
)
df["education"] = df["education"].replace(
    ["Assoc-voc", "Assoc-acdm", "Prof-school", "Some-college"], "higher"
)
df["education"].unique()

# Drop the fnlwgt column
df.drop(["fnlwgt"], axis=1, inplace=True)

"""
# Matthias's function, saved here for reference
categorical_columns = df.select_dtypes(include=['object']).columns
 
for column in df[categorical_columns].columns:
    print(f"Column: {column}")
    print(df[column].value_counts())
    print("\n" + "="*30 + "\n")
"""

# drop institute column
df_all = df.drop(["institute"], axis=1)
df_bank_a = df[df["institute"] == "Bank A"]
df_bank_b = df[df["institute"] == "Bank B"]
df_bank_c = df[df["institute"] == "Bank C"]

# encode the categorical variables
df = pd.get_dummies(df_all, drop_first=True)

# save finished preprocessed dataframe
df.to_csv("preprocessed_df.csv", index=False)
