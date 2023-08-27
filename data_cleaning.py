# importing libraries
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import scipy.stats as stats
import datetime

print(f"Started time : {datetime.datetime.now()")
#here are the functions to claen  the data 
def get_column_details(df, column):  # to get the column details
    print("Details of", column, "column")

    # DataType of column
    print("\nDataType: ", df[column].dtype)

    # Check if null values are present
    count_null = df[column].isnull().sum()
    if count_null == 0:
        print("\nThere are no null values")
    elif count_null > 0:
        print("\nThere are ", count_null, " null values")

    # Get Number of Unique Values
    print("\nNumber of Unique Values: ", df[column].nunique())

    # Get Distribution of Column
    print("\nDistribution of column:\n")
    print(df[column].value_counts())


def fill_missing_with_group_mode(df, groupby, column):  # filling missing data
    print("\nNo. of missing values before filling with group mode:",
          df[column].isnull().sum())

    # Fill with local mode
    mode_per_group = df.groupby(groupby)[column].transform(
        lambda x: x.mode().iat[0])
    df[column] = df[column].fillna(mode_per_group)

    print("\nNo. of missing values after filling with group mode:",
          df[column].isnull().sum())


# Handle Outliers and null values
def fix_inconsistent_values(df, groupby, column):

    print("\nExisting Min, Max Values:", df[column].apply(
        [min, max]), sep='\n', end='\n')

    df_dropped = df[df[column].notna()].groupby(groupby)[column].apply(list)
    x, y = df_dropped.apply(lambda x: stats.mode(x)).apply([min, max])
    mini, maxi = x[0], y[0]

    # assign Wrong Values to NaN
    col = df[column].apply(lambda x: np.NaN if (
        (x < mini) | (x > maxi) | (x < 0)) else x)

    # fill with local mode
    mode_by_group = df.groupby(groupby)[column].transform(
        lambda x: x.mode()[0] if not x.mode().empty else np.NaN)
    df[column] = col.fillna(mode_by_group)
    df[column].fillna(df[column].mean(), inplace=True)

    print("\nAfter Cleaning Min, Max Values:",
          df[column].apply([min, max]), sep='\n', end='\n')
    print("\nNo. of Unique values after Cleaning:", df[column].nunique())
    print("\nNo. of Null values after Cleaning:", df[column].isnull().sum())

# Method to clean categorical field


def clean_categorical_field(df, groupby, column, replace_value=None):

    # Replace with np.nan
    if replace_value != None:
        df[column] = df[column].replace(replace_value, np.nan)
        print(f"\nGarbage value {replace_value} is replaced with np.nan")

    # For each Customer_ID, assign same value for the column
    fill_missing_with_group_mode(df, groupby, column)

# Method to clean Numerical Fields


def clean_numerical_field(df, groupby, column, strip=None, datatype=None, replace_value=None):

    # Replace with np.nan
    if replace_value != None:
        df[column] = df[column].replace(replace_value, np.nan)
        print(f"\nGarbage value {replace_value} is replaced with np.nan")

    # Remove trailing & leading special characters
    if df[column].dtype == object and strip is not None:
        df[column] = df[column].str.strip(strip)
        print(f"\nTrailing & leading {strip} are removed")

    # Change datatype
    if datatype is not None:
        df[column] = df[column].astype(datatype)
        print(f"\nDatatype of {column} is changed to {datatype}")

    if column == 'Changed_Credit_Limit':
        mode_by_group = df.groupby(groupby)[column].transform(
            lambda x: x.mode()[0] if not x.mode().empty else np.NaN)
        df[column] = df[column].fillna(mode_by_group)
    else:
        fix_inconsistent_values(df, groupby, column)


def Month_Converter(val):  # Month converter method for credit history age
    if pd.notnull(val):
        years = int(val.split(' ')[0])
        month = int(val.split(' ')[3])
        return (years*12)+month
    else:
        return val


def replace_null_values_for_col_CHA(s):  # for credit history age
    s = s.copy()  # Make a copy to avoid modifying the original group
    for i in range(len(s)):
        if pd.isna(s.iloc[i]):  # Use .iloc to access elements by position
            if i == 0:
                s.iloc[i] = s.iloc[len(s)-1] + 1
            else:
                s.iloc[i] = s.iloc[i-1] + 1
    return s


# loading data..
print("Loading data...")
df_train_original = pd.read_csv('train.csv')
df_train = df_train_original.copy()


print("Checking null values count column wise : ")
df_train.isnull().sum()  # check null values count column wise

# Convert Month to datetime object
df_train['Month'] = pd.to_datetime(df_train.Month, format='%B').dt.month

# Handle Type of Loan null values
df_train['Type_of_Loan'].replace([np.NaN], 'Not Specified', inplace=True)

group_by = 'Customer_ID'  # group by column name
categorial_columns = ['Name', 'SSN', 'Occupation',
                      'Credit_Mix', 'Payment_Behaviour']

print("Started cleaning for categorial columns...")

for col_name in categorial_columns:
    if col_name == 'Name':
        clean_categorical_field(df_train, group_by, col_name)
    elif col_name == 'SSN':
        garbage_value = '#F%$D@*&8'
        clean_categorical_field(df_train, group_by, col_name, garbage_value)
    elif col_name == 'Occupation':
        garbage_value = '_______'
        clean_categorical_field(df_train, group_by, col_name, garbage_value)
    elif col_name == 'Credit_Mix':
        garbage_value = '_'
        clean_categorical_field(df_train, group_by, col_name, garbage_value)
    elif col_name == 'Payment_Behaviour':
        garbage_value == '!@9#%8'
        clean_categorical_field(df_train, group_by, col_name, garbage_value)

print("done cleaning for categorial columns...")


print("Started cleaning for numerical columns..")

numerical_columns = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Delay_from_due_date', 'Num_of_Delayed_Payment',
                     'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_History_Age', 'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance', 'Num_of_Loan',]


for col_name in numerical_columns:
    match col_name:
        case 'Age':
            clean_numerical_field(df_train, group_by,
                                  col_name, strip='_', datatype='int')
        case 'Annual_Income':
            clean_numerical_field(df_train, group_by,
                                  col_name, strip='_', datatype='float')
        case 'Monthly_Inhand_Salary':
            clean_numerical_field(df_train, group_by, col_name)
        case 'Num_Bank_Accounts':
            clean_numerical_field(df_train, group_by, col_name)
        case 'Num_Credit_Card':
            clean_numerical_field(df_train, group_by, col_name)
        case 'Interest_Rate':
            clean_numerical_field(df_train, group_by, col_name)
        case 'Delay_from_due_date':
            clean_numerical_field(df_train, group_by, col_name)
        case 'Num_of_Delayed_Payment':
            clean_numerical_field(df_train, group_by,
                                  col_name, strip='_', datatype='float')
        case 'Changed_Credit_Limit':
            clean_numerical_field(
                df_train, group_by, col_name, strip='_', datatype='float', replace_value='_')

        case 'Num_Credit_Inquiries':
            clean_numerical_field(df_train, group_by, col_name)
        case 'Outstanding_Debt':
            clean_numerical_field(df_train, group_by,
                                  col_name, strip='_', datatype=float)
        case 'Credit_History_Age':
            df_train['Credit_History_Age'] = df_train['Credit_History_Age'].apply(
                lambda x: Month_Converter(x)).astype(float)

            # first time calling
            mode_by_group = df_train.groupby('Customer_ID')['Credit_History_Age'].transform(
                lambda x: replace_null_values_for_col_CHA(x))
            df_train['Credit_History_Age'] = df_train['Credit_History_Age'].fillna(
                mode_by_group)

            # second time calling
            mode_by_group = df_train.groupby('Customer_ID')['Credit_History_Age'].transform(
                lambda x: replace_null_values_for_col_CHA(x))
            df_train['Credit_History_Age'] = df_train['Credit_History_Age'].fillna(
                mode_by_group)

        case 'Total_EMI_per_month':
            clean_numerical_field(df_train, group_by, col_name)
        case 'Amount_invested_monthly':
            clean_numerical_field(df_train, group_by,
                                  col_name, datatype=float, strip='_')
        case 'Monthly_Balance':
            df_train[col_name].replace('', np.nan)

            clean_numerical_field(df_train, group_by, col_name, strip='_',
                                  datatype=float, replace_value='__-333333333333333333333333333__')
        case 'Num_of_Loan':
            clean_numerical_field(df_train, group_by,
                                  col_name, strip='_', datatype=float)


print("Started cleaning for numerical columns..")

print(f" Null count after cleaning process : {df_train.isnull().sum()}")

print("Data transformation start..")

drop_columns = ['ID', 'Customer_ID', 'Name', 'SSN']

df_train.drop(drop_columns, axis=1, inplace=True)

# Define a mapping of labels to numerical values
label_mapping = {'Poor': 0, 'Standard': 1, 'Good': 2}

# Apply label encoding to the 'Category' column
df_train['Credit_Score'] = df_train['Credit_Score'].map(label_mapping)

label_mapping_credit_mix = {'Bad': 0, 'Standard': 1, 'Good': 2}

# Apply label encoding to the 'Category' column
df_train['Credit_Mix'] = df_train['Credit_Mix'].map(label_mapping_credit_mix)

print("Label encoding..")

# Label Encoding

categorical_columns = ['Occupation', 'Type_of_Loan',
                       'Payment_of_Min_Amount', 'Payment_Behaviour']
# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Loop through each column and apply label encoding
for column in categorical_columns:
    df_train[column] = label_encoder.fit_transform(df_train[column])

print("Data transformation end..")

print("storing dataframe to csv..")
df_train.to_csv("Final_data2.csv", index=False)
print("finally done..")

print(f"Ended time : {datetime.datetime.now()}")
