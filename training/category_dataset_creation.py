import pandas as pd
import numpy as np

# ---- import dataset from the Princeton Article
df = pd.read_csv('dark_patterns.csv')

print(df.info())

# ---- select from the dataset when 'Pattern String' is not NaN values.
df = df[pd.notnull(df["Pattern String"])]
print(df.info())

# check the distribution of the Pattern Category.
print('Distribution of Pattern Category:\n{}\n'.format(df['Pattern Category'].value_counts()))
# check the distribution of the Pattern Type.
print('Distribution of Pattern Type:\n{}\n'.format(df['Pattern Type'].value_counts()))

# select the types we are going to detect
types = ['Low-stock Message','Activity Notification','Confirmshaming','Countdown Timer',
         'Limited-time Message','High-demand Message','Pressured Selling','Trick Questions']

category = df[(df['Pattern Type'].isin(types))]
print(category.info())

# check the distribution of the Pattern Category.
print('Distribution of Pattern Category:\n{}\n'.format(category['Pattern Category'].value_counts()))
# check the distribution of the Pattern Type.
print('Distribution of Pattern Type:\n{}\n'.format(category['Pattern Type'].value_counts()))

# For later training the model, we should remove the duplicate input to reduce overfitting.
category_no_duplicate = category.drop_duplicates(subset="Pattern String")
print(category_no_duplicate.info())

# check the distribution of the Pattern Category.
print('Distribution of Pattern Category:\n{}\n'.format(category_no_duplicate['Pattern Category'].value_counts()))
# check the distribution of the Pattern Type.
print('Distribution of Pattern Type:\n{}\n'.format(category_no_duplicate['Pattern Type'].value_counts()))

# save the new category dataset
category_no_duplicate.to_csv('category.csv', index = False, header = True)