import pandas as pd
from scipy.stats import mannwhitneyu

df = pd.read_csv('/Users/annalisaszymanski/qualtrics_survey/annalisa/results/agreement-tables-dietician.csv')

df['judge'] = df['judge'].str.replace(r'\d+', '')

# Split the data into two groups
sme_data = df[df['judge'] == 'SME']['agreement']
lay_data = df[df['judge'] == 'LayUser']['agreement']

# Perform Mann-Whitney U Test
stat, p = mannwhitneyu(sme_data, lay_data, alternative='two-sided')

print(f"Mann-Whitney U Statistic: {stat}")
print(f"P-value: {p}")
