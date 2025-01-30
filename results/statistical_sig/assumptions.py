import pandas as pd
from scipy.stats import levene, shapiro, ttest_ind

# Assume 'df' is your DataFrame with columns: 'Participant', 'Group', 'AgreementRate'
df = pd.read_csv('/Users/annalisaszymanski/qualtrics_survey/annalisa/results/agreement-tables-dietician.csv')

# Convert the 'judge' column to just "SME" and "LayUser" labels if it contains numbers.
df['judge'] = df['judge'].str.replace(r'\d+', '')

# Split the data into two groups
sme_data = df[df['judge'] == 'SME']['agreement']
lay_data = df[df['judge'] == 'LayUser']['agreement']

stat, p_levene = levene(sme_data, lay_data)
print(f"Levene's Test P-value: {p_levene}")

_, p_shapiro_sme = shapiro(sme_data)
_, p_shapiro_lay = shapiro(lay_data)
print(f"Shapiro-Wilk P-value for SMEs: {p_shapiro_sme}")
print(f"Shapiro-Wilk P-value for Lay Users: {p_shapiro_lay}")

# If assumptions are met, perform t-test
stat, p_ttest = ttest_ind(sme_data, lay_data, equal_var=True)
print(f"T-test Statistic: {stat}")
print(f"P-value: {p_ttest}")
