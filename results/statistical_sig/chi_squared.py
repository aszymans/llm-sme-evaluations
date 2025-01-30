import numpy as np
import scipy.stats as stats
import pandas as pd

df = pd.read_csv('/Users/annalisaszymanski/qualtrics_survey/annalisa/results/agreement-tables-mental_health.csv')
df2 = pd.read_csv('/Users/annalisaszymanski/qualtrics_survey/annalisa/results/agreement-tables-dietician.csv')

df = pd.concat([df, df2], ignore_index=True)

# Filter for Lay Users and SMEs
lay_users = df[df['judge'].str.contains('LayUser')]
smes = df[df['judge'].str.contains('SME')]

# Count how many times each group agreed and disagreed
lay_user_agree = lay_users['agreement'].sum()
lay_user_disagree = len(lay_users) - lay_user_agree 
sme_agree = smes['agreement'].sum()
sme_disagree = len(smes) - sme_agree

print(lay_user_agree, lay_user_disagree, sme_agree, sme_disagree)

contingency_table = np.array([[lay_user_agree, lay_user_disagree],
                              [sme_agree, sme_disagree]])

# Perform the Chi-Square Test of Independence
chi2_stat, p_val_chi2, dof, expected = stats.chi2_contingency(contingency_table)

print(f"Chi-Square Test of Independence")
print(f"Chi2 Statistic: {chi2_stat}")
print(f"P-Value: {p_val_chi2}")
print(f"Degrees of Freedom: {dof}")
print(f"Expected Frequencies: {expected}")