import statsmodels.api as sm

# Number of agreements (alignments) for SMEs and Lay users
alignments_sme = 170  # 68% agreement for SMEs
alignments_lay_user = 190  # 76% agreement for Lay users

# Total number of evaluations for SMEs and Lay users
total_sme = 250  # 10 SMEs evaluating 25 questions each
total_lay_user = 250  # 10 Lay users evaluating 25 questions each

# Perform the z-test for two proportions
count = [alignments_sme, alignments_lay_user]
nobs = [total_sme, total_lay_user]

z_stat, p_value = sm.stats.proportions_ztest(count, nobs)

print(f"Z-statistic: {z_stat}")
print(f"P-value: {p_value}")

if p_value < 0.05:
    print("The difference in alignment rates is statistically significant.")
else:
    print("The difference in alignment rates is not statistically significant.")
