import numpy as np
import statsmodels.api as sm

# Based on the results provided, fill in the counts for these categories
both_agree = 17  # Both Lay User and SME agree with LLM
lay_agree_sme_disagree = 2  # Lay User agrees, SME disagrees
sme_agree_lay_disagree = 4  # SME agrees, Lay User disagrees
both_disagree = 2  # Both Lay User and SME disagree with LLM

# McNemar's test uses only the discordant pairs
contingency_table = np.array([[both_agree, lay_agree_sme_disagree],
                              [sme_agree_lay_disagree, both_disagree]])

# Perform McNemar's Test
result = sm.stats.mcnemar(contingency_table, exact=False)

print(f"McNemar's Test")
print(f"Chi2 Statistic: {result.statistic}")
print(f"P-Value: {result.pvalue}")
