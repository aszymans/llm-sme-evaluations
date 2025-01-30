import pandas as pd
import pingouin as pg

df = pd.read_csv('/Users/annalisaszymanski/qualtrics_survey/annalisa/results/agreement-tables-dietician.csv')

# Calculate ICC
icc_result = pg.intraclass_corr(data=df, targets='question', raters='judge', ratings='agreement')

print(icc_result)