import pandas as pd

# Load the CSV file
file_path = '/Users/annalisaszymanski/qualtrics_survey/annalisa/results/sme-eval/dietician/dietitian_updated_spreadsheet_09.22.24.csv'  # Replace with your actual file path
df = pd.read_csv(file_path, header=None)

# Combine the first and second rows into one header
new_header = df.iloc[0] + '_' + df.iloc[1].fillna('')
df.columns = new_header

# Drop the first three rows (original headers and the 3rd row)
df = df.drop([0, 1, 2])

# Drop the 'counter' column if it exists
if 'counter_' in df.columns:
    df = df.drop(columns=['counter_'])

# Reset index after dropping rows
df.reset_index(drop=True, inplace=True)

# Drop columns that start with 'PreQuestion'
df = df.loc[:, ~df.columns.str.startswith('PreQuestion')]

# Define the constant columns that should remain the same
constant_columns = [
    'StartDate_Start Date', 'EndDate_End Date', 'Status_Response Type', 'IPAddress_IP Address',
    'Progress_Progress', 'Duration (in seconds)_Duration (in seconds)', 'Finished_Finished',
    'RecordedDate_Recorded Date', 'ResponseId_Response ID', 'RecipientLastName_Recipient Last Name',
    'RecipientFirstName_Recipient First Name', 'RecipientEmail_Recipient Email',
    'ExternalReference_External Data Reference', 'LocationLatitude_Location Latitude',
    'LocationLongitude_Location Longitude', 'DistributionChannel_Distribution Channel',
    'UserLanguage_User Language', 'ConsentSign_By signing below, I confirm that I am 18 years old and a registered dietitian, and agree to take part in this study.'
]

# Separate Q and T columns
q_columns = [col for col in df.columns if col.startswith('Q')]
t_columns = [col for col in df.columns if col.startswith('T')]

# Pivoting the Q columns
q_pivot = df.melt(id_vars=constant_columns, value_vars=q_columns, var_name='QVariable', value_name='QValue')

# Pivoting the T columns
t_pivot = df.melt(id_vars=constant_columns, value_vars=t_columns, var_name='TVariable', value_name='TValue')

# Concatenate the Q and T pivoted dataframes side by side
combined_df = pd.concat([q_pivot, t_pivot[['TVariable', 'TValue']]], axis=1)

# Save the result to a new CSV file
combined_df.to_csv('dietitian_data_to_code.csv', index=False)

print("Pivoted data saved as 'pivoted_data_with_QT.csv'")
