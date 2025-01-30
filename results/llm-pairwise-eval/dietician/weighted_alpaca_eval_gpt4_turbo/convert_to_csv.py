import yaml
import csv

# Load YAML file
with open('/Users/annalisaszymanski/qualtrics_survey/annalisa/results/llm-pairwise-eval/dietician/weighted_alpaca_eval_gpt4_turbo/aspect_explanations.yaml', 'r') as yaml_file:
    yaml_data = yaml.safe_load(yaml_file)

# Define the output CSV file
csv_file = '../aspect_evaluations_dietitians.csv'

# Open the CSV file for writing
with open(csv_file, 'w', newline='') as file:
    csv_writer = csv.writer(file)

    # If the YAML data is a list of dictionaries
    if isinstance(yaml_data, list):
        # Extract headers (keys from the first dictionary)
        header = yaml_data[0].keys()
        csv_writer.writerow(header)

        # Write each row of data
        for row in yaml_data:
            csv_writer.writerow(row.values())

print(f"YAML data has been converted to {csv_file}")
