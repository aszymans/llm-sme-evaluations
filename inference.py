from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import pandas as pd
import tqdm
from openai import OpenAI

client = OpenAI()

def generate(instruction, model="gpt-4o"):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": 'Limit your response to 200 words. The output should paragraph style with multiple paragraphs. Dont use any markdown or special formatting. \n' + instruction}
        ]
    )
    return completion.choices[0].message.content

def main():
    df = pd.read_excel('sme-spreadsheet-1.xlsx')
    prompts = df['Prompt'].dropna()
    for model in ['gpt-4o', 'gpt-3.5-turbo', 'gpt-4-turbo']:
        for index, p in tqdm.tqdm(prompts.items(), total=prompts.size, desc="Processing Prompts"):  # Changed iteritems() to items()
            df.at[index, f'{model}_Output'] = generate(p, model=model)
    df.to_excel('sme-spreadsheet-2.xlsx')


if __name__ == "__main__":
    main()