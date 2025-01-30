from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import json
import yaml
from tqdm import tqdm
from openai import OpenAI
from argparse import ArgumentParser

client = OpenAI()

def generate(messages, model="gpt-4o"):
    completion = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return completion.choices[0].message.content

def main(prompt_template_path, annotations_path, output_path, sme_title):
    new_data = []
    with open(prompt_template_path, 'r') as f:
        prompt_template = f.read()
    with open(annotations_path, 'r') as f:
        data = json.load(f)
        for d in tqdm(data):
            prompt = prompt_template.replace('{instruction}', d['instruction']).replace('{output_a}', d['output_1']).replace('{output_b}', d['output_2']).replace('{sme_title}', sme_title)
            if 'aspect_question' in prompt:
                prompt = prompt.replace('{aspect_question}', d['aspect_question'])
            preference = "Output A" if d['preference'] < 1.5 else "Output B"
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": preference},
                {"role": "user", "content": f"Why is {preference} chosen over the other output? Provide a short 2-3 sentence explanation."}
            ]
            explanation = generate(messages)

            d['explanation'] = explanation
            new_d = d.copy()
            del new_d['raw_completion']
            new_data.append(new_d)

    # Convert the data to YAML format
    yaml_data = yaml.dump(data, default_flow_style=False)

    # Save the YAML data to a file
    with open(output_path, 'w') as f:
        f.write(yaml_data)

    print(f"Data has been saved as YAML in {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--type', type=str, required=True, choices=['aspect', 'overall'])
    parser.add_argument('--sme', type=str, required=True, choices=['dietician', 'mental_health'])
    parser.add_argument('--annotations_path', type=str, required=True)
    args = parser.parse_args()

    prompt_template_path, output_path = {
        'aspect': ('aspect_expl_prompt.txt', 'aspect_explanations.yaml'),
        'overall': ('overall_expl_prompt.txt', 'overall_explanations.yaml')
    }[args.type]

    sme_title = {
        'dietician': 'a registered dietitian',
        'mental_health': 'a mental health expert'
    }[args.sme]

    main(prompt_template_path, args.annotations_path, output_path, sme_title)