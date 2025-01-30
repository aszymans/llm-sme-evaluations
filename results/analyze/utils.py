import yaml
import json
import pandas as pd

def load_survey_annotations(path):
  df = pd.read_csv(path)
  return df

def load_llm_overall_annotations(path):
    llm_overall_annotations = yaml.load(open(path), Loader=yaml.FullLoader)
    return llm_overall_annotations

def load_llm_aspect_annotations(path):
    llm_aspect_annotations = []
    with open(path) as f:
        llm_aspect_annotations = yaml.load(f, Loader=yaml.FullLoader)
    grouped_annotations = []
    for annotation in llm_aspect_annotations:
        instruction = annotation['instruction']
        matching_group = next((group for group in grouped_annotations if group and group[0]['instruction'] == instruction), None)
        if matching_group:
            matching_group.append(annotation)
        else:
            grouped_annotations.append([annotation])

    return grouped_annotations


def merge_data(sme_survey_df, lay_user_survey_df, llm_overall_annotations, llm_aspect_annotations, question_grouping, save_path=None):
    merged_data = []

    for idx, (llm_d, llm_aspect_d) in enumerate(zip(llm_overall_annotations, llm_aspect_annotations)):
        qualtrics_idx = idx + 1
        out = {
            'qualtrics_idx': qualtrics_idx,
            'subquestions': {} 
        }
        for sub_q_idx, sub_q in enumerate(['a', 'b', 'c', 'd']):
            if f'Q{qualtrics_idx}_{sub_q}' in sme_survey_df.columns:
                out['subquestions'][sub_q] = {
                    'question': sme_survey_df[f'Q{qualtrics_idx}_{sub_q}'][0],
                    'sme_responses': list(sme_survey_df[f'Q{qualtrics_idx}_{sub_q}'][2:]),
                    'lay_user_responses': list(lay_user_survey_df[f'Q{qualtrics_idx}_{sub_q}'][2:]),
                    'llm_response': '' if sub_q == 'a' else 'GPT-4o' if llm_aspect_d[sub_q_idx-1]['preference'] < 1.5 else 'GPT-3.5',
                    'question_grouping': question_grouping[sme_survey_df[f'Q{qualtrics_idx}_{sub_q}'][0]]['question_grouping'],
                    'theme': question_grouping[sme_survey_df[f'Q{qualtrics_idx}_{sub_q}'][0]]['theme'],
                }
            if f'T{qualtrics_idx}_{sub_q}' in sme_survey_df.columns:
                out['subquestions'][sub_q]['explanation_question'] = sme_survey_df[f'T{qualtrics_idx}_{sub_q}'][0]
                out['subquestions'][sub_q]['explanation_sme_responses'] = list(sme_survey_df[f'T{qualtrics_idx}_{sub_q}'][2:])
                out['subquestions'][sub_q]['explanation_lay_user_responses'] = list(lay_user_survey_df[f'T{qualtrics_idx}_{sub_q}'][2:])
                out['subquestions'][sub_q]['explanation_llm_response'] = llm_aspect_d[sub_q_idx-1]['explanation']
        out['llm_preference'] = 'GPT-4o' if llm_d['preference'] < 1.5 else 'GPT-3.5'
        out['llm_overall_explanation'] = llm_d['explanation']
        out['instruction'] = llm_d['instruction']
        out['gpt4o_output'] = llm_d['output_1']
        out['gpt3.5_output'] = llm_d['output_2']
        merged_data.append(out)
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(merged_data, f, indent=2)
    return merged_data

def load_llm_orders(path):
    llm_orders = [l.strip().split('_') for l in open(path).readlines()]
    return llm_orders

def load_question_grouping(path):
    question_grouping = pd.read_csv(path)
    question_grouping_dict = {}
    for _, row in question_grouping.iterrows():
        question_grouping_dict[row['Question']] = {
            'question_grouping': row['Question Grouping'],
            'theme': row['Themes']
        }
    return question_grouping_dict

def make_agreement_tables(merged_data, llm_orders, save_path):
    questions = []
    judges = []
    agreement = []

    for d, order in zip(merged_data, llm_orders):
        question_id = d['qualtrics_idx']
        for sme_response_idx, sme_response in enumerate(d['subquestions']['a']['sme_responses']):
            questions.append(f'Q{question_id}')
            judges.append(f'SME{sme_response_idx}')
            sme_model_pref = 'gpt-4o' if (order[1] == 'a4.0' and sme_response == 'Response A') or (order[1] == 'a3.5' and sme_response == 'Response B') else 'gpt-3.5'
            agreement.append(sme_model_pref)
        for lay_user_response_idx, lay_user_response in enumerate(d['subquestions']['a']['lay_user_responses']):
            questions.append(f'Q{question_id}')
            judges.append(f'LayUser{lay_user_response_idx}')
            lay_user_model_pref = 'gpt-4o' if (order[1] == 'a4.0' and lay_user_response == 'Response A') or (order[1] == 'a3.5' and lay_user_response == 'Response B') else 'gpt-3.5'
            agreement.append(lay_user_model_pref)

    df = pd.DataFrame({
        'question': questions,
        'judge': judges,
        'agreement': agreement
    })
    df.to_csv(save_path, index=False)