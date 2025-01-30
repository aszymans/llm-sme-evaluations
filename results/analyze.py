import analyze.utils as utils
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def load_dietician_paths():
    return {
        "sme_survey_annotations": "./sme-eval/dietician/dietitian_updated_spreadsheet_09.22.24.csv",
        "expert_llm_overall_annotations": "llm-pairwise-eval/dietician/weighted_alpaca_eval_gpt4_turbo/expert_overall_explanations.yaml",
        "expert_llm_aspect_annotations": "llm-pairwise-eval/dietician/weighted_alpaca_eval_gpt4_turbo/expert_aspect_explanations.yaml",
        "standard_llm_overall_annotations": "llm-pairwise-eval/dietician/weighted_alpaca_eval_gpt4_turbo/standard_overall_explanations.yaml",
        "standard_llm_aspect_annotations": "llm-pairwise-eval/dietician/weighted_alpaca_eval_gpt4_turbo/standard_aspect_explanations.yaml",
        "llm_orders": "sme-eval/dietician/llm_orders.txt",
        "question_grouping": "sme-eval/dietician/question_grouping.csv",
        "lay_user_survey_annotations": "layuser-eval/dietician/lay_user_dietetics_results_9.26.csv",
    }

def load_mental_health_paths():
    return {
        "sme_survey_annotations": "sme-eval/mental_health/mental_health_domain_result_10-7.csv",
        "expert_llm_overall_annotations": "llm-pairwise-eval/mental_health/weighted_alpaca_eval_gpt4_turbo/expert_overall_explanations.yaml",
        "expert_llm_aspect_annotations": "llm-pairwise-eval/mental_health/weighted_alpaca_eval_gpt4_turbo/expert_aspect_explanations.yaml",
        "standard_llm_overall_annotations": "llm-pairwise-eval/mental_health/weighted_alpaca_eval_gpt4_turbo/standard_overall_explanations.yaml",
        "standard_llm_aspect_annotations": "llm-pairwise-eval/mental_health/weighted_alpaca_eval_gpt4_turbo/standard_aspect_explanations.yaml",
        "llm_orders": "sme-eval/mental_health/llm_orders.txt", 
        "question_grouping": "sme-eval/mental_health/question_grouping.csv",
        "lay_user_survey_annotations": "layuser-eval/mental_health/lay_user_mental_health_results_9.26.csv"
    }

def analyze_overall_agreement(merged_data, llm_orders, baseline):
    if baseline not in ["lay_user", "sme"]:
        raise ValueError("baseline must be either 'lay_user' or 'sme'")
    n_agree = 0
    sme_agreement_percentages = []
    for d, order in zip(merged_data, llm_orders):
        sme_overall_preferences = {
            'GPT-4o': d['subquestions']['a'][f'{baseline}_responses'].count('Response A') if order[1] == 'a4.0' else d['subquestions']['a'][f'{baseline}_responses'].count('Response B'),
            'GPT-3.5': d['subquestions']['a'][f'{baseline}_responses'].count('Response B') if order[1] == 'a4.0' else d['subquestions']['a'][f'{baseline}_responses'].count('Response A')
        }
        sme_overall_preference = max(sme_overall_preferences, key=sme_overall_preferences.get)
        percent_sme_agree = sme_overall_preferences[sme_overall_preference] / (sme_overall_preferences['GPT-4o'] + sme_overall_preferences['GPT-3.5'])
        sme_agreement_percentages.append(percent_sme_agree)
        llm_overall_preference = d['llm_preference']
        agree = sme_overall_preference == llm_overall_preference
        # print(f'{d["instruction"][:100]}...')
        # print(f'Q{d["qualtrics_idx"]} SME: {sme_overall_preference}({percent_sme_agree:.0%}) LLM: {llm_overall_preference} AGREE: {agree}')
        n_agree += 1 if agree else 0
    baseline_title = "Lay User" if baseline == "lay_user" else "SME"
    print(f'% AGREEMENT BETWEEN {baseline_title} AND LLM: {n_agree/len(merged_data):.2%}')
    return sme_agreement_percentages

def analyze_aspect_agreement(merged_data, llm_orders, baseline):
    if baseline not in ["lay_user", "sme"]:
        raise ValueError("baseline must be either 'lay_user' or 'sme'")
    n_agree = 0
    n_total = 0

    question_grouping_results = {}
    theme_grouping_results = {}

    sme_agreement_percentages = []
    for d, order in zip(merged_data, llm_orders):
        for sub_q in ['b', 'c', 'd']:
            if sub_q not in d['subquestions']:
                continue
            sme_aspect_preferences = {
                'GPT-4o': d['subquestions'][sub_q][f'{baseline}_responses'].count('Response A') if order[1] == 'a4.0' else d['subquestions'][sub_q][f'{baseline}_responses'].count('Response B'),
                'GPT-3.5': d['subquestions'][sub_q][f'{baseline}_responses'].count('Response B') if order[1] == 'a4.0' else d['subquestions'][sub_q][f'{baseline}_responses'].count('Response A')
            }
            sme_aspect_preference = max(sme_aspect_preferences, key=sme_aspect_preferences.get)
            percent_sme_agree = sme_aspect_preferences[sme_aspect_preference] / (sme_aspect_preferences['GPT-4o'] + sme_aspect_preferences['GPT-3.5'])
            sme_agreement_percentages.append(percent_sme_agree)
            llm_aspect_preference = d['llm_preference']
            agree = sme_aspect_preference == llm_aspect_preference

            if d['subquestions'][sub_q]['question_grouping'] not in question_grouping_results:
                question_grouping_results[d['subquestions'][sub_q]['question_grouping']] = {'agree': 0, 'total': 0}
            if d['subquestions'][sub_q]['theme'] not in theme_grouping_results:
                theme_grouping_results[d['subquestions'][sub_q]['theme']] = {'agree': 0, 'total': 0}

            question_grouping_results[d['subquestions'][sub_q]['question_grouping']]['agree'] += 1 if agree else 0
            question_grouping_results[d['subquestions'][sub_q]['question_grouping']]['total'] += 1

            theme_grouping_results[d['subquestions'][sub_q]['theme']]['agree'] += 1 if agree else 0
            theme_grouping_results[d['subquestions'][sub_q]['theme']]['total'] += 1

            # print(f'Q{d["qualtrics_idx"]} {sub_q} SME: {sme_aspect_preference}({percent_sme_agree:.0%}) LLM: {llm_aspect_preference} AGREE: {agree}')
            n_agree += 1 if agree else 0
            n_total += 1
    print(f'% ASPECT AGREEMENT BETWEEN SME AND LLM: {n_agree/n_total:.2%}')

    # print('Question Grouping Results:')
    # for k, v in question_grouping_results.items():
    #     print(f'{k}: {v["agree"] / v["total"]:.2%}')
    # print('-'*50)
    print('Theme Grouping Results:')
    for k, v in theme_grouping_results.items():
        print(f'{k}: {v["agree"] / v["total"]:.2%}')
    print('-'*50)
    return sme_agreement_percentages

def plot_agreement_percentages(sme_agreement_percentages):
    plt.figure(figsize=(10, 6))
    plt.hist(sme_agreement_percentages, bins=20, edgecolor='black', range=(0, 1))
    plt.title('Histogram of SME Agreement Percentages')
    plt.xlabel('Agreement Percentage')
    plt.ylabel('Frequency')
    plt.axvline(sum(sme_agreement_percentages) / len(sme_agreement_percentages), color='red', linestyle='dashed', linewidth=2, label='Mean')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.xticks([i/10 for i in range(11)], [f'{i*10}%' for i in range(11)])
    plt.show()

    print(f"Mean SME agreement percentage: {sum(sme_agreement_percentages) / len(sme_agreement_percentages):.2%}")
    print(f"Median SME agreement percentage: {sorted(sme_agreement_percentages)[len(sme_agreement_percentages)//2]:.2%}")
    
def analyze_intraclass_correlation(merged_data, llm_orders, path):
    questions = []
    judges = []
    agreement = []

    for d, order in zip(merged_data, llm_orders):
        question_id = d['qualtrics_idx']
        llm_preference = d['llm_preference']
        for sme_response_idx, sme_response in enumerate(d['subquestions']['a']['sme_responses']):
            questions.append(f'Q{question_id}')
            judges.append(f'SME{sme_response_idx}')
            sme_model_pref = 'GPT-4o' if (order[1] == 'a4.0' and sme_response == 'Response A') or (order[1] == 'a3.5' and sme_response == 'Response B') else 'GPT-3.5'
            agreement.append(1 if sme_model_pref == llm_preference else 0)
        for lay_user_response_idx, lay_user_response in enumerate(d['subquestions']['a']['lay_user_responses']):
            questions.append(f'Q{question_id}')
            judges.append(f'LayUser{lay_user_response_idx}')
            lay_user_model_pref = 'GPT-4o' if (order[1] == 'a4.0' and lay_user_response == 'Response A') or (order[1] == 'a3.5' and lay_user_response == 'Response B') else 'GPT-3.5'
            agreement.append(1 if lay_user_model_pref == llm_preference else 0)

    df = pd.DataFrame({
        'question': questions,
        'judge': judges,
        'agreement': agreement
    })
    df.to_csv(path, index=False)
    # icc = pg.intraclass_corr(data=df, targets='question', raters='judge', ratings='agreement')
    

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--subject",
        type=str,
        required=True,
        choices=["dietician", "mental_health"],
        help="Subject of the survey results"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        choices=["sme", "lay_user"],
        help="Baseline to compare LLM preferences to"
    )
    parser.add_argument(
        "--persona",
        type=str,
        required=True,
        choices=["expert", "standard"],
        help="Persona of the LLM to compare to the baseline"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(args)

    paths = {
        "dietician": load_dietician_paths(),
        "mental_health": load_mental_health_paths()
    }[args.subject]
    
    sme_survey_annotations = utils.load_survey_annotations(paths["sme_survey_annotations"])
    lay_user_survey_annotations = utils.load_survey_annotations(paths["lay_user_survey_annotations"])
    llm_overall_annotations = utils.load_llm_overall_annotations(paths[f"{args.persona}_llm_overall_annotations"])
    llm_aspect_annotations = utils.load_llm_aspect_annotations(paths[f"{args.persona}_llm_aspect_annotations"])
    question_grouping = utils.load_question_grouping(paths["question_grouping"])

    merged_data = utils.merge_data(sme_survey_annotations, lay_user_survey_annotations, llm_overall_annotations, llm_aspect_annotations, question_grouping)
    llm_orders = utils.load_llm_orders(paths["llm_orders"])

    # analyze_intraclass_correlation(merged_data, llm_orders, f"agreement-tables-{args.subject}.csv")

    sme_agreement_percentages = analyze_overall_agreement(merged_data, llm_orders, baseline=args.baseline)
    print(sme_agreement_percentages)
    # Calculate mean and standard deviation of agreement percentages
    mean_agreement = sum(sme_agreement_percentages) / len(sme_agreement_percentages)
    std_dev_agreement = (sum((x - mean_agreement) ** 2 for x in sme_agreement_percentages) / len(sme_agreement_percentages)) ** 0.5

    print(f"Mean agreement percentage: {mean_agreement:.2f}")
    print(f"Standard deviation of agreement percentages: {std_dev_agreement:.2f}")
    # plot_agreement_percentages(sme_agreement_percentages)

    print("Aspects:")
    analyze_aspect_agreement(merged_data, llm_orders, baseline=args.baseline)


if __name__ == "__main__":
    main()

#[0.6, 0.5, 0.6, 0.6, 0.8, 0.9, 0.8, 0.9, 0.9, 0.6, 0.7, 0.7, 0.7, 0.5, 0.8, 0.7, 0.9, 0.5, 0.7, 0.7, 0.8, 0.9, 0.7, 0.8, 0.7, 0.7, 0.7, 0.6, 0.8, 0.9, 1.0, 0.7, 0.8, 0.9, 1.0, 0.8, 0.7, 0.5, 0.6, 0.7, 0.6, 0.8, 0.6, 0.8, 0.9, 0.6, 0.9, 0.8, 0.6, 0.7]