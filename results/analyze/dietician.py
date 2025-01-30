import yaml

def load_overall_annoations(path):
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

    len(grouped_annotations[0])
    llm_aspect_annotations = grouped_annotations
    return llm_aspect_annotations