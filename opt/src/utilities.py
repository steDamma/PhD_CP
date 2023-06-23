import pickle
import os
import pandas as pd

def pd_get_groups(grouped, idxs, istance):
    filtered_groups = [group for i, (_, group) in enumerate(grouped) if i in idxs]
    combined_df = pd.concat(filtered_groups)#, ignore_index=True)
    return combined_df.groupby(istance)

def store_res(config, results):
    project_folder = config.project.name
    if not os.path.exists(project_folder):
        os.makedirs(project_folder)

    file_path = os.path.join(project_folder, f'results_loo_criterion_{config.project.loo_criterion}_target_{config.project.target.aim}_clf_{config.classifier.name}.pickle')
    with open(file_path, 'wb') as file:
        pickle.dump(results, file)