import os
import pickle
from src.data_management.DataObj import DataObj
from src.core.ExpProcessor import ExpProcessor
import src.models.model_functions as mf

list_id = ['06','08']
model_values = {}

current_path = os.getcwd()
if 'src' in current_path:
    current_path = os.path.join(current_path, '..', '..')

for id_num in list_id:
    print(f'Participant {id_num}')
    print('---')
    xdf_name = os.path.join(current_path, 'data', f'participant_{id_num}', 'S01', '01.xdf')
    subject_id = xdf_name.split('\\')[-2]
    data = DataObj(xdf_name)
    session_folder = os.path.dirname(xdf_name)
    if f'processor_{id_num}.pkl' not in os.listdir(session_folder):

        # Initialize ExpProcessor
        processor = ExpProcessor(
            emg_data=data.ElectrodeStream,
            trigger_stream=data.Trigger_Cog,
            fs=250,
            window_size=30.0,
            overlap=0.5,
            subject_id=data.subject_id,  # Pass the subject ID
            sorted_indices=data.sorted_indices,  # Pass the sorted indices,
            auto_process=False,
            path=xdf_name
        )

        # Save the object to a file with pickle and subject
        with open(f'{session_folder}\processor_{id_num}.pkl', 'wb') as f:
            pickle.dump(processor, f)

    else:
        # Load the object from the file
        with open(f'{session_folder}\processor_{id_num}.pkl', 'rb') as f:
            processor = pickle.load(f)
        f.close()

    final_model, selected_features, performance_dict, top_feature_names, top_correlations = mf.run_pipeline(processor, data, session_folder)
    model_values[id_num] = {'model': final_model, 'selected_features': selected_features, 'performance_dict': performance_dict,
                            'top_20_features': top_feature_names, 'top_correlations': top_correlations}


all_sets = [set(model_values[pid]['selected_features']) for pid in list_id]
common = set.intersection(*all_sets)
print(f'Common features: {common}')


all_sets = [set(model_values[pid]['top_features']) for pid in list_id]
