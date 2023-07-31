import os
import subprocess
from time import time
import shutil
import yaml


base_dir = 'models/factorization/2^16/'

sec_thresh = 2 * 7 * 24 * 60 * 60

curr_time = time()

for f in os.listdir(base_dir):
    path = base_dir + f + '/'
    print(path)
    if os.path.isdir(path):
        check_files = ['loss_hist.csv', 'metrics_oos.json', 'metrics_test.json', 'pred_df_oos.csv', 'pred_df_test.csv']
        run = False
        if all([os.path.exists(path + check_f) for check_f in check_files]):
            check_file = path + check_files[0]
            created_at = os.path.getctime(check_file)
            if (curr_time - created_at) > sec_thresh and not path.endswith('_BACKUP/'):
                print('RUN')
                run = True
            else:
                print(f'SKIP')
                run = False
        if run:
            move_to = path[:-1] + '_BACKUP/'
            # move_to_for_run = move_to.replace('^', '\^')
            to_run = f"python scripts\\train_model.py --config {move_to + 'config.yaml'}"
            shutil.move(path, move_to)
            with open(move_to + 'config.yaml', 'r') as f2:
                config_args = yaml.safe_load(f2)
                config_args['io']['save_path'] = config_args['io']['save_path'].replace('models/factorization/', 'models/factorization/2^16/')
            with open(move_to + 'config.yaml', 'w') as f2:
                yaml.dump(config_args, f2)
            subprocess.call(to_run)
            # print(move_to)
            # print(to_run)
            # print(f'finished running {path}')

        
    # sys.exit()
    print('='*100)