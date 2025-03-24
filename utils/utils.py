from utils.header import * 

from utils.airfoil_datasets import *

def dict2config(config):
    new_config = configparser.ConfigParser()

    for section, options in config.items():
        new_config[section] = options
    
    return new_config
    

class CFDDataSet(Dataset):
    def __init__(self, config, mode='train', load_ddpm_states=False, recursive_sampling=False):
        self.mode = mode
        self.recursive_sampling = recursive_sampling
        self.config_dataset = config['dataset']
        self.config_validation = config['training.validation']
        self.residual_data = config['dataset']['residual_data']
        self.load_ddpm_states = load_ddpm_states
        self.single_file = config['dataset']['single_file']

        if load_ddpm_states:
            if mode == 'train':
                self.suffix = "training"
            elif mode == 'valid':
                self.suffix = "validation"
            elif mode == 'test':
                self.suffix = "testing"
            else:
                raise ValueError("Incorrect Datset Type")

            self.ddpm_states_path = os.path.join(self.config_dataset['ddpm_states_path'], self.suffix)
        
        self.home = self.config_dataset['home']
        self.main_path = config['general']['main_path']
        self.name_prefix = self.config_dataset['name_prefix'].split(',')
        self.size_per_dataset = self.config_dataset['size_per_dataset']
        self.shift = self.config_dataset['shift']
        self.overlap = self.config_dataset['overlap']

        assert self.overlap >= 0
        assert self.overlap < self.size_per_dataset

        self.split = self.config_dataset['split'].split(',')
        self.split = {"train": eval(self.split[0]), "valid": eval(self.split[1]), "test": eval(self.split[2])}
        
        self.seed_values = self.config_dataset['seed_values'].split(',')
        self.seed_values = [int(i) for i in self.seed_values]
        self.seed_begin = self.seed_values[0]
        self.seed_step = self.seed_values[1]
        self.seed_end = self.seed_values[2]

        self.total_num_sims = int((self.seed_end - self.seed_begin)/self.seed_step+1)
        self.current_num_sims = int(self.total_num_sims * self.split[mode])
        self.model_capacity = config['UNet']['model_capacity']
        
        if mode == 'valid':
            self.seed_begin = int(self.total_num_sims * self.split['train']) * self.seed_step + self.seed_begin
        elif mode == 'test':
            self.seed_begin = int(self.total_num_sims * (self.split['train'] + self.split['valid'])) * self.seed_step + self.seed_begin

        self.seed_end = (self.current_num_sims - 1) * self.seed_step + self.seed_begin


        self.n_subsetdatasets_per_sim = self.config_dataset['sims_per_dataset']
        if self.n_subsetdatasets_per_sim == 'auto':
            self.n_subsetdatasets_per_sim = (self.config_dataset['total_frames_per_dataset']-self.overlap) // (self.size_per_dataset - self.overlap)

        self.data_size = self.n_subsetdatasets_per_sim * self.current_num_sims * self.model_capacity
        if not self.config_dataset['diff_fields']:
            self.data_size *= len(self.name_prefix)



        self.velocity_states = []
        self.fluid_params = []
        self.mean_norm = []
        self.std_norm = []
        self.fluid_params_mean = None
        self.fluid_params_std = None
        self.set_normalization_params()

        if self.config_dataset['name'] == 'kolmogorov':
            self.fluid_params = self.config_dataset['reynolds'].split(',')
            self.fluid_params = np.array([(int(param) - self.fluid_params_mean) / self.fluid_params_std for param in self.fluid_params]).astype('float32',casting='same_kind')

        if not self.single_file:
            self.scenes = []
            self.generate_scenes()

            if load_ddpm_states:
                self.ddpm_velocity_states = []
            self.load_data()
        else:
            self.load_single_file()
        
    def set_normalization_params(self):
        if self.config_dataset['name'] == 'tra':
            # ORDER (fields): velocity (x,y), pressure, density
            normMean = np.array([np.array([5.60642565e-01, -1.27761338e-04]), 6.37941355e-01, 9.03352441e-01], dtype=object)
            normStd =  np.array([np.array([0.24365195, 0.16342957]), 0.11995308, 0.14539101], dtype=object)
            
            for i in range(len(self.name_prefix)):
                mean = np.array(normMean[i]).astype('float32',casting='same_kind')
                std = np.array(normStd[i]).astype('float32',casting='same_kind')
                self.mean_norm.append(mean.reshape(1,1,mean.size,1,1))
                self.std_norm.append(std.reshape(1,1,std.size,1,1))
            self.fluid_params_mean = 0.7
            self.fluid_params_std = 0.118322
                
        elif self.config_dataset['name'] == 'kolmogorov':
            for i in range(len(self.name_prefix)):
                self.mean_norm.append(np.zeros((1,1,2,1,1)).astype('float32'))
                self.std_norm.append(np.ones((1,1,2,1,1)).astype('float32'))
            arr = [100, 200, 1000, 1750, 2500, 4000, 5000]
            self.fluid_params_mean = np.mean(arr).astype('float32',casting='same_kind')
            self.fluid_params_std = np.std(arr).astype('float32',casting='same_kind')

    def get_fluid_params_normalization_params(self):
        return self.fluid_params_mean, self.fluid_params_std
    
    def get_normalization_params(self):
        if len(self.mean_norm) > 0:
            return self.mean_norm, self.std_norm
        else:
            return None
    
    def get_mask(self):
        mask_file = self.config_dataset.get('mask_file', None)
        if mask_file is not None:
            return np.load(os.path.join(self.main_path, self.home, self.config_dataset['mask_file'] + '.npy'))[0].astype('float32',casting='same_kind')
        else:
            return None
    
    def load_single_file(self):
        dataset_size = 0
        temporal_stride = self.config_dataset['temporal_stride']
        for i, sim in enumerate(self.name_prefix):
            if self.mode == 'valid':
                sim = sim.replace('train', 'test')
            print(f'Loading file: {sim}.npy')

            read_velocity = np.load(os.path.join(self.main_path, self.home,sim+'.npy')).astype('float32',casting='same_kind')

            if self.mode == 'valid' and not self.config_validation['rec_sampling']:
                self.current_num_sims = 2
                self.data_size = self.model_capacity * self.current_num_sims
                self.n_subsetdatasets_per_sim = 1
                if not self.config_dataset['diff_fields']:
                    self.data_size *=  len(self.name_prefix)
                total_read = read_velocity.shape[0]
                read_velocity = read_velocity[0:total_read:total_read-1, self.shift:self.shift+(self.size_per_dataset*self.n_subsetdatasets_per_sim*temporal_stride):temporal_stride]
            else:
                read_velocity = read_velocity[self.seed_begin:self.seed_end+1:self.seed_step, self.shift:self.shift+(self.size_per_dataset*self.n_subsetdatasets_per_sim*temporal_stride):temporal_stride]

            if i == 0 or not self.config_dataset['diff_fields']:
                dataset_size += read_velocity.shape[0]*(read_velocity.shape[1]-(2*self.residual_data))

            mean = self.mean_norm[i]
            std = self.std_norm[i]
            read_velocity = (read_velocity - mean) / std
            self.velocity_states += [read_velocity]
        
        self.shift = 0
        
        if self.overlap == 0:
            assert dataset_size == self.data_size, f'loaded_data = {dataset_size}, requested_data = {self.data_size}'

        if self.config_dataset['name'] == 'tra':
            mode = 'test' if self.mode == 'valid' else self.mode
            filename = f'128_tra_mach_number_{mode}.npy'
            print(f'Loading file: {filename}')
            self.fluid_params = np.load(os.path.join(self.main_path, self.home,filename)).astype('float32',casting='same_kind')
            self.fluid_params = (self.fluid_params - self.fluid_params_mean) / self.fluid_params_std
            if self.mode == 'valid' and not self.config_validation['rec_sampling']:
                self.fluid_params = np.array(self.fluid_params[0:total_read:total_read-1])
            else:
                self.fluid_params = self.fluid_params[self.seed_begin:self.seed_end+1:self.seed_step]

            assert self.fluid_params.shape[0] == self.velocity_states[0].shape[0], f'{self.fluid_params.shape[0]}, {self.velocity_states[0].shape[0]}'

    
    def load_data(self):
        print('Started loading data into RAM.')
        
        for i in tqdm(range(len(self.scenes))):
            sim_vel_states = []
            scene = self.scenes[i]
            for frame in range(self.size_per_dataset):
                frame = str(frame + self.shift)
                frame_n_digits = len(frame)
                path = scene.path
                initial_str = '000000'
                velocity_frame = initial_str[:-frame_n_digits] + str(frame)

                read_velocity = np.load(f'{path}/velocity_{velocity_frame}.npz')['data'][0:64, 0:64, :].astype('float32',casting='same_kind')
                sim_vel_states.append(torch.tensor(np.moveaxis(read_velocity, -1, 0)))

            self.velocity_states.append(sim_vel_states)

        if self.load_ddpm_states:
            print('Started loading DDPM data into RAM.')

            loaded_ddpm_data = np.load(os.path.join(self.main_path, self.ddpm_states_path, f'predictions_{self.suffix}.npy'))

            for i in tqdm(range(len(self.scenes))):
                sim_pred_vel_states = torch.tensor(loaded_ddpm_data[:,i])
                self.ddpm_velocity_states.append(sim_pred_vel_states)  


    def generate_scenes (self):
        valid_seed = 999
        if self.mode == 'valid':
            for name_prefix in self.name_prefix:
                self.scenes.append(Scene.at(os.path.join(self.main_path, self.home, name_prefix, 'seed' + str(int(valid_seed))), 0))
            self.data_size = self.n_subsetdatasets_per_sim * self.model_capacity * len(self.name_prefix)
            self.current_num_sims = 1
        else:
            current_seed_values = np.linspace(self.seed_begin, self.seed_end, num=self.current_num_sims)
            for name_prefix in self.name_prefix:
                for _, seed in enumerate(current_seed_values):
                    if seed == valid_seed:
                        self.data_size -= self.n_subsetdatasets_per_sim * self.model_capacity
                    else:
                        self.scenes.append(Scene.at(os.path.join(self.main_path, self.home, name_prefix, 'seed' + str(int(seed))), 0))
        

    def __getitem__(self, index):
        if index == -1:
            index = self.data_size - 1

        initial_condition = None
        data_s = []

        index += self.shift

        if self.recursive_sampling:
            total_sims = self.n_subsetdatasets_per_sim * self.current_num_sims
            if not self.config_dataset['diff_fields']:
                total_sims *= len(self.name_prefix)
            index = index % (total_sims) * self.model_capacity + index//total_sims
            
        dataset_idx = (index-self.shift) // (self.model_capacity * self.n_subsetdatasets_per_sim)
        if not self.config_dataset['diff_fields']:
            dataset_idx = dataset_idx % self.current_num_sims
        subdataset_idx = (index % (self.model_capacity * self.n_subsetdatasets_per_sim)) // self.model_capacity
        init_cond_frame_subdataset = subdataset_idx*(self.model_capacity - self.overlap) + (1*self.residual_data)


        if 'reynolds' in self.config_dataset.keys() and self.config_dataset['name'] == 'kolmogorov':
            loaded_file_idx = (index-self.shift) // (self.model_capacity * self.n_subsetdatasets_per_sim) // self.current_num_sims
            fluid_param_idx = loaded_file_idx
        else:
            fluid_param_idx = dataset_idx

        fluid_param = self.fluid_params[fluid_param_idx]


        t_label = int((index - self.shift) % self.model_capacity)
        assert t_label >= 0, f'{t_label}, {t_label}'
       
        frame = t_label + (1*self.residual_data) + subdataset_idx*(self.model_capacity - self.overlap)

        if self.residual_data:
            assert frame > 0, frame
        else:
            assert frame >= 0, frame

        dataset_idx_ddpm = (index-1) // self.model_capacity
        frame_ddpm = (index-1) % self.model_capacity
        
        if self.residual_data:
            t_labels = [t_label-1, t_label, t_label+1]
            frames = [frame-1, frame, frame+1]
        else:   
            t_labels = [t_label]
            frames = [frame]

        if not self.single_file:
            initial_condition = torch.tensor(self.velocity_states[dataset_idx][init_cond_frame_subdataset])
        else:
            if self.config_dataset['name'] == 'tra':
                initial_condition = []
                for field_idx in range(len(self.velocity_states)):
                    initial_condition.append(torch.tensor(self.velocity_states[field_idx][dataset_idx, init_cond_frame_subdataset]))
                initial_condition = torch.cat(initial_condition, dim=0)
            else:
                initial_condition = torch.tensor(self.velocity_states[loaded_file_idx][dataset_idx, init_cond_frame_subdataset])

        for _, frame in enumerate(frames):
            if not self.single_file:
                data = self.velocity_states[dataset_idx][frame]
            else:
                if self.config_dataset['name'] == 'tra':
                    data = []
                    for field_idx in range(len(self.velocity_states)):
                        data.append(torch.tensor(self.velocity_states[field_idx][dataset_idx, frame]))
                    data = torch.cat(data, dim=0)
                else:
                    data = self.velocity_states[loaded_file_idx][dataset_idx, frame]
                
            data_s.append(torch.tensor(data))
        
        if self.load_ddpm_states:
            data_s.append(torch.tensor(self.ddpm_velocity_states[dataset_idx_ddpm][frame_ddpm].float()))
        
        if not self.residual_data:
            return initial_condition,data_s[0], t_labels[0], fluid_param
        
        return initial_condition, data_s, t_labels, fluid_param

    def __len__(self):
        return self.data_size

def prepare_airfoil_dataset_files (N, mode, home):
    if mode == 'train':
        path_to_txt_lists = f"{home}/cases_lists_{N}/train/"
        path_to_dataset_files = f"{home}/train/"
    elif mode == 'test_int':
        path_to_txt_lists = f"{home}/cases_lists_{N}/int/"
        path_to_dataset_files = f"{home}/test/interpolation/"    
    elif mode == 'test_ext':
        path_to_txt_lists = f"{home}/cases_lists_{N}/ext/"
        path_to_dataset_files = f"{home}/test/extrapolation/"
    else:  
        raise NotImplementedError("No dataset available for this name")
    
    txt_lists_exist = False

    if os.path.isdir(path_to_txt_lists):
        # shutil.rmtree(path_to_txt_lists)
        txt_lists_exist = True

    cases = os.listdir(path_to_dataset_files)
    selected_training_cases  = np.zeros((N, 25), dtype=object)

    if not txt_lists_exist:
        for i, case in enumerate(cases):
            if i == N:
                break
            folder_df=FolderDataFiles(f"{path_to_dataset_files}{case}")
            for j, case_i in enumerate(folder_df):
                selected_training_cases[i, j] = case_i
            airfoil = case_i['path'].split('/')[-1]
            save_case_list_to_filedataFiles(selected_training_cases[i],f"{path_to_txt_lists}{airfoil}.txt")

    df_files = None
    cases_lists = os.listdir(path_to_txt_lists)
    all_len = len(cases_lists)
    assert all_len == N, f'{all_len = }, {N = }'

    for i, case in enumerate(cases_lists):
        if i == N:
            break

        if 'test' in mode:
            case_name = '.'.join(case.split('.')[:-1])
        else:
            case_name = case.split('.')[0]
        path = f'{path_to_dataset_files}{case_name}/'

        df=FileDataFiles(f"{path_to_txt_lists}{case}", base_path=path)
        if i == 0:
            df_files = df.case_list
        else:
            df_files += df.case_list
    
    return df_files

def get_data(config, mode='train', bsz=None, load_ddpm_states=False, recursive_sampling=False):
    shuffle = config['dataset']['shuffle']
            
    if config['dataset']['name'] == 'airfoil':
        main_path = config['general']['main_path']
        home = config['dataset']['home']
        home = f'{main_path}/{home}'
        batch_size = bsz if bsz is not None else config['training']['batch_size']


        if mode == 'train':

            if config['dataset'].get('multi', False):
                N = config['dataset']['size_per_dataset']
                
                df_files = prepare_airfoil_dataset_files(N, 'train', home)

                df=DataFiles(df_files)
            else:
                df=FileDataFiles(f"{home}/train_cases.txt",base_path=f"{home}/data/")

        else:
            batch_size = bsz if bsz is not None else config['sampling']['batch_size']

            shuffle = False
            if config['dataset'].get('multi', False):
                if mode == 'test_int':
                    N = 100
                    df_files = prepare_airfoil_dataset_files(N, mode, home)
                elif mode == 'test_ext':
                    N = 30
                    df_files = prepare_airfoil_dataset_files(N, mode, home)
                else:
                    raise NotImplementedError()

                df=DataFiles(df_files)
            else:
                df=FileDataFiles(f"{home}/test_cases.txt",base_path=f"{home}/data/")

        dataset=AirfoilDataset(df, data_size=config['general']['grid_size'])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataset, dataloader

    dataset = CFDDataSet(config, mode, load_ddpm_states, recursive_sampling)


    if mode == 'train':
        batch_size = config['training']['batch_size']
    elif mode =='valid':
        batch_size = config['training.validation']['batch_size']
        shuffle=False
    elif mode =='test':
        batch_size = 1
        shuffle=False

    if bsz is not None:
        batch_size = bsz

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataset, dataloader


def setup_logging(config):
    run_name = config['general']['run_name']
    sampling = config['sampling']['output_folder']

    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join(sampling, run_name), exist_ok=True)


def convert_value(value):
    try:
        return int(value)
    except ValueError:
        pass
    
    try:
        return float(value)
    except ValueError:
        pass
    
    if value.lower() in ['true', 'yes', 'on']:
        return True
    elif value.lower() in ['false', 'no', 'off']:
        return False
    
    return value

def get_configs (filename):
    config_file = configparser.ConfigParser()
    print(f"\n*Config file selected: {filename}\n")
    config_file.read(filename)

    config = {}

    for section in config_file.sections():
        section_dict = {}
        
        for option in config_file.options(section):
            value = config_file.get(section, option)
            section_dict[option] = convert_value(value)

        config[section] = section_dict

    return config