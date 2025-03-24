from utils.header import *

import matplotlib.pyplot as plt


def calc_uncertainty(simulations):
    res = []
    mus = []
    sigmas = []

    for key in simulations.keys():
        try:
            vel = float(key)
            res.append(vel * 1e5)
        except ValueError:
            res.append(key)
        data = simulations[key]
        sigmas.append(np.std(data, axis=0))
        mus.append(np.mean(data, axis=0))
    mus = np.stack(mus, axis=0)
    sigmas = np.stack(sigmas, axis=0)
    return res, mus, sigmas


def uncertainty_sampling(config, diffusion, model, dataset, dataset_name, ground_truth=None, sims_per_re=500, return_num=True, seed=None):
    device = config['general']['device']
    run_name = config['general']['run_name']
    main_path = config['general']['main_path']
    batch_size = config['sampling']['batch_size']
    output_filepath = os.path.join(main_path, config['sampling']['output_folder'], run_name, dataset_name)
    os.makedirs(output_filepath, exist_ok=True)
    calc_gt = True if ground_truth is None else False 

    if config['dataset'].get('multi', False):
        sims_per_re = 100

    assert config['dataset']['name'] == 'airfoil'
    
    predictions = {}
    if calc_gt:
        ground_truth = {}
    sims_per_re = sims_per_re // batch_size
    if sims_per_re == 0:
        sims_per_re = 1
        batch_size = 25

    if config['sampling']['ddim_sampling']:
        diffusion.setup_ddim_sampling()
    elif config['sampling']['iterative_refinement']:
        diffusion.setup_iterative_refinement()

    if not config['dataset'].get('multi', False):
        mean_vals = torch.tensor([ 5.06011047e-01,  4.78371327e-01,  2.09350586e-02, -3.46981432e-11,  4.02468078e-01,  1.19019356e-01])
        std_vals = torch.tensor([0.30319728, 0.2866358,  0.14316697, 0.04403772, 0.14416263, 0.07302476])
    else:
        N = config['dataset']['size_per_dataset']
        if N == 1250: 
            mean_vals = torch.tensor([6.00407019e-01, -8.11805355e-04,  2.02856091e-02, -2.07392593e-12, 4.28425369e-01, -4.01127457e-03])
            std_vals = torch.tensor([0.27305001, 0.47155936, 0.13406734, 0.03254326, 0.12029214, 0.10524579])
        elif N == 5000:
            mean_vals = torch.tensor([6.03294313e-01, -6.56819590e-03, 1.98924163e-02, -8.57931236e-13, 4.29187068e-01, -4.26987535e-03])
            std_vals = torch.tensor([0.26793277, 0.46612973, 0.13254571, 0.03251991, 0.11912243, 0.10441608])
        else:
            raise ValueError(f"Normalization params for N = {N} not implemented!")

    mean_vals = mean_vals.reshape(1,len(mean_vals),1,1)
    std_vals = std_vals.reshape(1,len(std_vals),1,1)

    N = 25

    settings = {'same_random':False, 'seed':None}
    if config['sampling']['method'] == "edm":
        settings['edm_solver'] = config['sampling']['edm_solver']
        settings['edm_stoch'] = config['sampling']['edm_stoch']
    

    for i, data in enumerate(tqdm(dataset)):
        if calc_gt:
            initial_cond, states, meta = data
        else:
            initial_cond, _, meta = data
        initial_cond  = (initial_cond - mean_vals[0,:3]) / std_vals[0,:3]
        vel = ''
        if config['dataset'].get('multi', False):
            vel = str(meta['airfoil'])+str(meta['velocity'])+str(meta['AoA'])
        else:
            vel = meta['velocity']

        if calc_gt:
            states = np.expand_dims(normalized2dimless(states), 0)
            if vel in ground_truth.keys():
                ground_truth[vel] = np.concatenate([ground_truth[vel], states], axis=0)
            else:
                ground_truth[vel] = states
        
        if i % N != 0:
            continue

        initial_cond = initial_cond.to(device)
        
        prediction = []
        for _ in range(sims_per_re):
            if config['sampling']['ddim_sampling']:
                curr_prediction = diffusion.ddim_sampling(model, batch_size, None, initial_cond, None, **settings)
            elif config['sampling']['iterative_refinement']:
                curr_prediction = diffusion.iterative_refinement(model, batch_size, None, initial_cond, None, x0_idx=None, **settings)
            else:
                curr_prediction = diffusion.sample(model, batch_size, None, initial_cond, None, **settings)

            if len(curr_prediction.shape) == 3:
                curr_prediction = curr_prediction.unsqueeze(0)
            prediction.append(curr_prediction)

        prediction = torch.concat(prediction, dim=0).cpu()
        prediction  = (prediction * std_vals[:,3:]) + mean_vals[:,3:]
        prediction = normalized2dimless(prediction).numpy()
        
        if vel in predictions.keys():
            predictions[vel] = torch.concat([predictions[vel], prediction], dim=0)
        else:
            predictions[vel] = prediction

    
    res, mus, sigmas = calc_uncertainty(predictions)
    if calc_gt:
        np.save(os.path.join(output_filepath, f'ground_truth_data.npy'), ground_truth)
        ground_truth = calc_uncertainty(ground_truth)
    gt_res, gt_mus, gt_sigmas = ground_truth
    
    assert gt_res == res, 'Ground truth Res are different from the values used for prediction!'

    mus_error = torch.nn.functional.mse_loss(torch.tensor(mus), torch.tensor(gt_mus), reduction='none').mean(dim=(1,2,3))
    sigmas_error = torch.nn.functional.mse_loss(torch.tensor(sigmas), torch.tensor(gt_sigmas), reduction='none').mean(dim=(1,2,3))


    np.save(os.path.join(output_filepath, f'predictions_{dataset_name}.npy'), predictions)
    np.save(os.path.join(output_filepath, f'mus_{dataset_name}.npy'), mus)
    np.save(os.path.join(output_filepath, f'sigmas_{dataset_name}.npy'), sigmas)
    np.save(os.path.join(output_filepath, f'gt_mus_{dataset_name}.npy'), gt_mus)

    np.save(os.path.join(output_filepath, f'gt_sigmas_{dataset_name}.npy'), gt_sigmas)
    output_filename = f'mus_sigmas_errors.txt'
    with open(os.path.join(output_filepath, output_filename), 'w') as file:
        for i, item in enumerate(mus_error):
            loss_sigma = sigmas_error[i]
            statement = f"mus_MSE = {item} - sigmas_MSE = {loss_sigma}\n"
            file.write(statement)
        mean_mus_error = np.nanmean(mus_error)
        mean_sigmas_error = np.nanmean(sigmas_error)
        statement = f"MSE_mu = {mean_mus_error:.3E}, MSE_sigma = {mean_sigmas_error:.3E}"
        file.write(statement)
        print(statement)
    
    
    if return_num:
        return np.nanmean(mus_error), np.nanmean(sigmas_error), ground_truth


def recursive_sampling(config, diffusion, model, dataset, dataloader, dataset_name, return_num=False, ignore_plot=False, seed=None):
    """
    Perform recusrive sampling on the ENTIRE dataset to print out the MEAN loss for each timestep (depending on step size)
    """
    device = config['general']['device']
    run_name = config['general']['run_name']
    main_path = config['general']['main_path']
    recursive_step = config['sampling']['recursive_step']
    ch_features = config['UNet']['c_out']
    grid_size = config['general']['grid_size']
    all_frames = config['sampling']['all_frames']

    if type(grid_size) == str:
        grid_size = config['general']['grid_size'].split(',')
        grid_size = [int(i) for i in grid_size]
    else:
        grid_size = [grid_size, grid_size]

    if config['general']['train']:
        batch_size = config['training.validation']['batch_size']
    else:
        batch_size = config['sampling']['batch_size']
    
    output_filepath = os.path.join(main_path, config['sampling']['output_folder'], run_name, dataset_name)
    os.makedirs(output_filepath, exist_ok=True)
    output_filename = dataset_name + "_recursive_prediction_errors.txt"
  
    prediction_errors = []

    pbar = tqdm(dataloader)
    if batch_size == 1:
        mse = nn.MSELoss()
    else:
        mse = nn.MSELoss(reduction='none')

    n_simulations = len(pbar)//config['UNet']['model_capacity'] * batch_size

    predictions = np.zeros((config['UNet']['model_capacity'],n_simulations,ch_features,grid_size[0],grid_size[1]))
    references = np.zeros_like(predictions)
    mse_predictions = np.zeros((config['UNet']['model_capacity'],n_simulations))
    mse_predictions[:] = np.nan

    sim_counter = -1

    norm_params = dataset.get_normalization_params()
    t_label_norm_params = [5, 3.1622776601683795]
    mask = dataset.get_mask()
    fluid_params_mean, fluid_params_std = dataset.get_fluid_params_normalization_params()
    if config['dataset']['name'] == 'tra':
        mean = np.concatenate(norm_params[0], axis=2).squeeze(0)
        std = np.concatenate(norm_params[1], axis=2).squeeze(0)
    else:
        mean = norm_params[0][0].squeeze(0)
        std = norm_params[1][0].squeeze(0)
        
    if config['sampling']['ddim_sampling']:
        diffusion.setup_ddim_sampling()
    elif config['sampling']['iterative_refinement']:
        diffusion.setup_iterative_refinement()


    temp_coh = np.zeros_like(mse_predictions)
    temp_coh[:] = np.nan
    temp_coh_ref = np.zeros_like(temp_coh)
    temp_coh_ref[:] = np.nan

    settings = {'same_random':False, 'seed':None}
    if config['sampling']['method'] == "edm":
        settings['edm_solver'] = config['sampling']['edm_solver']
        settings['edm_stoch'] = config['sampling']['edm_stoch']

    with torch.no_grad():
        for i, (initial_cond, states, t_labels, fluid_params) in enumerate(pbar):
            if config['dataset']['residual_data']:
                states = states[1].to(device)
                t_labels = t_labels[1]
            else:
                states = states.to(device)
            initial_cond = initial_cond.to(device)
            fluid_params = fluid_params.to(device)

            i = i % config['UNet']['model_capacity']
            
            recursive_step = config['sampling']['recursive_step']

            if i == 0:
                sim_counter += 1

            if t_labels[0] == 0:
                continue
            else: 
                if i % recursive_step != 0:
                    if t_labels[0] == config['UNet']['model_capacity']-1:
                        flag = True
                        recursive_step = i % recursive_step
                    else:
                        if all_frames:
                            flag = False
                            if i != 1:
                                initial_cond = prediction 
                            recursive_step = 1
                        else:
                            continue
                else:
                    if i == config['sampling']['recursive_step']:
                        prev_pred = initial_cond
                        flag = True


            if i > config['sampling']['recursive_step'] and (recursive_step == config['sampling']['recursive_step'] or recursive_step == i % config['sampling']['recursive_step']):
                if recursive_step != config['sampling']['recursive_step']:
                   initial_cond = prediction
                else:       
                    flag = True
                    initial_cond = prev_pred

            if i == 1 or (config['sampling']['recursive_step'] and not all_frames):
                previous_state = initial_cond
                previous_state_gt = initial_cond
            else:
                previous_state = prediction
            
            # prediction
            recursive_step = torch.tensor([recursive_step]).repeat(batch_size).to(device)
            if not config['dataset'].get('norm_fluid_params', True):
                fluid_params = fluid_params*fluid_params_std + fluid_params_mean
            if config['general']['is_diffusion']:
                if config['sampling']['ddim_sampling']:
                    prediction = diffusion.ddim_sampling(model, batch_size, recursive_step, initial_cond, fluid_params, **settings)
                elif config['sampling']['iterative_refinement']:
                    prediction = diffusion.iterative_refinement(model, batch_size, recursive_step, initial_cond, fluid_params, x0_idx=[i,sim_counter], **settings)
                else:
                    prediction = diffusion.sample(model, batch_size, recursive_step, initial_cond, fluid_params, **settings)
            else:
                prediction = model(x=None, t=None, initial_cond=initial_cond, t_labels=recursive_step, Res=fluid_params)

            if flag:
                prev_pred = prediction

            # De-normalization and mask addition
            if norm_params is not None:
                states_denormed = states.cpu()*std + mean
                prediction_denormed = prediction.cpu()*std + mean
                previous_prediction_denormed = previous_state.cpu()*std + mean
                previous_state_gt_denormed = previous_state_gt.cpu()*std + mean

                if mask is not None:
                    states_denormed *= mask
                    prediction_denormed *= mask
                    previous_prediction_denormed *= mask
                    previous_state_gt_denormed *= mask
                        
            # pointwise errors (MSE, MAE)
            if batch_size == 1:
                pred_error = mse(prediction_denormed, states_denormed).item()
            else:
                pred_error = torch.mean( mse(prediction_denormed, states_denormed),  dim=(1,2,3)).cpu()
            
            print(f'{pred_error = }')

            if batch_size == 1:
                predictions[i,sim_counter] = prediction_denormed.cpu().numpy()
                references[i,sim_counter] = states_denormed.cpu().numpy()
                mse_predictions[i,sim_counter] = pred_error
            else:
                predictions[i] = prediction_denormed.cpu().numpy()
                references[i] = states_denormed.cpu().numpy()
                mse_predictions[i] = pred_error

            previous_state_gt = states

    
    # save the pointwise MSE for each timestep
    for i in range(config['UNet']['model_capacity']):
        prediction_errors.append(np.nanmean(mse_predictions[i,:]))


    np.save(os.path.join(output_filepath, f'predictions_{dataset_name}.npy'), predictions)
    np.save(os.path.join(output_filepath, f'references_{dataset_name}.npy'), references)
    np.save(os.path.join(output_filepath, f'mse_predictions_{dataset_name}.npy'), mse_predictions)
    
    if not ignore_plot:

        with open(os.path.join(output_filepath, output_filename), 'w') as file:
            for i, item in enumerate(prediction_errors):
                statement = f"t: {i} - MSE = {item}\n"
                file.write(statement)
                print(statement)
            statement = f"TA-MSE: {np.nanmean(prediction_errors)}  +- {np.nanstd(prediction_errors)}"
            file.write(statement)
            print(statement)
    
    if return_num:
        return np.nanmean(prediction_errors), np.nanstd(prediction_errors)
        

def simulation_params(name, ft=False, long=False):    
    if ft:
        if long:
            dict = {
                'split': '0,0,1',
                'size_per_dataset': 120,
                'sims_per_dataset': 1,
                'overlap': 0,
                'temporal_stride': 1,
                'shift': 0,
                'name_prefix': 'kolmogorov_res64_cfl0.7_re100_seeds500-501,kolmogorov_res64_cfl0.7_re1750_seeds500-501,kolmogorov_res64_cfl0.7_re5000_seeds500-501',
                'reynolds': '100,1750,5000',
                'seed_values': '0,1,1'
            }
        else:
            dict = {
                'split': '0,0,1',
                'size_per_dataset': 30,
                'sims_per_dataset': 1,
                'overlap': 0,
                'temporal_stride': 1,
                'shift': 0
            }

            if name == 'int':
                dict['name_prefix'] = 'kolmogorov_res64_cfl0.7_re1750_seeds100-199'
                dict['reynolds'] = '1750,1750'
            elif name == 'ext':
                dict['name_prefix'] = 'kolmogorov_res64_cfl0.7_re100_seeds0-99,kolmogorov_res64_cfl0.7_re5000_seeds200-299'
                dict['reynolds'] = '100,5000'
            else:
                raise NotImplementedError("Dataset name not yet implemented!")
    else: 
        if not long:
            dict = {
                'split': '0,0,1',
                'size_per_dataset': 60,
                'sims_per_dataset': 2,
                'overlap': 0,
                'temporal_stride': 2,
                'shift': 500
            }
            if name == 'int':
                dict['seed_values'] = '5,1,7'
            elif name == 'ext':
                dict['seed_values'] = '0,1,2'
            else:
                raise NotImplementedError("Dataset name not yet implemented!")

        else:
            dict = {
                'split': '0,0,1',
                'size_per_dataset': 240,
                'sims_per_dataset': 2,
                'overlap': 0,
                'temporal_stride': 2,
                'shift': 0,
                'seed_values': '3,1,4'
            }

    return dict


def sample(config):
    """
    Load model and loop through all datasets to calculate train, validation, and test losses
    """
    device = config['general']['device']
    run_name = config['general']['run_name']
    main_path = config['general']['main_path']


    if config['dataset']['name'] == 'airfoil' and config['UNet']['same']:
        model=AifNet(f"{main_path}/modules/network_configs.yaml").to(device)
    else:
        model = UNet_conditional(c_in=config['UNet']['c_in'], c_out=config['UNet']['c_out'], config=config).to(device)
    
    if config['sampling']['model'] == 'ema':
        ckpt = "ema_ckpt_best.pt"
    else:
        ckpt = "ckpt_best.pt"

    subfolder_name = None
    if config['dataset']['name'] == 'kolmogorov':
        subfolder_name = 'Fturb'
    elif config['dataset']['name'] == 'tra':
        subfolder_name = 'Tra'
    elif config['dataset']['name'] == 'airfoil':
        subfolder_name = 'Air'
        if config['dataset'].get('multi', False):
            subfolder_name += '_multi'
        else:
            subfolder_name += '_one'
    
    if config['UNet'].get('pre_trained', True):
        model_ckpt = torch.load(os.path.join(main_path,"pretrained_models", subfolder_name, run_name, ckpt))
    else:
        model_ckpt = torch.load(os.path.join(main_path,"models", run_name, ckpt))
    
    model.load_state_dict(model_ckpt)

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model size: {model_size:,}')
    
    if config['general']['is_diffusion']:
        if not config['sampling'].get('method','ddpm') == 'edm':
            diffusion = Diffusion(config)  
        else :
            from diffusion.diffusion_algEDM import DiffusionEDM
            diffusion = DiffusionEDM(config)
    else:
        diffusion = None
    
    if config['dataset']['name'] == 'airfoil':
        ground_truth = None
        ground_truth_int = None
        ground_truth_ext = None
        
        
        if config['dataset'].get('multi', False):
            dataset_int, _ = get_data(config, mode='test_int')
            dataset_ext, _ = get_data(config, mode='test_ext')
        else:
            dataset, _ = get_data(config, mode='test')

        if config['dataset'].get('multi', False):
            mus_errors_int = []
            sigmas_errors_int = []
            mus_errors_ext = []
            sigmas_errors_ext = []
            
            repetitions = 5
            for _ in range(repetitions):
                dataset_name = f"multi_param_test_int"
                mus_error, sigmas_error, ground_truth_int = uncertainty_sampling(config, diffusion, model, dataset_int, dataset_name=dataset_name, ground_truth=ground_truth_int, return_num=True)
                mus_errors_int.append(mus_error)
                sigmas_errors_int.append(sigmas_error)

                dataset_name = f"multi_param_test_ext"
                mus_error, sigmas_error, ground_truth_ext = uncertainty_sampling(config, diffusion, model, dataset_ext, dataset_name=dataset_name, ground_truth=ground_truth_ext, return_num=True)
                mus_errors_ext.append(mus_error)
                sigmas_errors_ext.append(sigmas_error)
            
            # Number of samples
            N_int = 2500
            N_ext = 75

            # Compute mean for each dataset
            mu_int = np.mean(mus_errors_int)
            mu_ext = np.mean(mus_errors_ext)

            # Compute standard deviation for each dataset
            sigma_int = np.sqrt(np.mean(np.square(sigmas_errors_int)) + np.mean(np.square(np.array(mus_errors_int) - mu_int)))
            sigma_ext = np.sqrt(np.mean(np.square(sigmas_errors_ext)) + np.mean(np.square(np.array(mus_errors_ext) - mu_ext)))

            # Compute combined mean
            mu_combined = (N_int * mu_int + N_ext * mu_ext) / (N_int + N_ext)

            # Compute combined standard deviation
            var_combined = ((N_int - 1) * sigma_int**2 + (N_ext - 1) * sigma_ext**2 +
                            N_int * (mu_int - mu_combined)**2 + N_ext * (mu_ext - mu_combined)**2) / (N_int + N_ext - 1)
            sigma_combined = np.sqrt(var_combined)

            print(f"Combined: {mu_combined}, {sigma_combined}")
        else:
            dataset_name = f"1_parameter_test"
            _, _, ground_truth = uncertainty_sampling(config, diffusion, model, dataset, dataset_name="1_parameter_test", ground_truth=ground_truth, return_num=True)
            
    
    elif config['sampling']['recursive_sampling']:
        long = config['dataset'].get('long', False)
        ft = False
        if config['dataset']['name'] == 'kolmogorov':
            ft = True

        if long:
            sim_params = simulation_params(None, ft, long)
            for param in sim_params.keys():
                config['dataset'][param] = sim_params[param]
            config['UNet']['model_capacity'] = sim_params['size_per_dataset']
            config['sampling']['batch_size'] = 4 if not ft else 6
            batch_size = config['sampling']['batch_size']
            dataset, dataloader = get_data(config, bsz=batch_size, mode='test', recursive_sampling=True if batch_size > 1 else False)
            
            recursive_sampling(config, diffusion, model, dataset, dataloader, 'testing_long')
        else:
            sim_params = simulation_params('ext', ft, long)
            for param in sim_params.keys():
                config['dataset'][param] = sim_params[param]
            config['UNet']['model_capacity'] = sim_params['size_per_dataset']
            if ft:
                config['sampling']['batch_size'] = 4
            batch_size = config['sampling']['batch_size']
            print("\n=====\nStarting ext region:\n")
            
            dataset, dataloader = get_data(config, bsz=batch_size, mode='test', recursive_sampling=True if batch_size > 1 else False)
            recursive_sampling(config, diffusion, model, dataset, dataloader, 'testing_ext')

            print("\n=====\nStarting int region:\n")
            sim_params = simulation_params('int', ft, long)
            for param in sim_params.keys():
                config['dataset'][param] = sim_params[param]
            config['UNet']['model_capacity'] = sim_params['size_per_dataset']
            if ft:
                config['sampling']['batch_size'] = 2
            batch_size = config['sampling']['batch_size']
            dataset, dataloader = get_data(config, bsz=batch_size, mode='test', recursive_sampling=True if batch_size > 1 else False)
            recursive_sampling(config, diffusion, model, dataset, dataloader, 'testing_int')            
    else:
        raise ValueError("No sampling procedure selected!")

        
    out_path = os.path.join(main_path, config['sampling']['output_folder'], run_name)
    print("\n=======\nSampling done!")
    print("\nResults saved at:")
    print(out_path)