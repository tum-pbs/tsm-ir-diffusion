from multiprocessing import reduction
from utils.header import * 
import copy

from .sampling import recursive_sampling, uncertainty_sampling

def load_models_for_resume(config, model, ema_model, optimizer, lr_scheduler):
    """
    Loads the previous states for model, ema_model, and optimizer
    """
    main_path = config['general']['main_path']
    run_name = config['general']['run_name']

    # reading original model
    ckpt = torch.load(os.path.join(main_path,"models", run_name, f"ckpt.pt"))
    model.load_state_dict(ckpt)
    
    # reading EMA model
    ema_model_ckpt = torch.load(os.path.join(main_path,"models", run_name, f"ema_ckpt.pt"))
    ema_model.load_state_dict(ema_model_ckpt)

    # reading Optimizer state
    optimizer_ckpt = torch.load(os.path.join(main_path,"models", run_name, f"optim.pt"))
    optimizer.load_state_dict(optimizer_ckpt)

    # reading learning rate scheduler state
    if lr_scheduler is not None:
        lr_scheduler_ckpt = torch.load(os.path.join(main_path,"models", run_name, f"lr_scheduler.pt"))
        lr_scheduler.load_state_dict(lr_scheduler_ckpt)

    max_MSE_valid = None
    max_MSE_train = None

    return model, ema_model, optimizer, lr_scheduler, max_MSE_valid, max_MSE_train


def training_epoch(config, dataloader, diffusion, model, ema, ema_model, optimizer, logger, epoch, accumulation_steps):
    """
    Runs the main logic of a training epoch and return the main MSE for the entire training dataset
    """
    device = config['general']['device']
    sim_time = config['UNet']['sim_time']
    is_diffusion = False if diffusion is None else True
    is_edm = False
    if is_diffusion:
        skip_percent = config['diffusion']['skip_percent']
        noise_steps = config['diffusion']['noise_steps']
        skip_t = int(skip_percent * noise_steps)

        if config['sampling']['method'] == 'edm':
            is_edm = True

    mse = nn.MSELoss()
    MSE_train = []

    pbar = tqdm(dataloader)
    n_batches = len(pbar)

    running_loss = 0.0
    
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

    for i, data in enumerate(pbar):
        if len(data) == 3 and config['dataset']['name'] == 'airfoil':
            initial_cond, states, meta = data
            t_labels = Res = torch.zeros(1,)

            initial_cond  = (initial_cond - mean_vals[:,:3]) / std_vals[:,:3]
            states  = (states - mean_vals[:,3:]) / std_vals[:,3:]

        elif len(data) == 4:
            initial_cond, states, t_labels, Res = data
    
        if config['dataset']['residual_data']:
            states = states[1].to(device)
            t_labels = t_labels[1].to(device)
        else:
            states = states.to(device)
            ground_truth = states
            t_labels = t_labels.to(device)

        initial_cond = initial_cond.to(device)
    
        Res = Res.to(device)

        if is_diffusion:
            if is_edm:
                loss = diffusion.train_step(model, states, initial_cond, t_labels, Res, gt_guide_type='l2')
            else:
                t = diffusion.sample_timesteps(states.shape[0]).to(device)
                t = t.to(device)
            
                if skip_percent > 0:
                    indices = torch.nonzero(torch.lt(t, skip_t)).squeeze(-1)                
                    t[indices] = skip_t

                x_t, noise = diffusion.noise_states(states, t)
                predicted_noise = model(x_t, t=t, initial_cond=initial_cond, t_labels=t_labels, Res=Res) # 1
                loss = mse(noise, predicted_noise)      # 2
        
        loss = loss / accumulation_steps 
        running_loss += loss.item()

        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()                      
            optimizer.zero_grad()                   
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=running_loss)
            MSE_train.append(running_loss)
            logger.add_scalar(f'MSE_step_Train', running_loss, global_step=epoch * n_batches + i)
            
            running_loss = 0
    
    return np.mean(MSE_train)

def validation_step(device, initial_cond, states, t_labels, Res, diffusion, model, batch_size, mse, residual_data, norm_params=None, mask=None):
    """
    Runs only a single batch of validation data and returns its mean MSE + predicted states
    """
    is_diffusion = False if diffusion is None else True
    initial_cond = initial_cond.to(device)
    if residual_data:
        states = states[1].to(device)
        t_labels = t_labels[1].to(device)
    else:
        states = states.to(device)
        t_labels = t_labels.to(device)
    Res = Res.to(device)

    if is_diffusion:
        predicted_states = diffusion.sample(model, batch_size, t_labels, initial_cond, Res, same_random=True)
    else:
        predicted_states = model(x=None, t=None, initial_cond=initial_cond, t_labels=t_labels, Res=Res)

    if norm_params is not None:
        mean = np.concatenate(norm_params[0], axis=2).squeeze(0)
        std = np.concatenate(norm_params[1], axis=2).squeeze(0)

        states = states.cpu()*std + mean
        predicted_states = predicted_states.cpu()*std + mean

        if mask is not None:
            states *= mask
            predicted_states *= mask
    
    residual_loss = 0

    if batch_size > 1:
        mse_loss = torch.nn.functional.mse_loss(states, predicted_states, reduction='none').mean(dim=(1,2,3)).numpy()
    else:
        mse_loss = mse(states, predicted_states).item()

    return mse_loss, predicted_states.to(device), residual_loss



def validation_epoch(config, dataloader, diffusion, model, dataset, ground_truth=None):
    """
    Runs full validation epoch and returns the dataset mean MSE + predicted states
    """
    device = config['general']['device']
    batch_size = config['training.validation']['batch_size']
    airfoil_case = config['dataset']['name'] == 'airfoil'

    mse = nn.MSELoss()
    MSE_valid = []

    pbar = tqdm(dataloader)

    if not airfoil_case:
        norm_params = dataset.get_normalization_params()
        mask = dataset.get_mask()
    else:
        norm_params = mask = None
        dataset_name = '1_parameter'

    if config['training.validation']['rec_sampling']:
        config['sampling']['all_frames'] = False
        config['sampling']['recursive_step'] = 1
        mse_valid, mse_valid_last = recursive_sampling(config, diffusion, model, dataset, dataloader, dataset_name='validation', return_num=True, ignore_plot=True)
        return mse_valid, None, mse_valid_last
    elif airfoil_case:
        mus_error, sigmas_error, ground_truth = uncertainty_sampling(config, diffusion, model, dataset, dataset_name=dataset_name, ground_truth=ground_truth, return_num=True)
        return mus_error, sigmas_error, ground_truth
    else:
        for _, (initial_cond, states, t_labels, Res) in enumerate(pbar):
            mse_valid_item, predicted_states, _ = validation_step(device, initial_cond, states, t_labels, Res, diffusion, model, batch_size, mse, config['dataset']['residual_data'], norm_params, mask)
            MSE_valid.append(mse_valid_item)

        return np.mean(MSE_valid), predicted_states, None
         

def set_rec_valid_dataset(config):
    if config['dataset']['name'] == 'tra':
        bsz = config['training.validation']['batch_size']
        assert bsz == 6
        config['sampling']['batch_size'] = bsz

        config['dataset']['split'] = '0,1,0'
        config['dataset']['shift'] = 500
        config['dataset']['size_per_dataset'] = 60
        config['dataset']['sims_per_dataset'] = 2
        config['dataset']['overlap'] = 0
        config['UNet']['model_capacity'] = config['dataset']['size_per_dataset']
        
        config['dataset']['seed_values'] = '0,1,2'
        dataset_validation_ext, dataloader_validation_ext = get_data(config, mode='valid', bsz=bsz, recursive_sampling=True)

        config['dataset']['seed_values'] = '5,1,7'
        dataset_validation_int, dataloader_validation_int = get_data(config, mode='valid', bsz=bsz, recursive_sampling=True)
    
    elif config['dataset']['name'] == 'kolmogorov':
        config['dataset']['split'] = '0,1,0'
        config['dataset']['shift'] = 0
        config['dataset']['size_per_dataset'] = 30
        config['dataset']['sims_per_dataset'] = 1
        config['dataset']['overlap'] = 0
        config['dataset']['seed_values'] = '0,1,1'
        config['UNet']['model_capacity'] = config['dataset']['size_per_dataset']

        config['dataset']['name_prefix'] = 'kolmogorov_res64_cfl0.7_re100_seeds0-99,kolmogorov_res64_cfl0.7_re5000_seeds200-299'
        config['dataset']['reynolds'] = '100,5000'
        config['training.validation']['batch_size'] = 4
        dataset_validation_ext, dataloader_validation_ext = get_data(config, mode='valid', bsz=config['training.validation']['batch_size'], recursive_sampling=True)

        config['dataset']['name_prefix'] = 'kolmogorov_res64_cfl0.7_re1750_seeds100-199'
        config['dataset']['reynolds'] = '1750,1750'
        config['training.validation']['batch_size'] = 2
        dataset_validation_int, dataloader_validation_int = get_data(config, mode='valid', bsz=config['training.validation']['batch_size'], recursive_sampling=True)
    else:
        raise NotImplementedError()

    return dataset_validation_ext, dataloader_validation_ext, dataset_validation_int, dataloader_validation_int


    
def train(config):
    """
    Runs the main training and validation loops for full model training 
    """
    main_path = config['general']['main_path']
    run_name = config['general']['run_name']
    device = config['general']['device']
    resume = config['training']['resume']
    previous_epochs = config['training']['previous_epochs']
    is_diffusion = config['general']['is_diffusion']

    output_filepath = os.path.join(main_path, config['training.validation']['output_folder'], run_name)
    
    logger = SummaryWriter(os.path.join("runs", run_name))

    # dataloaders
    _, dataloader_training = get_data(config, mode='train', load_ddpm_states=False)

    if config['training.validation']['rec_sampling']:
        dataset_validation_ext, dataloader_validation_ext, dataset_validation_int, dataloader_validation_int = set_rec_valid_dataset(config)
    else:
        if config['dataset']['name'] == 'airfoil' and config['dataset'].get('multi', False):
            dataset_validation_int, dataloader_validation_int = get_data(config, mode='test_int')
            dataset_validation_ext, dataloader_validation_ext = get_data(config, mode='test_ext')
        else:
            dataset_validation, dataloader_validation = get_data(config, mode='valid')

    if not config['training.validation']['full_epoch'] and not config['training.validation']['rec_sampling']:
        dataloader_validation = iter(itertools.cycle(dataloader_validation))
        initial_cond_val, states_val, t_labels_val, Re = dataset_validation[-1]
        Re = torch.tensor(Re)
    
    # model and optimizer definition
    mse = nn.MSELoss()
    diffusion = None
    if is_diffusion:
        diffusion = Diffusion(config) if not config['sampling']['method'] == 'edm' else DiffusionEDM(config)

    if config['dataset']['name'] == 'airfoil' and config['UNet']['same']:
        model=AifNet(f"{main_path}/modules/network_configs.yaml").to(device)
    else:
        model = UNet_conditional(c_in=config['UNet']['c_in'], c_out=config['UNet']['c_out'], config=config).to(device)

    ema = EMA(config['ema']['ema_param'])
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model size: {model_size:,}')

    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])

    # learning rate scheduler
    if config['training']['lr_schedule_type'] == 'exp':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['training']['lr_gamma'])
    elif config['training']['lr_schedule_type'] == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'], eta_min=config['training']['lr_eta_min'])
    elif config['training']['lr_schedule_type'] == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['training']['lr_step_size'], gamma=config['training']['lr_gamma_step'])
    elif config['training']['lr_schedule_type'] == 'none':
        lr_scheduler = None
    else:
        raise NotImplementedError("Learning rate scheduler not implemented!")
    
    # reload previous models and optimizer states
    if resume:
        model, ema_model, optimizer, lr_scheduler, _, _ = load_models_for_resume(config, model, ema_model, optimizer, lr_scheduler)

    # arbitrary large initialization value to keep track of min mean MSE of validation  
    min_mean_MSE_valid = 200 * config['dataset']['size_per_dataset']
    
    accumulation_steps = config['training']['accumulation_steps']


    is_multiParam_airfoil = config['dataset']['name'] == 'airfoil' and config['dataset'].get('multi', False)
    is_recSample_tra = (config['dataset']['name'] == 'tra' or config['dataset']['name'] == 'kolmogorov') and config['training.validation']['rec_sampling']
    consider_ext_and_int_data = is_recSample_tra or is_multiParam_airfoil
    
    if is_multiParam_airfoil:
        ground_truth_int = None
        ground_truth_ext = None
    else:
        ground_truth = None

    for epoch in range(config['training']['epochs']):
        logging.info(f"Starting epoch {epoch}:")

        # main training epoch
        mean_MSE_train = training_epoch(config, dataloader_training, diffusion, model, ema, ema_model, optimizer, logger, epoch, accumulation_steps)
        
        print(f'{mean_MSE_train = }')
        
        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()
            logger.add_scalar(f'lr', lr_scheduler.get_last_lr()[0], global_step=epoch+previous_epochs)
        else:
            logger.add_scalar(f'lr', config['training']['learning_rate'], global_step=epoch+previous_epochs)    

        mse_dict = { 'training': mean_MSE_train }

        # Tensorboard logging
        logger.add_scalar(f'train', mse_dict['training'], global_step=epoch+previous_epochs)        

        # validation step/epoch
        if (epoch+1) % config['training.validation']['frequency'] == 0:
            # full epoch (i.e. looping through entire validation dataset)
            if config['training.validation']['full_epoch']:
                if is_recSample_tra:
                    if config['dataset']['name'] == 'kolmogorov':
                        config['training.validation']['batch_size'] = 4
                        config['UNet']['model_capacity'] = 30
                    mean_MSE_valid_ext, _, mean_MSE_valid_ext_last = validation_epoch(config, dataloader_validation_ext, diffusion, ema_model, dataset_validation_ext)
                    if config['dataset']['name'] == 'kolmogorov':
                        config['training.validation']['batch_size'] = 2
                        config['UNet']['model_capacity'] = 30
                    mean_MSE_valid_int, _, mean_MSE_valid_int_last = validation_epoch(config, dataloader_validation_int, diffusion, ema_model, dataset_validation_int)
                    mean_MSE_valid = np.mean([mean_MSE_valid_ext, mean_MSE_valid_int])
                else:
                    if config['dataset']['name'] == 'airfoil':
                        if config['dataset'].get('multi', False):
                            mean_MSE_valid_1, mean_MSE_valid_2, ground_truth_ext = validation_epoch(config, dataloader_validation_ext, diffusion, ema_model, dataset_validation_ext, ground_truth_ext)
                            mean_MSE_valid_ext = np.mean([mean_MSE_valid_1, mean_MSE_valid_2])
                            mean_MSE_valid_1, mean_MSE_valid_2, ground_truth_int = validation_epoch(config, dataloader_validation_int, diffusion, ema_model, dataset_validation_int, ground_truth_int)
                            mean_MSE_valid_int = np.mean([mean_MSE_valid_1, mean_MSE_valid_2])
                            mean_MSE_valid = np.mean([mean_MSE_valid_ext, mean_MSE_valid_int])
                        else:
                            mean_MSE_valid_1, mean_MSE_valid_2, ground_truth = validation_epoch(config, dataloader_validation, diffusion, ema_model, dataset_validation, ground_truth)
                            mean_MSE_valid = np.mean([mean_MSE_valid_1, mean_MSE_valid_2])
                    else:
                        mean_MSE_valid, _, _ = validation_epoch(config, dataloader_validation, diffusion, ema_model, dataset_validation)
            # single batch only
            else:
               
                mean_MSE_valid, predicted_states, _ = validation_step(device, initial_cond_val, states_val, torch.tensor(t_labels_val), Re, diffusion, ema_model, 1, mse)

            
            if consider_ext_and_int_data:
                if is_recSample_tra:
                    print(f'{mean_MSE_valid_ext = }, {mean_MSE_valid_ext_last = }')
                    print(f'{mean_MSE_valid_int = }, {mean_MSE_valid_int_last = }')
                else:
                    print(f'{mean_MSE_valid_ext = }')
                    print(f'{mean_MSE_valid_int = }')

            print(f'{mean_MSE_valid = }')

            if consider_ext_and_int_data:
                mse_dict['valid_ext'] = mean_MSE_valid_ext
                mse_dict['valid_int'] = mean_MSE_valid_int
                if is_recSample_tra:
                    mse_dict['valid_ext_last'] = mean_MSE_valid_ext_last
                    mse_dict['valid_int_last'] = mean_MSE_valid_int_last

            mse_dict['valid'] = mean_MSE_valid
            
            # Tensorboard logging
            for param in mse_dict.keys():
                if param == 'training':
                    continue
                logger.add_scalar(param, mse_dict[param], global_step=epoch+previous_epochs)        

            if mean_MSE_valid < min_mean_MSE_valid:
                min_mean_MSE_valid = mean_MSE_valid
                torch.save(model.state_dict(), os.path.join("models", run_name, f"ckpt_best.pt"))
                torch.save(ema_model.state_dict(), os.path.join("models", run_name, f"ema_ckpt_best.pt"))

        # saving models
        if epoch % config['training']['save_frequency'] == 0:
            if config['training']['checkpoint']:
                model_name = f"ckpt_{epoch}.pt"
                model_name_ema = f"ema_ckpt_{epoch}.pt"
            else:
                model_name = f"ckpt.pt"
                model_name_ema = f"ema_ckpt.pt"

            torch.save(model.state_dict(), os.path.join("models", run_name, model_name))
            torch.save(ema_model.state_dict(), os.path.join("models", run_name, model_name_ema))
            torch.save(optimizer.state_dict(), os.path.join("models", run_name, f"optim.pt"))
            if lr_scheduler is not None:
                torch.save(lr_scheduler.state_dict(), os.path.join("models", run_name, f"lr_scheduler.pt"))