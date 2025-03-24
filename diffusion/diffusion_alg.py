from utils.header import * 

class Diffusion:
    def __init__(self, config):
        self.config = config
        self.diffusion_config = self.config['diffusion']

        self.device = self.config['general']['device']
        self.grid_size = self.config['general']['grid_size']
        if type(self.grid_size) == str:
            self.grid_size = self.config['general']['grid_size'].split(',')
            self.grid_size = [int(i) for i in self.grid_size]
        else:
            self.grid_size = [self.grid_size, self.grid_size]
        
        self.schedule = self.diffusion_config['schedule']
        self.skip_percent = self.diffusion_config['skip_percent']
        self.noise_steps = self.diffusion_config['noise_steps']
        self.beta_start = self.diffusion_config['beta_start']
        self.beta_end = self.diffusion_config['beta_end']

        clip_min = 1e-9

        if self.schedule == 'cosine_airfoil':
            s = 0.008
            tlist = torch.arange(1, self.noise_steps+1, 1)
            temp1 = torch.cos((tlist/self.noise_steps+s)/(1+s)*np.pi/2)
            temp1 = temp1*temp1
            temp2 = np.cos(((tlist-1)/self.noise_steps+s)/(1+s)*np.pi/2)
            temp2 = temp2*temp2

            self.beta_source = 1-(temp1/temp2)
            self.beta_source[self.beta_source > 0.999] = 0.999
            self.beta = torch.cat((torch.tensor([0]), self.beta_source), dim=0)[1:]
            self.beta = self.beta.to(self.device)
            self.alpha = 1-self.beta
            self.alpha_hat = torch.cumprod(self.alpha, 0)
        elif self.schedule == 'simple_linear':
            t = torch.linspace(0, 1, self.noise_steps).to(self.device)
            self.alpha_hat = torch.clip(1-t, min=clip_min, max=1)
            self.alpha = torch.div(self.alpha_hat, torch.cat((torch.tensor([1]).to(self.device), self.alpha_hat[:-1]), 0))
            self.beta = 1 - self.alpha
        else :
            self.beta = self.prepare_noise_schedule().to(self.device)
            self.alpha = 1. - self.beta
            self.alpha_hat = torch.cumprod(self.alpha, dim=0)
            self.alpha_hat = torch.clip(self.alpha_hat, min=clip_min, max=1)
        
        

    def prepare_noise_schedule(self):
        """
        Creates the noise schedule beta
        """
        if self.schedule == 'linear':
            scale = (self.diffusion_config['scale']/self.noise_steps)
            self.beta_start = self.beta_start * scale
            self.beta_end = self.beta_end * scale
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        elif self.schedule == 'cosine':
            timesteps = torch.linspace(0, 1, self.noise_steps)
            betas = self.beta_end + 0.5 * (self.beta_start - self.beta_end) * (1 + np.cos(np.pi/2 * timesteps)**2)
            return torch.tensor(betas, dtype=torch.float)
        else:
            return NotImplementedError("Noising Schedule Not Implemented!")

    def noise_states(self, x, t, same_random=False):
        """
        Noises a batch of states
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        if len(x.shape) > 3:
            num_channels = x.shape[1]
        else: 
            num_channels = x.shape[0]

        if same_random:
            eps = torch.randn((num_channels, self.grid_size[0], self.grid_size[1])).repeat(x.shape[0], 1, 1, 1).to(self.device)
        else:
            eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps


    def sample_timesteps(self, n):
        """
        Returns random outputs for diff_time for training only 
        """
        low = int(self.skip_percent * self.noise_steps)
        return torch.randint(low=low, high=self.noise_steps, size=(n,))
    
    
    def setup_ddim_sampling(self):
        config = self.config
        noise_steps = self.noise_steps

        delta = noise_steps//config['sampling']['ddim_steps']
        ref_arr = list(reversed(range(0, noise_steps, delta)))
        ref_arr = np.array(ref_arr)/noise_steps

        for val in ref_arr:
            assert val < 1, val
            assert val >= 0, val            

        self.ref_arr = ref_arr
    
    def ddim_sampling(self, model, n, t_labels, initial_cond, fluid_params, same_random=True, seed=None):
        model.eval()

        if seed is not None:
            torch.manual_seed(seed)

        ref_arr = self.ref_arr

        if len(initial_cond.shape) > 3:
            num_channels = initial_cond.shape[1]
            assert initial_cond.shape[0] == n
        else: 
            num_channels = initial_cond.shape[0]
            initial_cond = initial_cond.repeat(n,1,1,1)

        with torch.no_grad():
            if same_random:
                x = torch.randn((num_channels, self.grid_size[0], self.grid_size[1])).repeat(n, 1, 1, 1).to(self.device)
            else:
                x = torch.randn((n, num_channels, self.grid_size[0], self.grid_size[1])).to(self.device)
            initial_cond = initial_cond.to(self.device)
            if t_labels is not None:
                t_labels = t_labels.to(self.device)
            if fluid_params is not None:
                fluid_params = fluid_params.to(self.device)

            denoising_steps = self.noise_steps
            skip = self.noise_steps // self.config['sampling']['ddim_steps']
            trange = reversed(range(0, self.noise_steps, skip))

            for idx, i in enumerate(ref_arr):
                t = (torch.ones(n) * int(i * denoising_steps)).long().to(self.device)
                
                if idx > 0:
                    sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t][:, None, None, None])
                    sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t][:, None, None, None])
                    x = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * predicted_noise
                                        
                predicted_noise = model(x, t=t, initial_cond=initial_cond, t_labels=t_labels, Res=fluid_params)               
                
                sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t][:, None, None, None])
                sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t][:, None, None, None])

                x = (x - sqrt_one_minus_alpha_hat * predicted_noise)/sqrt_alpha_hat

        return x



    def setup_iterative_refinement(self):
        config = self.config
        noise_steps = config['diffusion']['noise_steps']
        x_init = config['sampling']['x_init']
        dataset_name = 'testing'

        self.trunc_sample_ir = False
        if x_init == 'noise':
            self.itr_ref_x0_noise = True
        elif x_init == 'sample':
            self.trunc_sample_ir = True
            self.itr_ref_x0_noise = False
        else:
            raise ValueError("Allowed values for x_init are \{none, sample\}.")
        
        gamma = config['sampling']['ir_gamma_schedule']

        if gamma == 1:
            ref_arr = np.arange(start=0.005, stop=0.91, step=0.2)[::-1]
        elif gamma == 2:
            ref_arr = np.arange(start=0.005, stop=0.9, step=0.1)[::-1]
        elif gamma == 3:
            ref_arr = np.arange(start=0.005, stop=0.7, step=0.05)[::-1]
        elif gamma == 4:
            ref_arr = np.arange(start=0.005, stop=0.907, step=0.05)[::-1]
        elif gamma == 5:
            ref_arr = [1/noise_steps]
        elif gamma == 'linear':
            delta = noise_steps//self.config['sampling']['ir_steps']
            ref_arr = list(reversed(range(0, noise_steps, delta)))
            ref_arr = np.array(ref_arr)/noise_steps
        else:
            raise ValueError("Schedule undefined!")

        for val in ref_arr:
            assert val >= 0, val
            assert val < 1, val 

        self.ref_arr = ref_arr

    
    def iterative_refinement(self, model, n, t_labels, initial_cond, fluid_params, x0_idx=None, same_random=True, seed=None):
        model.eval()

        if seed is not None:
            torch.manual_seed(seed)

        ref_arr = self.ref_arr

        if len(initial_cond.shape) > 3:
            num_channels = initial_cond.shape[1]
            assert initial_cond.shape[0] == n
        else: 
            num_channels = initial_cond.shape[0]
            initial_cond = initial_cond.repeat(n,1,1,1)

        if same_random:
            x = torch.randn((num_channels, self.grid_size[0], self.grid_size[1])).repeat(n, 1, 1, 1).to(self.device)
        else:
            x = torch.randn((n, num_channels, self.grid_size[0], self.grid_size[1])).to(self.device)

        with torch.no_grad():
            x = x.to(self.device)
            initial_cond = initial_cond.to(self.device)
            if t_labels is not None:
                t_labels = t_labels.to(self.device)
            if fluid_params is not None:
                fluid_params = fluid_params.to(self.device)
            
            if self.trunc_sample_ir:
                x = self.sample(model, n, t_labels, initial_cond, fluid_params, same_random=same_random)

            denoising_steps = self.noise_steps

            for i in ref_arr:
            # for i in tqdm(ref_arr, position=0):
                t = (torch.ones(n) * int(i * denoising_steps)).long().to(self.device)
                
                x, _ = self.noise_states(x, t, same_random)

                predicted_noise = model(x, t=t, initial_cond=initial_cond, t_labels=t_labels, Res=fluid_params)
                
                sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t][:, None, None, None])
                sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t][:, None, None, None])

                x = (x - sqrt_one_minus_alpha_hat * predicted_noise)/sqrt_alpha_hat

        return x


    def sample(self, model, n, t_labels, initial_cond, fluid_params, same_random=False, seed=None):
        """
        Main sample loop for DDPMs & TSMs
        """

        model.eval()

        if seed is not None:
            torch.manual_seed(seed)

        if len(initial_cond.shape) > 3:
            num_channels = initial_cond.shape[1]
            assert initial_cond.shape[0] == n, f'{initial_cond.shape}, {n}'
        else: 
            num_channels = initial_cond.shape[0]
            initial_cond = initial_cond.repeat(n,1,1,1)
        
        with torch.no_grad():
            if same_random:
                x = torch.randn((num_channels, self.grid_size[0], self.grid_size[1])).repeat(n, 1, 1, 1).to(self.device)
            else:
                x = torch.randn((n, num_channels, self.grid_size[0], self.grid_size[1])).to(self.device)

            initial_cond = initial_cond.to(self.device)
            if fluid_params is not None:
                fluid_params = fluid_params.to(self.device)

            denoising_steps = self.noise_steps

            for i in reversed(range(0, denoising_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                if fluid_params is not None:
                    fluid_params = fluid_params.to(self.device)
                
                if i > 0:
                    if same_random:
                        noise = torch.randn_like(x[0]).repeat(n, 1, 1, 1)
                    else:
                        noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                predicted_noise = model(x, t=t, initial_cond=initial_cond, t_labels=t_labels, Res=fluid_params)

                if i == int(self.skip_percent * denoising_steps) and self.skip_percent > 0:
                    sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t][:, None, None, None])
                    sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t][:, None, None, None])
                    x = (x - sqrt_one_minus_alpha_hat * predicted_noise)/sqrt_alpha_hat
                    break
                
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                                
        model.train()
        return x