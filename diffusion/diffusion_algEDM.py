# This implementation code is largely based on the code from this GitHub repository: https://github.com/yuanzhi-zhu/mini_edm

from utils.header import * 

class DiffusionEDM:
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
        
        self.skip_percent = 0
        self.noise_steps = self.diffusion_config['noise_steps']

        self.num_steps=self.noise_steps
        self.sigma_min=0.002
        self.sigma_max=80
        self.rho=7
        self.sigma_data = 0.5
        self.P_mean = -1.2
        self.P_std = 1.2


    def noise_states(self, x, t, same_random=False):
        """
        Noises a batch of states
        """
        if len(x.shape) > 3:
            num_channels = x.shape[1]
        else: 
            num_channels = x.shape[0]

        if same_random:
            eps = torch.randn((num_channels, self.grid_size[0], self.grid_size[1])).repeat(x.shape[0], 1, 1, 1).to(self.device)
        else:
            eps = torch.randn_like(x)
        
        self.noise_scheduler.set_timesteps(self.noise_steps)
        self.noise_scheduler.set_begin_index()

        x = self.noise_scheduler.add_noise(x, eps, t)
        
        return x, eps

    def sample_timesteps(self, n):
        """
        Returns random outputs for diff_time for training only 
        """
        return torch.randint(low=0, high=self.noise_steps, size=(n,))


    def train_step(self, model, images, initial_cond, t_labels, Res, gt_guide_type='l2'):
        rnd_normal = torch.randn([images.shape[0]], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        
        y = images
        
        noise = torch.randn_like(y)
        n = torch.einsum('b,bijk->bijk', sigma, noise)
        D_yn = self.model_forward_wrapper(y + n, sigma, model, initial_cond=initial_cond, t_labels=t_labels, Res=Res)
        
        if gt_guide_type == 'l2':
            loss = torch.einsum('b,bijk->bijk', weight, ((D_yn - y) ** 2))
        elif gt_guide_type == 'l1':
            loss = torch.einsum('b,bijk->bijk', weight, (torch.abs(D_yn - y)))
        else:
            raise NotImplementedError(f'gt_guide_type {gt_guide_type} not implemented')
        
        return loss.mean()

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma) 
    
    def model_forward_wrapper(self, x, sigma, model, initial_cond, t_labels, Res):
        """Wrapper for the model call"""
        sigma[sigma == 0] = self.sigma_min
        ## edm preconditioning for input and output
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        model_output = model(torch.einsum('b,bijk->bijk', c_in, x), t=c_noise, initial_cond=initial_cond, t_labels=t_labels, Res=Res)

        return torch.einsum('b,bijk->bijk', c_skip, x) + torch.einsum('b,bijk->bijk', c_out, model_output)
        

    def edm(self, x, sigma, model, initial_cond, t_labels, Res):
        if sigma.shape == torch.Size([]):
            sigma = sigma * torch.ones([x.shape[0]]).to(x.device)
        return self.model_forward_wrapper(x.float(), sigma.float(), model, initial_cond=initial_cond, t_labels=t_labels, Res=Res)

    def sample(self, model, n, t_labels, initial_cond, fluid_params, edm_solver, edm_stoch, same_random=False, seed=None):
        """
        Main sample loop for EDMs
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
            # EDM sampling params
            num_steps=self.num_steps
            sigma_min=self.sigma_min
            sigma_max=self.sigma_max
            rho=self.rho

            # Time step discretization.
            step_indices = torch.arange(num_steps, dtype=torch.float64, device=self.device)
            t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
            t_steps = torch.cat([self.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
            
            if same_random:
                x_next = torch.randn((num_channels, self.grid_size[0], self.grid_size[1])).repeat(n, 1, 1, 1).to(self.device)
            else:
                x_next = torch.randn((n, num_channels, self.grid_size[0], self.grid_size[1])).to(self.device)

            x_next = x_next * t_steps[0]


            initial_cond = initial_cond.to(self.device)
            if fluid_params is not None:
                fluid_params = fluid_params.to(self.device)

            for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1

                if fluid_params is not None:
                    fluid_params = fluid_params.to(self.device)
                
                # ============= Start Deter sampling =============
                if not edm_stoch:
                    t_hat = t_cur
                    x_hat = x_next
                # ============= End Deter sampling =============
                

                # # ============= Start stochastic sampling =============
                else:
                    noise = torch.randn((n, num_channels, self.grid_size[0], self.grid_size[1])).to(self.device)
                    
                    S_churn = 10
                    S_tmin = 0
                    S_tmax = 1e6
                    S_noise = 1

                    gamma = min(S_churn/self.noise_steps, 2**0.5 -1) if t_cur >= S_tmin and t_cur <= S_tmax else 0
                    noise = noise * S_noise
                    t_hat = t_cur + gamma * t_cur
                    x_hat = x_next + (t_hat**2 - t_cur**2)**0.5 * noise
                # # ============= End stochastic sampling =============
                

                # Euler step.
                denoised = self.edm(x_hat, t_hat, model, initial_cond=initial_cond, t_labels=t_labels, Res=fluid_params)
                d_cur = (x_hat - denoised) / t_hat
                x_next = x_hat + (t_next - t_hat) * d_cur

                if edm_solver == 'heun':
                    # Apply 2nd order correction.
                    if i < num_steps - 1:
                        denoised = self.edm(x_next, t_next, model, initial_cond=initial_cond, t_labels=t_labels, Res=fluid_params)
                        d_prime = (x_next - denoised) / t_next
                        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
                                                
        model.train()
        return x_next