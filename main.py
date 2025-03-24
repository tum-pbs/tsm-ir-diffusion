from utils.header import * 

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def parse_args(args):
    if len(args) < 2:
        raise ValueError("Usage: python main.py -config filename.ini")
    else:
        config_file_path = None
        
        for i in range(1, len(args)):
            if args[i] == '-config' and i < len(args) - 1:
                config_file_path = args[i + 1]
                break
        
        if config_file_path:
            print(f'Configuration file: {config_file_path}')
        else:
            raise ValueError('No configuration file specified.')
    
    return config_file_path


if __name__ == '__main__':
    args = sys.argv
    config_file = parse_args(args)

    config_filename = os.path.join(os.getcwd(), config_file)
    config = get_configs(config_filename)
    
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=None, abbreviated=False)

    config['general']['main_path'] = os.getcwd()

    if not config['training']['resume']:
        config['training']['previous_epochs'] = 0

    setup_logging(config)
    
    print('Started Run: ', config['general']['run_name'])

    if config['general']['train']:
        train(config)
    if config['general']['sample']:
        sample(config)    