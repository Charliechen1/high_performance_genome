import configparser

def load_conf(config_path):
    """
    function to load the global configuration
    """
    conf = configparser.ConfigParser()
    conf.read(config_path)
            
    model_conf = configparser.ConfigParser()
    model_conf.read(conf['path']['model'])
    return conf, model_conf
    