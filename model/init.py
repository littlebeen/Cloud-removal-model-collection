

def init(model,config):
    if('My' in config.datasets_dir):
        config.in_ch=4
        config.out_ch=4
    if(model=='spagan'):
        from .models_spagan.train_spagan import train
        print('model: spagan')
    if(model=='amgan'):
        from .models_amgan_cr.train_amgan_cr import train
        print('model: amgan')
    if(model=='msda'):
        from .msda.train_msda import train
        print('model: msda')
    if(model=='cvae'):
        from .cvae.train_cvae import train
        print('model: cvae')
    if(model=='p2p'):
        from .p2p.train_p2p import train
        print('model: p2p')
    if(model=='mn'):
        from .mn.train_memory import train
        print('model: mn')
    if(model=='spaformer'):
        from .spaformer.train_spaformer import train
        print('model: spaformer')
    if(model=='mdsa'):
        from .msda.train_msda import train
        print('model: mdsa')
    if(model=='spaformercomp'):
        from .spaformercomp.train_spaformer import train
        print('model: spaformercomp')
    train(config)