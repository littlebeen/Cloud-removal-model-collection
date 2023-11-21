

def getdata(config):
    data=config.datasets_dir
    if('RICE' in data):
        from .RICE import TrainDataset
    if(data=='T-Cloud'):
        from .Tcloud import TrainDataset
    if('My' in data):
        from .My import TrainDataset
    if(data=='WHU'):
        from .WHU import TrainDataset
    train = TrainDataset(config)
    test = TrainDataset(config,isTrain=False)
    return train, test
