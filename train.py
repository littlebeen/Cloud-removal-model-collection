import os
import shutil
import yaml
from attrdict import AttrMap
from model.init import init
import utils.utils as utils
if __name__ == '__main__':
    with open('config.yml', 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = AttrMap(config)

    utils.make_manager()
    os.makedirs(config.out_dir,exist_ok=True)

    # 保存本次训练时的配置
    shutil.copyfile('config.yml', os.path.join(config.out_dir, 'config.yml'))

    init(config.model,config)