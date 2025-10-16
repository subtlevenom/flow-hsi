# color-sr
DNN train framework

# usage
usage: main.py [--help] [--hydra-help] [--version] [--cfg {job,hydra,all}] [--resolve] [--package PACKAGE]
               [--run] [--multirun] [--shell-completion] [--config-path CONFIG_PATH]
               [--config-name CONFIG_NAME] [--config-dir CONFIG_DIR]
               [--experimental-rerun EXPERIMENTAL_RERUN]
               [--info [{all,config,defaults,defaults-tree,plugins,searchpath}]]
               [overrides ...]

run: python main.py [--config-name CONFIG_NAME] [--config-dir CONFIG_DIR] [[config_group.name=VALUE] ...] 

print: python main.py --cfg {job,hydra,all}
help: python main.py --help

# examples
python main.py
python main.py --config.name config.test
python main.py --config.name config dataset.type=img

# features
python main.py sample-image-data -i /data/korepanov/color-transfer/huawei/src -o /home/korepanov/work/color-sr/.data/x3 -n 3
    -i - input folder containing source images
    -o - output dataset folder
    -n - scale
    
