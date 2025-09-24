import os
from dinov2.logging import setup_logging
from dinov2.train import get_args_parser as get_train_args_parser
from dinov2.train import main as train_main

if __name__ == "__main__":

    setup_logging()
    parser = get_train_args_parser()    #获取训练参数
    args = parser.parse_args()
    args.model_type = None

    assert os.path.isfile(args.config_file), f"Configuration file not found: {args.config_file}"
    train_main(args)

