import sys

from llamafactory.cli import main


if __name__ == "__main__":
    arg_list = ["train", "configs/qwen3/qwen3_nothink_bt_squad_v2_train.yaml"]
    sys.argv = sys.argv + arg_list
    main()
