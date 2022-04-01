DATA_NAME = "wangwen"
CTX_LEN = 512  # 模型关注的句子长度
N_LAYER = 12
N_HEAD = 12
N_EPOCH = 100
BATCH_SIZE = 16
EPOCH_SAVE_FREQUENCY = 1
TRAINED_MODEL = "./model/"

"""
RUN_DEVICE 运行的设备
gpu：只支持 nvidia 显卡，速度最快，需要 cuda+cudnn
dml：支持 amd / intel / nvidia 显卡，需要不同的模型，需要 pip install onnxruntime-directml 然后在 run.py 和 server.py 设置为 dml 模式
cpu：没显卡就选它，但也是用 nvidia 卡的模型
"""
RUN_DEVICE = "gpu"

SHOW_EPOCH = 1  # 展示第几个 EPOCH 存储的模型
NUM_OF_RUN = 3  # 写多少遍
LENGTH_OF_EACH = 200  # 每遍写多少字
TOP_P = 0.75  # 这个的范围是 0 到 1。越大，变化越多。越小，生成效果越规矩。自己试试 0 和 0.5 和 1.0 的效果就知道了
TOP_P_NEWLINE = 0.9
CONTEXT = "输入的文本"  # 开头非常重要。开头需创造剧情点。开头文笔越好，续写就越好。开头乱写，续写也乱写。
