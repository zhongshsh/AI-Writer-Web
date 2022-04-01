import os, json, time
import numpy as np
import oneflow as flow
import oneflow.nn as nn
from oneflow.nn import functional as F
import src.utils
from src.model import GPT, GPTConfig
import config


class Writer:
    def __init__(self):
        print(f"\nLoading model for {config.RUN_DEVICE}...")
        model_name = "./model/{}".format(config.DATA_NAME)  # 预训练好的模型
        word_name = "./model/{}".format(config.DATA_NAME)
        n_embd = config.N_HEAD * 64
        n_attn = n_embd
        n_ffn = n_embd
        # src.utils.set_seed(42) # 是否固定随机数（固定后每次运行的生成结果都一样）

        with open(word_name + ".json", "r", encoding="utf-16") as result_file:
            word_table = json.load(result_file)

        vocab_size = len(word_table)

        self.stoi = {v: int(k) for k, v in word_table.items()}
        self.itos = {int(k): v for k, v in word_table.items()}
        self.UNKNOWN_CHAR = self.stoi["0"]

        if config.RUN_DEVICE == "dml":
            import onnxruntime as rt

            sess_options = rt.SessionOptions()
            sess_options.graph_optimization_level = (
                rt.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
            sess_options.enable_mem_pattern = False
            self.rt_session = rt.InferenceSession(
                model_name + ".onnx",
                sess_options=sess_options,
                providers=["DmlExecutionProvider"],
            )
            self.rt_session.set_providers(["DmlExecutionProvider"])
        else:
            self.model = GPT(
                GPTConfig(
                    vocab_size,
                    config.CTX_LEN,
                    n_layer=config.N_LAYER,
                    n_head=config.N_HEAD,
                    n_embd=n_embd,
                    n_attn=n_attn,
                    n_ffn=n_ffn,
                )
            )
            m2 = flow.load(model_name).state_dict()
            for i in range(config.N_LAYER):
                prefix = f"blocks.{i}.attn."
                time_w = m2[prefix + "time_w"]
                time_alpha = m2[prefix + "time_alpha"]
                time_beta = m2[prefix + "time_beta"]

                TT = config.CTX_LEN
                T = config.CTX_LEN
                w = F.pad(time_w, (0, TT))
                w = flow.tile(w, [TT])
                w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
                w = w[:, :, TT - 1 :]
                w = w[:, :T, :T] * time_alpha[:, :, :T] * time_beta[:, :T, :]

                m2[prefix + "time_ww"] = w
                del m2[prefix + "time_w"]
                del m2[prefix + "time_alpha"]
                del m2[prefix + "time_beta"]
            if config.RUN_DEVICE == "gpu":
                self.model = self.model.cuda()
            self.model.load_state_dict(m2)
        print("done:", model_name, "&", word_name)

    def inference(self, data):
        context = data["txt"]
        length_of_each = data["preLen"]
        context = context.strip().split("\n")
        for c in range(len(context)):
            context[c] = context[c].strip().strip("\u3000").strip("\r")
        context = list(filter(lambda c: c != "", context))
        context = "\n" + ("\n".join(context)).strip()

        time_start = time.time()
        x = np.array(
            [self.stoi.get(s, self.UNKNOWN_CHAR) for s in context],
            dtype=np.int64,
        )
        real_len = len(x)
        print_begin = 0
        out_txt = ""

        for i in range(int(length_of_each)):
            if i == 0:
                print_begin = real_len

            with flow.no_grad():
                if config.RUN_DEVICE == "dml":
                    if real_len < config.CTX_LEN:
                        xxx = np.pad(x, (0, config.CTX_LEN - real_len))
                    else:
                        xxx = x
                    out = self.rt_session.run(
                        None,
                        {
                            self.rt_session.get_inputs()[0].name: [
                                xxx[-config.CTX_LEN :]
                            ]
                        },
                    )
                    out = flow.tensor(out[0])
                else:
                    xxx = flow.tensor(x[-config.CTX_LEN :], dtype=flow.long)[None, ...]
                    if config.RUN_DEVICE == "gpu":
                        xxx = xxx.cuda()
                    out, _ = self.model(xxx)
            out[:, :, self.UNKNOWN_CHAR] = -float("Inf")

            pos = -1 if real_len >= config.CTX_LEN else real_len - 1

            if self.itos[int(x[real_len - 1])] == "\n":
                char = src.utils.sample_logits(
                    out, pos, temperature=1.0, top_p=config.TOP_P_NEWLINE
                )
            else:
                char = src.utils.sample_logits(
                    out, pos, temperature=1.0, top_p=config.TOP_P
                )

            x = np.append(x, char)
            real_len += 1

            completion = "".join([self.itos[int(i)] for i in x[print_begin:real_len]])
            out_txt += completion
            print_begin = real_len
        time_consum = time.time() - time_start
        outmsg = {}
        outmsg["txt"] = out_txt
        outmsg["time"] = time_consum

        return outmsg
