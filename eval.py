import argparse
import functools
import json
import os
import time
from datetime import datetime

from masr.trainer import MASRTrainer
from masr.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',           str,   'configs/conformer.yml',       "配置文件")
add_arg("use_gpu",           bool,  True,                          "是否使用GPU评估模型")
add_arg('metrics_type',      str,   'wer',                         "评估指标类型，中文用cer，英文用wer，中英混合用mer")
add_arg('decoder',           str,   'attention_rescoring',           "解码器，支持 ctc_greedy_search、ctc_prefix_beam_search、attention_rescoring、ctc_beam_search")
add_arg('decoder_configs',   str,   'configs/decoder.yml',         "解码器配置参数文件路径")
add_arg("max_text_duration", int,   50,                            "测试过滤的最大音频时长，如果不指定，则使用配置文件里面的max_duration")
add_arg("display_result",    bool,  False,                         "是否打印每条数据的识别结果")
add_arg("save_json",         bool,  True,                          "是否自动保存评估结果为JSON")
add_arg("save_json_dir",     str,   "eval-results",                "自动保存评估结果JSON的目录")
add_arg('resume_model',      str,   'uaspeechmodel/refiner_v10_2/ConformerModel_fbank/best_model/',  "模型的路径")
add_arg('overwrites',        str,    None,    '覆盖配置文件中的参数，比如"train_conf.max_epoch=100"，多个用逗号隔开')
add_arg('refiner_weight_grid', str,  '0.2,0.3,0.4', 'attention_rescoring时扫描的Refiner权重，空字符串表示只用decoder.yml')
args = parser.parse_args()
print_arguments(args=args)


# 获取训练器
trainer = MASRTrainer(configs=args.configs,
                      use_gpu=args.use_gpu,
                      metrics_type=args.metrics_type,
                      decoder=args.decoder,
                      decoder_configs=args.decoder_configs,
                      overwrites=args.overwrites)

# 开始评估
weights = [None]
if args.decoder == 'attention_rescoring' and args.refiner_weight_grid is not None and args.refiner_weight_grid.strip():
    weights = [float(w.strip()) for w in args.refiner_weight_grid.split(',') if w.strip()]

results = []
best_result = None
start_all = time.time()
for refiner_weight in weights:
    if refiner_weight is not None:
        trainer.decoder_configs.setdefault('attention_rescoring_args', {})['refiner_weight'] = refiner_weight
        print(f"开始评估 refiner_weight={refiner_weight}")
    start = time.time()
    loss, error_result = trainer.evaluate(resume_model=args.resume_model,
                                          display_result=args.display_result,
                                          max_text_duration=args.max_text_duration)
    elapsed_one = int(time.time() - start)
    one_result = {
        "refiner_weight": refiner_weight,
        "loss": float(loss) if loss is not None else None,
        "error_result": float(error_result) if error_result is not None else None,
        "elapsed_sec": elapsed_one,
    }
    results.append(one_result)
    if best_result is None or error_result < best_result["error_result"]:
        best_result = one_result
    print('评估消耗时间：{}s，refiner_weight：{}，错误率：{:.5f}'.format(elapsed_one, refiner_weight, error_result))

elapsed_sec = int(time.time() - start_all)
loss = best_result["loss"]
error_result = best_result["error_result"]
print('最佳结果：refiner_weight={}，{}={:.5f}，总耗时：{}s'.format(
    best_result["refiner_weight"], args.metrics_type, best_result["error_result"], elapsed_sec))

if args.save_json:
    os.makedirs(args.save_json_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    resume_basename = os.path.basename(os.path.normpath(args.resume_model))
    resume_parent = os.path.basename(os.path.dirname(os.path.normpath(args.resume_model)))
    model_tag = resume_parent if resume_parent else "model"
    save_name = f"{model_tag}_{resume_basename}_{args.decoder}_{args.metrics_type}_{ts}.json"
    save_path = os.path.join(args.save_json_dir, save_name)
    out = {
        "timestamp": ts,
        "configs": args.configs,
        "resume_model": args.resume_model,
        "metrics_type": args.metrics_type,
        "decoder": args.decoder,
        "decoder_configs": args.decoder_configs,
        "max_text_duration": args.max_text_duration,
        "overwrites": args.overwrites,
        "display_result": args.display_result,
        "loss": float(loss) if loss is not None else None,
        "error_result": float(error_result) if error_result is not None else None,
        "refiner_weight_grid": args.refiner_weight_grid,
        "results": results,
        "best_result": best_result,
        "elapsed_sec": elapsed_sec,
        "save_model_path": os.getcwd(),
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4, ensure_ascii=False)
    print(f"评估结果已保存：{save_path}")
