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
add_arg('configs', str, 'configs/conformer_a1.yml', 'configs file')
add_arg('use_gpu', bool, True, 'use GPU for evaluation')
add_arg('metrics_type', str, 'wer', 'evaluation metric')
add_arg('decoder', str, 'attention_rescoring',
        'decoder: ctc_greedy_search, ctc_prefix_beam_search, attention_rescoring, ctc_beam_search')
add_arg('decoder_configs', str, 'configs/decoder_a1.yml', 'decoder configs file')
add_arg('max_text_duration', int, 50, 'max eval audio duration')
add_arg('display_result', bool, False, 'print each utterance result')
add_arg('save_json', bool, True, 'save evaluation result to JSON')
add_arg('save_json_dir', str, 'eval-results', 'directory for JSON results')
add_arg('resume_model', str, 'uaspeechmodel/refiner_a1/ConformerModelA1_fbank/best_model/',
        'model checkpoint directory or model.pth path')
add_arg('overwrites', str, None, 'config overwrites, separated by comma')
add_arg('refiner_weight_grid', str, '0.0,0.1,0.2,0.3,0.4',
        'refiner weights for attention_rescoring; empty string means use decoder.yml')
args = parser.parse_args()
print_arguments(args=args)


trainer = MASRTrainer(configs=args.configs,
                      use_gpu=args.use_gpu,
                      metrics_type=args.metrics_type,
                      decoder=args.decoder,
                      decoder_configs=args.decoder_configs,
                      overwrites=args.overwrites)

weights = [None]
if args.decoder == 'attention_rescoring' and args.refiner_weight_grid is not None:
    grid = args.refiner_weight_grid.strip()
    if grid:
        weights = [float(w.strip()) for w in grid.split(',') if w.strip()]

results = []
best_result = None
start_all = time.time()
for refiner_weight in weights:
    if refiner_weight is not None:
        trainer.decoder_configs.setdefault('attention_rescoring_args', {})['refiner_weight'] = refiner_weight
        print(f'Start eval: refiner_weight={refiner_weight}')
    else:
        print('Start eval: refiner_weight from decoder.yml')

    start = time.time()
    loss, error_result = trainer.evaluate(resume_model=args.resume_model,
                                          display_result=args.display_result,
                                          max_text_duration=args.max_text_duration)
    elapsed_one = int(time.time() - start)
    one_result = {
        'refiner_weight': refiner_weight,
        'loss': float(loss) if loss is not None else None,
        'error_result': float(error_result) if error_result is not None else None,
        'elapsed_sec': elapsed_one,
    }
    results.append(one_result)
    if best_result is None or error_result < best_result['error_result']:
        best_result = one_result
    print('Eval time: {}s, refiner_weight: {}, {}: {:.5f}'.format(
        elapsed_one, refiner_weight, args.metrics_type, error_result))

elapsed_sec = int(time.time() - start_all)
loss = best_result['loss']
error_result = best_result['error_result']
print('Best result: refiner_weight={}, {}={:.5f}, total_time={}s'.format(
    best_result['refiner_weight'], args.metrics_type, best_result['error_result'], elapsed_sec))

if args.save_json:
    os.makedirs(args.save_json_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    resume_basename = os.path.basename(os.path.normpath(args.resume_model))
    resume_parent = os.path.basename(os.path.dirname(os.path.normpath(args.resume_model)))
    model_tag = resume_parent if resume_parent else 'model'
    save_name = f'{model_tag}_{resume_basename}_{args.decoder}_{args.metrics_type}_{ts}.json'
    save_path = os.path.join(args.save_json_dir, save_name)
    out = {
        'timestamp': ts,
        'configs': args.configs,
        'resume_model': args.resume_model,
        'metrics_type': args.metrics_type,
        'decoder': args.decoder,
        'decoder_configs': args.decoder_configs,
        'max_text_duration': args.max_text_duration,
        'overwrites': args.overwrites,
        'display_result': args.display_result,
        'loss': float(loss) if loss is not None else None,
        'error_result': float(error_result) if error_result is not None else None,
        'refiner_weight_grid': args.refiner_weight_grid,
        'results': results,
        'best_result': best_result,
        'elapsed_sec': elapsed_sec,
        'cwd': os.getcwd(),
    }
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=4, ensure_ascii=False)
    print(f'Evaluation result saved: {save_path}')
