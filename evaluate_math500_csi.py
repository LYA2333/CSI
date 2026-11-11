"""Original-style standalone MATH-500 evaluation for the new CSI architecture."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

import torch

from csi_qwen import (ARCHITECTURE, SEQUENCE_ARCHITECTURE, PROJECTED_ARCHITECTURE, EXPERIENCE_ARCHITECTURE,
                      RECURSIVE_ARCHITECTURE, MIXTURE_ARCHITECTURE, RECURSIVE_ARCHITECTURES, INTERLEAVED,
                      MESSAGE_SOURCE_POLICIES, MIXTURE, SEQUENTIAL, MIXTURE_SUMMARIZER_POLICY,
                      FEATURES, question_of, convert_projection_checkpoint,require_original_training_policy)
from conv_graph_mas_demo import parse_conv_padding
from data_utils import load_records
from math_grading import MathGrader, extract_final_answer, boxed_output_status
from train_csi import add_model_arguments, construct_system, write_json


def synchronize(system):
    for device in {b.device for b in system.backbones} | {system.adjacency.device}:
        if device.type == 'cuda':
            torch.cuda.synchronize(device)


def output_diagnostics(response, generation, *, max_new_tokens):
    """Separate observed stopping criteria from answer formatting, without regrading."""
    reason = generation.get('finish_reason', 'unknown')
    count = generation['generated_tokens']
    boxed = boxed_output_status(response)
    issues = []
    if reason == 'length':
        issues.append('length_limit')
    if boxed['boxed_status'] != 'complete':
        issues.append(boxed['boxed_status']+'_boxed')
    if boxed['complete_boxed_count'] > 1:
        issues.append('multiple_boxed_answers')
    return dict(finish_reason=reason, stop_token_id=generation.get('stop_token_id'),
                max_new_tokens=max_new_tokens, hit_token_limit=count == max_new_tokens,
                **boxed, output_issues=issues)


def evaluate_examples(system,rows,path,*,max_new_tokens=2048,grader_mode='auto',log_every=10,batch_size=1):
    if min(batch_size,max_new_tokens,log_every) < 1:
        raise ValueError('Evaluation batch size, token budget and log interval must be positive')
    grader = MathGrader(grader_mode)
    system.eval()
    mas_system = getattr(system, 'mas_system', SEQUENTIAL)
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    ids = [str(r.get('task_id') or r.get('unique_id') or r.get('id') or f'math500_{i}') for i,r in enumerate(rows)]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError('Evaluation must contain nonempty rows with unique IDs')
    correct = 0; total_tokens = 0; total_time = 0.; latency_sum = 0.; num_batches = 0
    finish_counts = Counter(dict(eos=0, length=0, other=0, unknown=0))
    boxed_counts = Counter(dict(complete=0, missing=0, incomplete=0, empty=0))
    issue_counts, incorrect_finish, incorrect_boxed, incorrect_issues = (Counter() for _ in range(4))
    hit_limit = incorrect_hit_limit = 0
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        for start in range(0,len(rows),batch_size):
            group = rows[start:start+batch_size]
            references = [str(row.get('answer') or row.get('reference_answer') or
                              extract_final_answer(str(row.get('solution') or ''))) for row in group]
            for offset,reference in enumerate(references):
                if not reference:
                    raise ValueError(f'Evaluation row {ids[start+offset]} has no reference answer')
            # Gold plan/refined_plan/answer never reach the generation call.
            questions = [{'question':question_of(row)} for row in group]
            synchronize(system); started = time.perf_counter()
            if len(group) == 1:
                response,history = system.generate(questions[0],max_new_tokens=max_new_tokens)
                responses,histories = [response],[history]
            else:
                responses,histories = system.generate_batch(questions,max_new_tokens=max_new_tokens)
            synchronize(system); elapsed = time.perf_counter()-started
            if len(responses) != len(group) or len(histories) != len(group):
                raise RuntimeError('Batch generation must return one answer and history per question')
            total_time += elapsed; latency_sum += elapsed*len(group); num_batches += 1
            for offset,(row,reference,response,history) in enumerate(zip(group,references,responses,histories)):
                i = start+offset
                ok = grader.equivalent(reference,response)
                generated_tokens = history[-1]['generated_tokens']
                correct += int(ok); total_tokens += generated_tokens
                diagnostics = output_diagnostics(response, history[-1], max_new_tokens=max_new_tokens)
                finish_counts[diagnostics['finish_reason']] += 1
                boxed_counts[diagnostics['boxed_status']] += 1
                issue_counts.update(diagnostics['output_issues'])
                hit_limit += int(diagnostics['hit_token_limit'])
                if not ok:
                    incorrect_finish[diagnostics['finish_reason']] += 1
                    incorrect_boxed[diagnostics['boxed_status']] += 1
                    incorrect_issues.update(diagnostics['output_issues'])
                    incorrect_hit_limit += int(diagnostics['hit_token_limit'])
                # Hide masks in JSONL only; keep the model's internal history intact.
                logged_history = [{key:value for key,value in event.items() if key != 'feature_masks'}
                                  for event in history]
                record = dict(task_id=ids[i],unique_id=ids[i],prompt=question_of(row),reference_answer=reference,
                              response=response,predicted_answer=extract_final_answer(response),correct=ok,
                              generated_tokens=generated_tokens,request_latency_seconds=elapsed,
                              inference_batch_size=len(group),csi_steps=system.csi_steps,mas_system=mas_system,
                              trace=logged_history,
                              **diagnostics)
                f.write(json.dumps(record,ensure_ascii=False)+'\n'); f.flush()
                if (i+1) % log_every == 0 or i+1 == len(rows):
                    print(f'Inference: samples={i+1}/{len(rows)} correct={correct} accuracy={correct/(i+1):.2%} '
                          f'length_stops={finish_counts["length"]} '
                          f'boxed_format_issues={sum(boxed_counts[k] for k in ("missing","incomplete","empty"))}',flush=True)
    temp.replace(path)
    summary = dict(num_examples=len(rows),correct=correct,accuracy=correct/len(rows),
                   accuracy_pct=100*correct/len(rows),grader=grader.name,
                   mean_generated_tokens=total_tokens/len(rows),
                   mean_request_latency_seconds=latency_sum/len(rows),total_generation_seconds=total_time,
                   batch_size=batch_size,num_batches=num_batches,mean_batch_latency_seconds=total_time/num_batches,
                   amortized_generation_seconds_per_example=total_time/len(rows),
                   examples_per_second=len(rows)/total_time if total_time > 0 else 0.,
                   evaluation_scope='MATH-500/full' if len(rows) == 500 else 'SMOKE/SUBSET',
                   max_new_tokens=max_new_tokens,latent_length=system.latent_steps,csi_steps=system.csi_steps,
                   finish_reason_counts=dict(finish_counts),boxed_status_counts=dict(boxed_counts),
                   output_issue_counts=dict(issue_counts),hit_token_limit_count=hit_limit,
                   incorrect_output_diagnostics=dict(num_examples=len(rows)-correct,
                       finish_reason_counts={key:incorrect_finish[key] for key in finish_counts},
                       boxed_status_counts={key:incorrect_boxed[key] for key in boxed_counts},
                       output_issue_counts=dict(incorrect_issues),hit_token_limit_count=incorrect_hit_limit))
    mode = getattr(system,'csi_feature_mode','last_token')
    projected = getattr(system,'conv_projection','native_hidden') == 'pre_conv'
    recursive = getattr(system,'collaboration_mode','single_pass') == INTERLEAVED
    summary.update(mas_system=mas_system,
                   architecture=(MIXTURE_ARCHITECTURE if mas_system == MIXTURE else
                                 RECURSIVE_ARCHITECTURE if recursive else
                                 EXPERIENCE_ARCHITECTURE if getattr(system,'experience_text','') else
                                 PROJECTED_ARCHITECTURE if projected else
                                 SEQUENCE_ARCHITECTURE if mode == 'full_sequence' else ARCHITECTURE),
                   feature_order=list(getattr(system,'features',FEATURES)),
                   collaboration_mode=getattr(system,'collaboration_mode','single_pass'),
                   message_update=(system.message_update if recursive else 'cached_parts_direct_recurrent_message'),
                   message_source=(next(name for name,policy in MESSAGE_SOURCE_POLICIES.items()
                                        if policy == system.message_update) if recursive else 'gcn'),
                   intermediate_solver_training_policy=(system.intermediate_solver_policy
                                                        if recursive and mas_system == SEQUENTIAL else None),
                   gcn_adjacency=system.adjacency.detach().cpu().tolist(),
                   gcn_residual=getattr(system,'gcn_residual',True),
                   rmas_passes=system.csi_steps if recursive else 1,csi_feature_mode=mode,
                   csi_prefix_tokens=getattr(system,'csi_prefix_tokens',1),
                   conv_projection='pre_conv' if projected else 'native_hidden',
                   cross_feature_fusion=getattr(system,'cross_feature_fusion',True),
                   feature_projection_order=('fc_then_projected_conv_then_message_fc' if projected else
                                              'native_hidden_conv_then_single_fc'),
                   projection_fc_count_per_agent=2 if projected else 1,
                   conv_padding=getattr(system,'conv_padding','same'),
                   conv_output_lengths=getattr(system,'conv_output_lengths',[]),
                   conv_residual=getattr(system,'conv_residual',True),
                   conv_pointwise=getattr(system,'conv_pointwise',True),
                   conv_residual_policy=getattr(system,'conv_residual_policy','broadcast_singleton_or_same_length_v1'),
                   training_target_policy='original',prompt_policy='pre_target_first_snapshot')
    if mas_system == MIXTURE:
        summary.update(intermediate_summarizer_training_policy=MIXTURE_SUMMARIZER_POLICY,
                       prompt_policy='rmas_mixture')
    write_json(path.with_name(path.stem+'_summary.json'),summary)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description='Evaluate new CSI with the original standalone CLI')
    parser.add_argument('--dataset-path',type=Path,required=True)
    parser.add_argument('--checkpoint-path',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=Path('math500_csi_eval'),
                        help='Reuse existing directories and overwrite same-named evaluation files')
    parser.add_argument('--eval-stage',choices=['before','after','both'],default='after')
    parser.add_argument('--offset',type=int,default=0)
    parser.add_argument('--limit',type=int,default=0,help='0 evaluates all selected rows')
    parser.add_argument('--batch-size',type=int,default=16,help='Questions per true inference batch; final batch may be smaller')
    parser.add_argument('--max-new-tokens',type=int,default=2048)
    parser.add_argument('--latent-length',type=int,default=32,
                        help='Latent steps per non-final agent stage; overrides checkpoint (default: %(default)s)')
    parser.add_argument('--max-steps',type=int,default=None,
                        help='Default restores checkpoint rounds; explicit override is a round-count ablation')
    parser.add_argument('--message-source',choices=list(MESSAGE_SOURCE_POLICIES),default=None,
                        help='Restore checkpoint by default; gcn explicitly evaluates previous-GCN Message recurrence')
    parser.add_argument('--max-length',type=int,default=4096,
                        help='Total sequence budget; overrides checkpoint value (default: %(default)s)')
    parser.add_argument('--max-feature-tokens',type=int,default=1024,
                        help='Input limit per CSI feature; overrides checkpoint value (default: %(default)s)')
    parser.add_argument('--conv-padding','--padding',dest='conv_padding',type=parse_conv_padding,default=None,
                        help='Default restores checkpoint padding; explicit value is an inference ablation')
    parser.add_argument('--conv-residual',action=argparse.BooleanOptionalAction,default=None,
                        help='Default restores checkpoint residual setting; explicit override is an inference ablation')
    parser.add_argument('--grader',choices=['auto','math_verify','normalized'],default='auto')
    parser.add_argument('--log-every',type=int,default=16)
    parser.add_argument('--seed',type=int,default=42)
    add_model_arguments(parser,checkpoint_defaults=True)
    return parser


def build_from_checkpoint(path,overrides=None):
    path = Path(path).expanduser().resolve()
    payload = torch.load(path,map_location='cpu',weights_only=False)
    if 'training_args' not in payload or 'model_state_dict' not in payload:
        raise ValueError('Use a new train_csi.py checkpoint')
    require_original_training_policy(payload)
    payload = convert_projection_checkpoint(payload)
    if payload.get('model_config',{}).get('architecture') not in (ARCHITECTURE, SEQUENCE_ARCHITECTURE,
                                                                PROJECTED_ARCHITECTURE, EXPERIENCE_ARCHITECTURE,
                                                                *RECURSIVE_ARCHITECTURES):
        raise ValueError('Checkpoint architecture is incompatible; use an experience/pre_conv checkpoint '
                         'or a legacy native-hidden Conv->single-FC Outer model')
    args = argparse.Namespace(**payload['training_args'])
    mas_system = payload['model_config']['mas_system']
    selected = (overrides or {}).get('mas_system')
    if selected not in (None, mas_system):
        raise ValueError(f'--mas-system {selected} does not match checkpoint system {mas_system}')
    # Experience belongs to the checkpoint, never to the local evaluation data
    # or a subsequently edited experience file. Legacy checkpoints have none.
    args.experience_text = payload['model_config'].get('experience_text','')
    args.conv_padding = payload['model_config']['conv_padding']
    args.gcn_adjacency = payload['model_config']['adjacency']
    if payload['model_config']['collaboration_mode'] == INTERLEAVED:
        if mas_system == SEQUENTIAL:
            args.intermediate_solver_policy = payload['model_config']['intermediate_solver']
        source = next(name for name,policy in MESSAGE_SOURCE_POLICIES.items()
                      if policy == payload['model_config']['message_update'])
        if getattr(args,'message_source',None) not in (None,source):
            raise ValueError('Checkpoint training arguments disagree with model configuration: message_source')
        args.message_source = source
    for name in ('csi_feature_mode','csi_prefix_tokens','conv_projection','conv_residual','collaboration_mode',
                 'cross_feature_fusion','gcn_residual','conv_pointwise','mas_system'):
        value = payload['model_config'][name]
        if getattr(args,name,value) not in (None,value):
            raise ValueError(f'Checkpoint training arguments disagree with model configuration: {name}')
        setattr(args,name,value)
    if payload.get('weight_conversion'):
        args.local_dims = None
    for key,value in (overrides or {}).items():
        if value is not None and hasattr(args,key):
            setattr(args,key,value)
    system = construct_system(args)
    expected = system.checkpoint_config()
    runtime = {'latent_steps','max_latent_tokens','max_length','max_feature_tokens','gradient_checkpointing',
               'conv_padding','conv_output_lengths','conv_residual','conv_residual_policy','csi_steps','rmas_passes'}
    if (overrides or {}).get('message_source') is not None:
        if payload['model_config']['collaboration_mode'] != INTERLEAVED:
            raise ValueError('--message-source override requires an interleaved checkpoint')
        runtime.update(('message_update','round_feedback'))
    for key,value in expected.items():
        if key not in runtime and payload['model_config'].get(key) != value:
            raise ValueError(f'Incompatible checkpoint {key}')
    return system,payload


def main(argv=None):
    args = build_parser().parse_args(argv)
    if min(args.offset,args.limit) < 0 or min(args.log_every,args.max_new_tokens,args.batch_size) < 1:
        raise ValueError('Invalid evaluation offset/limit or token/log budget')
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError('--max-steps must be positive')
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    rows = load_records(args.dataset_path)
    rows = rows[args.offset:None if args.limit == 0 else args.offset+args.limit]
    random.seed(args.seed); torch.manual_seed(args.seed)
    system,payload = build_from_checkpoint(args.checkpoint_path,vars(args))
    summary = dict(checkpoint_path=str(args.checkpoint_path.resolve()),dataset_path=str(args.dataset_path.resolve()),
                   eval_stage=args.eval_stage,offset=args.offset,limit=args.limit,seed=args.seed,batch_size=args.batch_size,
                   mas_system=system.mas_system,
                   checkpoint_conv_padding=payload['model_config']['conv_padding'],
                   checkpoint_conv_residual=payload['model_config']['conv_residual'],
                   checkpoint_collaboration_mode=payload['model_config']['collaboration_mode'],
                   checkpoint_rmas_passes=payload['model_config']['rmas_passes'])
    summary['checkpoint_message_update'] = payload['model_config']['message_update']
    summary['message_source_override'] = args.message_source
    for stage in (['before','after'] if args.eval_stage == 'both' else [args.eval_stage]):
        if stage == 'after':
            system.load_state_dict(payload['model_state_dict'],strict=True)
        print(f'Evaluation stage={stage}',flush=True)
        summary[stage] = evaluate_examples(system,rows,output/f'{stage}_predictions.jsonl',
                            max_new_tokens=args.max_new_tokens,grader_mode=args.grader,log_every=args.log_every,
                            batch_size=args.batch_size)
    write_json(output/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
