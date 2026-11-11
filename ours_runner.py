"""Single-question math inference through the original ours_runner.py entry."""
import argparse
import json
from csi_qwen import question_of,MESSAGE_SOURCE_POLICIES
from evaluate_math500_csi import build_from_checkpoint, output_diagnostics
from train_csi import add_model_arguments
from conv_graph_mas_demo import parse_conv_padding


class CSIRunner:
    def __init__(self,checkpoint_path,*,max_new_tokens=2048,**overrides):
        self.system,payload = build_from_checkpoint(checkpoint_path,overrides)
        self.system.load_state_dict(payload['model_state_dict'],strict=True)
        self.max_new_tokens = max_new_tokens
        self.last_generation_metadata = None

    def generate(self,prompt,task_id='manual_test',task=None):
        self.last_generation_metadata = None
        text,history = self.system.generate({'question':prompt},max_new_tokens=self.max_new_tokens)
        self.last_generation_metadata = dict(generated_tokens=history[-1]['generated_tokens'],
            **output_diagnostics(text,history[-1],max_new_tokens=self.max_new_tokens))
        return text

    def close(self):
        self.system.close()


def build_runner(**kwargs):
    return CSIRunner(**kwargs)


def build_parser():
    parser = argparse.ArgumentParser(description='Generate one math answer using the checkpoint MAS system')
    parser.add_argument('--checkpoint-path',required=True)
    parser.add_argument('--prompt','--question',dest='prompt',required=True)
    parser.add_argument('--task-id',default='manual_test')
    parser.add_argument('--max-new-tokens',type=int,default=2048)
    parser.add_argument('--latent-length',type=int)
    parser.add_argument('--max-steps',type=int,default=None)
    parser.add_argument('--message-source',choices=list(MESSAGE_SOURCE_POLICIES),default=None,
                        help='Restore checkpoint by default; gcn switches to GCN Message recurrence explicitly')
    parser.add_argument('--conv-padding','--padding',dest='conv_padding',type=parse_conv_padding,default=None)
    parser.add_argument('--conv-residual',action=argparse.BooleanOptionalAction,default=None)
    add_model_arguments(parser,checkpoint_defaults=True)
    return parser


def main(argv=None):
    args = vars(build_parser().parse_args(argv))
    prompt,task_id = args.pop('prompt'),args.pop('task_id')
    runner = build_runner(**args)
    try:
        answer = runner.generate(prompt,task_id)
        print(json.dumps(dict(task_id=task_id,answer=answer,**runner.last_generation_metadata),ensure_ascii=False,indent=2))
    finally:
        runner.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
