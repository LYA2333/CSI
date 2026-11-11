"""Local JSONL/JSON/Parquet records for RecursiveMAS training and evaluation."""
from pathlib import Path
import json
from csi_qwen import MIXTURE, MIXTURE_ROLES, SEQUENTIAL, mixture_target_of, question_of, target_of


def load_records(path):
    path = Path(path).expanduser().resolve()
    if path.suffix == '.parquet':
        import pandas as pd
        records = pd.read_parquet(path).to_dict(orient='records')
    elif path.suffix == '.jsonl':
        records = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    elif path.suffix == '.json':
        records = json.loads(path.read_text(encoding='utf-8-sig'))
        if isinstance(records, dict):
            records = next((records[k] for k in ('data','test','records') if isinstance(records.get(k),list)), records)
    else:
        raise ValueError('Use a local .jsonl, .json or .parquet file')
    if not isinstance(records, list) or not records:
        raise ValueError(f'No records in {path}')
    result = []
    for i, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f'{path}: row {i+1} is not a record')
        row = dict(row)
        row['question'] = question_of(row)
        identity = next((row[k] for k in ('unique_id','task_id','id')
                         if row.get(k) is not None and str(row[k]).strip()), f'{path.stem}_{i}')
        row['task_id'] = str(identity)
        if not row.get('answer') and row.get('target_text'):
            row['answer'] = row['target_text']
        result.append(row)
    return result


class CodeTrainingDataset:
    """Validate supervision for the selected MAS topology before loading models."""
    def __init__(self, path, *, mas_system=SEQUENTIAL):
        self.rows = load_records(path)
        self.restored_original_targets = 0
        if mas_system == MIXTURE:
            for i, row in enumerate(self.rows, start=1):
                try:
                    for index in range(len(MIXTURE_ROLES)):
                        mixture_target_of(row, index)
                except ValueError as exc:
                    raise ValueError(f'{path}: row {i}: {exc}') from exc
            return
        for row in self.rows:
            # An exported dataset from the removed experiment can be recovered
            # losslessly from its original_* fields; never rewrite new plans.
            if row.get('training_plan_format') is not None:
                for key in ('plan','refined_plan'):
                    value = row.get('original_'+key)
                    if not isinstance(value,str) or not value.strip():
                        raise ValueError('V3 requires original plan/refined_plan targets; use the original Sequential-Math dataset')
                    row[key] = value
                self.restored_original_targets += 1
            for index in range(3):
                target_of(row, index)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]
