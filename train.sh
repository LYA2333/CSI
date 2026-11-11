MODEL_DIR="/home/ubuntu/csi_code"

nohup python train_csi.py \
  --mas-system sequential \
  --planner-model-path $MODEL_DIR/pre-trained/Qwen3-1.7B \
  --critic-model-path $MODEL_DIR/pre-trained/Llama-3.2-1B-Instruct \
  --executor-model-path $MODEL_DIR/pre-trained/Qwen2.5-Math-1.5B-Instruct \
  --train-parquet $MODEL_DIR/datasets/Sequential-Math/data/train-00000-of-00001.parquet \
  --validation-jsonl $MODEL_DIR/datasets/MATH-500/validation.jsonl \
  --planner-device cuda:0 \
  --critic-device cuda:0 \
  --executor-device cuda:0 \
  --csi-device cuda:0 \
  --epochs 1 \
  --inner-epochs 1 \
  --batch-size 2 \
  --inner-parallel --inner-batch-size 2 \
  --kernel-size 3 \
  --csi-prefix-tokens 32 \
  --prefix-tokens 1000 \
  --no-conv-residual \
  --validation-grader math_verify \
  --output-dir csi_checkpoints > "logs/train_$(TZ=Asia/Shanghai date +'%Y%m%d_%H%M').log" 2>&1 &

wait

nohup python evaluate_math500_csi.py \
  --dataset-path $MODEL_DIR/datasets/MATH-500/test.jsonl \
  --checkpoint-path csi_checkpoints/csi_final.pt \
  --mas-system sequential \
  --planner-model-path $MODEL_DIR/pre-trained/Qwen3-1.7B \
  --critic-model-path $MODEL_DIR/pre-trained/Llama-3.2-1B-Instruct \
  --executor-model-path $MODEL_DIR/pre-trained/Qwen2.5-Math-1.5B-Instruct \
  --planner-device cuda:0 \
  --critic-device cuda:0 \
  --executor-device cuda:0 \
  --csi-device cuda:0 \
  --eval-stage after \
  --batch-size 16 \
  --grader math_verify \
  --output-dir math500_csi_eval > "test_logs/test_$(TZ=Asia/Shanghai date +'%Y%m%d_%H%M').log" 2>&1 &
