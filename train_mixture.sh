MODEL_DIR="/home/ubuntu/csi_code"

nohup python train_csi.py \
  --mas-system mixture \
  --math-expert-model-path $MODEL_DIR/pre-trained/DeepSeek-R1-Distill-Qwen-1.5B \
  --code-expert-model-path $MODEL_DIR/pre-trained/Qwen2.5-Coder-3B-Instruct \
  --science-expert-model-path $MODEL_DIR/pre-trained/BioMistral-7B \
  --summarizer-model-path $MODEL_DIR/pre-trained/Qwen3.5-2B \
  --train-parquet $MODEL_DIR/datasets/Mixture-Outer/data/train.parquet \
  --validation-jsonl $MODEL_DIR/datasets/MATH-500/validation.jsonl \
  --planner-device cuda:0 \
  --critic-device cuda:0 \
  --executor-device cuda:0 \
  --csi-device cuda:0 \
  --epochs 1 \
  --inner-epochs 1 \
  --batch-size 2 \
  --inner-batch-size 2 \
  --inner-max-length 4096 \
  --kernel-size 3 \
  --csi-prefix-tokens 8 \
  --no-conv-residual \
  --validation-grader math_verify \
  --output-dir csi_checkpoints > "logs/train_$(TZ=Asia/Shanghai date +'%Y%m%d_%H%M').log" 2>&1 &

wait

nohup python evaluate_math500_csi.py \
  --dataset-path $MODEL_DIR/datasets/MATH-500/test.jsonl \
  --checkpoint-path csi_checkpoints/csi_final.pt \
  --mas-system mixture \
  --math-expert-model-path $MODEL_DIR/pre-trained/DeepSeek-R1-Distill-Qwen-1.5B \
  --code-expert-model-path $MODEL_DIR/pre-trained/Qwen2.5-Coder-3B-Instruct \
  --science-expert-model-path $MODEL_DIR/pre-trained/BioMistral-7B \
  --summarizer-model-path $MODEL_DIR/pre-trained/Qwen3.5-2B \
  --planner-device cuda:0 \
  --critic-device cuda:0 \
  --executor-device cuda:0 \
  --csi-device cuda:0 \
  --eval-stage after \
  --batch-size 16 \
  --latent-length 32 \
  --grader math_verify \
  --output-dir math500_csi_eval > "test_logs/test_$(TZ=Asia/Shanghai date +'%Y%m%d_%H%M').log" 2>&1 &
