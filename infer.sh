MODEL_DIR="/home/ubuntu/csi_code"

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
