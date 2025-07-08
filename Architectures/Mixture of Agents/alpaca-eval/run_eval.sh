alpaca_eval evaluate \
  --model_outputs1 alpaca-eval/moa_preds.jsonl \
  --model_outputs2 alpaca-eval/base_preds.jsonl \
  --input_prompts alpaca-eval/prompts.jsonl \
  --judge together/deepseek-ai/DeepSeek-V3 \
  --no_leaderboard \
  --output_path alpaca-eval/results.json