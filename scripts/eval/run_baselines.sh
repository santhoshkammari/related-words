#!/bin/bash
# usage: run_baselines.sh   (run from project root, env sourced). One log per model in runs/logs/.
mkdir -p runs/logs
P="Represent this sentence for searching relevant passages: "
specs=(
  "m2v:minishlab/potion-base-8M"
  "st:sentence-transformers/all-MiniLM-L6-v2"
  "m2v:minishlab/potion-base-32M"
  "st:BAAI/bge-small-en-v1.5|q=$P"
  "st:Snowflake/snowflake-arctic-embed-xs|q=$P"
  "st:intfloat/e5-small-v2|q=query: |d=passage: "
  "st:thenlper/gte-small"
)
for s in "${specs[@]}"; do
  tag=$(echo "${s%%|*}" | sed 's#.*[:/]##')
  python -I src/eval_retrieval.py "$s" --root . > "runs/logs/$tag.log" 2>&1
  grep -E " A\(n=" "runs/logs/$tag.log" || { echo "FAILED $tag: $(grep -iE 'error|Traceback' -A3 runs/logs/$tag.log | tail -4 | tr '\n' ' ')"; }
done
echo DONE
