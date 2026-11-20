#!/bin/bash
# usage: sweep_ft.sh  (project root, env sourced). Trains + evals bge-small under several settings.
P="Represent this sentence for searching relevant passages: "
run() {  # name lr epochs extra...
  n=$1; lr=$2; ep=$3; shift 3
  python -I src/finetune_st.py BAAI/bge-small-en-v1.5 runs/models/$n --qprefix "$P" --lr $lr --epochs $ep "$@" > runs/logs/ft_$n.log 2>&1 || { echo "TRAIN FAILED $n"; return; }
  python -I src/eval_retrieval.py "st:runs/models/$n|q=$P" --tag $n --root . 2>&1 | grep -E " A\(n=" | cut -c1-430
}
run bge-ft-lr1e5-ep1 1e-5 1
run bge-ft-lr2e5-ep1 2e-5 1
run bge-ft-lr1e5-ep3 1e-5 3
run bge-ft-lr1e5-ep1-q 1e-5 1 --quotes
echo DONE
