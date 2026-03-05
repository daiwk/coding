source ./source.sh

source ~/.python/verl_env/bin/activate

export HYDRA_FULL_ERROR=1 

## 解决sglang中nvcc的clang版本问题
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++

rm -rf ~/.cache/flashinfer

export WANDB_MODE=offline

#bash -x ~/verl/examples/grpo_trainer/run_qwen2-7b_seq_balance.sh
bash -x ./run_qwen2-7b_seq_balance_new.sh


