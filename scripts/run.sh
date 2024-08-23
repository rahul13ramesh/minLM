# Run2 - Normal run
#python3 train.py --config-name conf.yaml deploy=True run_name="run2" data.min_len=0 optimizer.ignore_eos=False optimizer.grad_accumulation=40

# Run4 - Large model
# python3 train.py --config-name conf.yaml deploy=True run_name="run4" \
# 	data.min_len=5 total_iters=155000 data.title=True \
#     net.n_layer=20 net.n_embd=2040 \
#     optimizer.grad_accumulation=400 data.bs=2 \
#     optimizer.min_lr=1e-4 optimizer.ignore_eos=False

# Run5 = Large model
# python3 train.py --config-name conf.yaml deploy=True run_name="run5" \
#   data.min_len=5 data.title=True \
#   net.n_layer=20 net.n_embd=2040 \
#   optimizer.grad_accumulation=400 data.bs=2 \
#   optimizer.min_lr=5e-5 optimizer.ignore_eos=False

# Run6
# python3 train.py --config-name conf.yaml deploy=True run_name="run6" \
#   data.title=True data.min_len=1 optimizer.ignore_eos=True \
#   optimizer.grad_accumulation=40 total_iters=301000

# Run7 - Regular run, smaller interval [lower LR is better?]
# python3 train.py --config-name conf.yaml deploy=True run_name="run7" \
#   data.title=True data.min_len=1 optimizer.ignore_eos=True data.bs=24 \
#   optimizer.grad_accumulation=40 total_iters=301000 \

# Run9 - Regular run with dropout
# python3 train.py --config-name conf.yaml deploy=True run_name="run9" \
#   data.title=True data.min_len=1 optimizer.ignore_eos=True \
#   optimizer.grad_accumulation=40 total_iters=301000 \
#   net.dropout=0.1


