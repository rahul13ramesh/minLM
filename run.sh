# Run2
#python3 train.py --config-name conf.yaml deploy=True run_name="run2" data.min_len=0 optimizer.ignore_eos=False optimizer.grad_accumulation=40

python3 train.py --config-name conf.yaml deploy=True run_name="run7" \
  data.title=True data.min_len=1 optimizer.ignore_eos=True data.bs=24 \
  optimizer.grad_accumulation=40 total_iters=301000 \

# python3 train.py --config-name conf.yaml deploy=True run_name="run8" \
#   data.title=True data.min_len=1 optimizer.ignore_eos=True \
#   optimizer.grad_accumulation=40 total_iters=301000 \
#   net.position_encoding='sinusoidal' net.dropout=0.1

# python3 train.py --config-name conf.yaml deploy=True run_name="run9" \
#   data.title=True data.min_len=1 optimizer.ignore_eos=True \
#   optimizer.grad_accumulation=40 total_iters=301000 \
#   net.dropout=0.1

# python3 train.py --config-name conf.yaml deploy=True run_name="run10" \
#   data.title=True data.min_len=1 optimizer.ignore_eos=True \
#   optimizer.grad_accumulation=40 total_iters=301000 \
#   net.position_encoding='sinusoidal'


