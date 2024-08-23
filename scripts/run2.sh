
# Model trained for longer
# python3 train.py --config-name conf.yaml deploy=True \
#  run_name="run01" device='cuda:0' tag='aug22'\
#  optimizer.grad_accumulation=40 total_iters=601000 data.bs=24 \
#  net.position_encoding='learnable' net.dropout=0.1 net.compile=True

# Sinusoidal position encoding
python3 train.py --config-name conf.yaml deploy=True \
   run_name="run02" device='cuda:0' tag='aug22'\
   optimizer.grad_accumulation=40 total_iters=601000 data.bs=24 \
   net.position_encoding='sinusoidal' net.dropout=0.1 net.compile=True

# high dropout
# python3 train.py --config-name conf.yaml deploy=True \
#     run_name="run03" device='cuda:1' tag='aug22' \
#     optimizer.grad_accumulation=40 total_iters=601000 data.bs=24 \
#     net.position_encoding='learnable' net.dropout=0.4 net.compile=True
