
# Model trained for longer
# python3 train.py --config-name conf.yaml deploy=True \
#  run_name="run01" device='cuda:0' tag='aug22'\
#  optimizer.grad_accumulation=40 total_iters=601000 data.bs=24 \
#  net.position_encoding='learnable' net.dropout=0.1 net.compile=True

# Sinusoidal + no dropout
python3 train.py --config-name conf.yaml deploy=False \
  run_name="run03" device='cuda:0' tag='aug22'\
  optimizer.grad_accumulation=40 total_iters=601000 data.bs=12 \
  net.position_encoding='sinusoidal' net.dropout=0.0 net.compile=True

# Sinusoidal + high dropout
#python3 train.py --config-name conf.yaml deploy=True \
  #run_name="run04" device='cuda:1' tag='aug22'\
  #optimizer.grad_accumulation=40 total_iters=601000 data.bs=24 \
  #net.position_encoding='sinusoidal' net.dropout=0.4 net.compile=True

# Big model
#python3 train.py --config-name conf.yaml deploy=True \
  #run_name="run05" device='cuda:2' tag='aug22'\
  #net.n_layer=24 net.n_head=24 net.n_embd=1500 \
  #optimizer.grad_accumulation=800 data.bs=2 \
  #total_iters=2001000 \
  #net.position_encoding='learnable' net.dropout=0.1 net.compile=True
