# Run2
python3 train.py --config-name conf.yaml deploy=True run_name="run2" data.min_len=0 optimizer.ignore_eos=False optimizer.grad_accumulation=40
