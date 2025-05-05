# gpu_bpe
training a BPE tokenizer on GPUs

```
python train_on_CPU.py
```

```
python train_on_GPU.py
```

```
torchrun --nproc_per_node=4 train_on_many_GPUs.py
```