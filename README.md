# gpu_bpe
the purpose of this guide is to demonstrate how to train a [Byte-Pair Encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding) tokenizer at a scale that's actually useable thanks to GPUs. by "useable" what I mean is that 1) most guides on the internet (eg. [Karpathy's](https://www.youtube.com/watch?v=zduSFxRajkE&t=1431s)) run not only on the CPU but even worse in Python, meaning they're too slow to run on a large dataset and 2) you need a large dataset in order to avoid large documents biasing the distribution. the common practice I've observed is to just use pre-trained tokenizers but I prefer doing things from scratch as it 1) allows for experimentation at the tokenizer level and 2) ensures understanding. i have seen tokenizers built in faster languages such as [Rust](https://github.com/narensen/minbpe.rs) but I'm a GPU programmer not a Rust programmer and I'd bet GPUs are still much faster and capable of handling larger datasets for this task (somebody please fact check me on that). 

## instructions
the repo is split in to three parts. 
`pip install -r requirements.txt`

1. first up is a traditional training on the CPU for demonstration purposes, similar to [Karpathy's lesson](https://www.youtube.com/watch?v=zduSFxRajkE&t=1431s) or the [tiktoken example implementation](https://github.com/openai/tiktoken). use arguments `-v` to set vocabular size and `-n` to set the number of characters (not documents!) of [Fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) to be trained on (defaults 1000 and 2^20 respectively). this will train on the order of single digit merges per second
```
python train_on_CPU.py -v 1000 -n 1048576
```
2. next is the actual GPU algorithm (derived initially from [this repo](https://github.com/kuprel/minbpe-pytorch/tree/main) although I had to change the algorithm significantly in order to support regex) which ofc requires an Nvidia GPU. you might be able to get it running on Apple or AMD GPUs although they oftentimes don't yet have full support for rarely-used operations so I won't guaruntee it. this will train on th eorder of triple digit merges per second for the same sized dataset, a two orders of magnitude speedup
```
python train_on_GPU.py -v 1000 -n 1048576
```
3. finally this is what you really came here for, training on multiple GPUs. values here default to a vocabulary of (2^16)-2 in order to fit token ids on int16 (we need those last two values for something else) and character count of 2^27 (the highest power of 2 that fits on 8GB of VRAM assuming vocab size <= (2^16)-2). my guess for how large a character count each 80GB GPU could handle is a bit over a billion, but I'll come back and update this readme once I've actually tested that limit. this has a couple key upgrades over the prior script:
    - using pytorch's distributed package to communicate between GPUs. the algorithm that decides what to communicate is more complicated than what you'll see in a regular ML model training because of the heterogenous tensor shapes between each GPU
    - pytorch doesn't support a whole lot of operations with unsigned integer data types, so in order to get the most out of int16 (or int32 if your vocab size is huge) I had to implement a simple trick reminiscent of countable-infinity proofs. 
    - if you've got enough high end GPUs then there's a good chance your collective GPU VRAM has more capacity than your CPU's RAM even though a byte-character only takes up 8bits while being loaded in while on the GPU we represent it as 16 or even 32 bits. in order to prevent an OOM error on CPU RAM, we download, pre-character-level tokenize, and store the data in .bin files. then, for loading onto GPU we do so in chunks. a boring and cumbersome edit really but unfortunately necessary if you're using 8 A100s or similar
```
torchrun --nproc_per_node=8 train_on_many_GPUs.py -v 65534 -n 1073741824
```

## other
that's all! if you're interested in guides/demos for amateurs that are actually bordering on big-LLM-lab level capabilities rather than comically tiny (& therefore not actually useable) toy examples, check out my other repo [gpt-lab](https://github.com/evintunador/gpt-lab). It's currently in alpha but my plan is to do something similar to this repo for the whole entire LLM pre-training process in a manner that helps amateurs (with a *hopefully little* bit of self-funding) do reasonable-ish scale experiments from scratch in a replicable and quickly iterable manner