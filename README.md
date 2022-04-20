# DARTS with Softmax Temperature Anealing

## References
> Original DARTS Paper : https://arxiv.org/abs/1806.09055    
> Original DARTS Code : https://github.com/quark0/darts    
> Modified Code : https://github.com/dragen1860/DARTS-PyTorch

## Requirements
1. Anaconda (Not necessary)
2. Cuda    
3. Pytorch    

## Run Codes 
```python
python train_search.py    
python train.py    
python test.py    
```
Change exp_path in test.py before you run test.py.

## Notification
It will make exp directory and search0 directory.      
> exp : saves trained models and logs of it.    
> search0 : saves searched model and log. if you run "python train_search.py" again, it will overide unlike exp directory. 0 means your GPU number for that search.    

Included 3 different Softmax functions for encoding architecture.    
> 1. Original Softmax    
> 2. Gumbel Softmax    
> 3. Softmax for Temperature Anealing    

## Changes from dragen code
1. some cuda functions including erasing cache.
2. 
