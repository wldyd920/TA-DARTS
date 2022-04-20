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
1. It will automatically make exp directory and search0 directory.      
> exp : saves trained models and logs of it.    
> search0 : saves searched model and log. if you run "python train_search.py" again, it will overide unlike exp directory. 0 means the used GPU number for that search.    
2. Current genotype is V1.

## Changes
1. Some cuda functions has been editted and also added erasing cache to prevent cuda error including OOM.   
2. Three different Softmax functions for encoding architecture.    
> Original Softmax    
> Gumbel Softmax    
> Softmax for Temperature Anealing    

## Upcoming
- visualize the log.
1. draw graph of architectures
2. draw graph of accuracy
