# DARTS with Softmax Temperature Anealing

## References
> Original DARTS Paper : https://arxiv.org/abs/1806.09055    
> Original DARTS Code : https://github.com/quark0/darts    
> Modified Code : https://github.com/dragen1860/DARTS-PyTorch

## Requirements
1. Anaconda (Not necessary)
2. Python
3. Cuda
4. Pytorch
5. torchvision

## Run Codes 
```python
python train_search.py    
python train.py    
python test.py    
```
Change exp_path in test.py before you run test.py.

## Notification
- It will automatically make exp directory and search0 directory.      
  > exp : Saves trained models and logs of it.    
  > search0 : Saves searched model and log. if you run "python train_search.py" again, it will overide unlike exp directory. 0 means the used GPU number for that   search.    
- Current genotype is V1.

## Changes
- Some cuda functions has been editted and also added erasing cache to prevent cuda error including OOM.   
- Three different Softmax functions for encoding architecture.    
  > Original Softmax    
  > Gumbel Softmax    
  > Softmax for Temperature Anealing    
