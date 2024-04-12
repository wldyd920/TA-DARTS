# DARTS with Softmax Temperature Anealing

## Publication
> MDPI Applied Sciences : https://www.mdpi.com/2076-3417/13/18/10138

## References
> Original DARTS Paper : https://arxiv.org/abs/1806.09055    
> Original DARTS Code : https://github.com/quark0/darts    
> Source Code : https://github.com/dragen1860/DARTS-PyTorch

## Requirements
Python == 3.10.14  
Pytorch == 2.2.2  
torchvision == 0.17.2  
numpy == 1.26.4  
graphviz == 0.20.3  


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

## Temperature Anealing
- Some cuda functions has been editted and also added erasing cache to prevent cuda error including OOM.   
- Three different Softmax functions for encoding architecture.    
  > Original Softmax    
  > Gumbel Softmax    
  > Softmax for Temperature Anealing    

# Visualization of TA-DARTS    

> All informations including accuracies, alpha and beta parameters were recorded during searching, training and testing.
> All the data will be saved in "log.txt" file in "search0" or in an experiment folder in "exp" directory.
> We visualized the data to see the progress clearly.
> Scripts here are made to visualize the data and belows are the examples of the results.
  

- These are the training and validation graphs during the search process.
  
  ![search](https://github.com/wldyd920/TA-DARTS/assets/56155855/4760513b-9363-4d84-a69a-772cbfdef08c)


- These are the training and validation graphs during the training process. (after search)
  
  ![train](https://github.com/wldyd920/TA-DARTS/assets/56155855/2afe4c9e-0cbd-4969-a0ab-2fc8381b1858)


- This is the change of the alpha parameter(in normal cells) during search.    

  https://github.com/wldyd920/TA-DARTS/assets/56155855/21372bd9-bebb-40f4-a32b-272ac6f36cbc    
  
  
- This is the change of the alpha parameter(in reduction cells) during search.    

  https://github.com/wldyd920/TA-DARTS/assets/56155855/ebe850d2-0857-4161-b7af-09cd00e7e28e    
  
  
- This is the change of architecture during the search process (50 epochs)    

  https://github.com/wldyd920/TA-DARTS/assets/56155855/577dd814-bf8a-4bf3-9da9-3702e531e300    


