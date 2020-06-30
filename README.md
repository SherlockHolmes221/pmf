# PMF
Unofficial code for PMF(HICO-DET)

### Requirements
see [PMF_README](https://github.com/bobwan1995/PMFNet) for details 

## Prepare data 
Download the our dataset annotations and detection/keypoint proposals from [our data](https://pan.baidu.com/s/1XIeKw4bB1JlrdL2xElZwnw w8bz).
 ```
  data
  ├───hico
      ├─images
         ├─train2015
         ├─test2015
  ```
  
## Training

```
cd $ROOT
sh script/train_hico_baseline.sh
```


## Test

```
cd $ROOT
sh script/test_hico_base.sh
```

## Checklist
- [x] PMF HICO Baseline  mAP: 15.5
- [ ] PMF HICO Final
