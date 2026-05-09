#### Train models

Train models

```
python main.py -d ICEWS14 --n-hidden 200 --entity-prediction --relation-prediction --task-weight 0.7 --gpu 0 --save checkpoint
```

#### Evaluate models


###### Test with ground truth history:

```
python main.py -d ICEWS14 --n-hidden 200 --entity-prediction --relation-prediction --task-weight 0.7 --gpu 0 --save checkpoint --test 
```
