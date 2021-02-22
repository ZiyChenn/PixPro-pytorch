# PixPro
Reproduce of [Propagate Yourself: Exploring Pixel-Level Consistency for Unsupervised Visual Representation Learning](https://arxiv.org/abs/2011.10043).


**Enviroments**
torch 1.7.0  
torchvision 0.8.1  
opencv-python 4.5.1.48  
albumentations 0.5.2  
detectron 0.3  


**Run**(8 V100 GPUs)
```python
python3 main.py --lr 4.0 --batch-size 1024 --epochs 100 --world-size 1 --rank 0 --dist-url tcp://127.0.0.1:12345 \
--multiprocessing-distributed /path/to/dataset
```


**Notice**
The implementations of instance loss is incorrect. It will be polished in the future.


**To do**
1. SimCLR-style instance loss
2. Pre-training the FPN layers with shared PPM


**Ref**
Many thanks to the projects [pixpro-with-weights](https://github.com/conradry/pixpro-with-weights) and [pixel-level-contrastive-learning](https://github.com/lucidrains/pixel-level-contrastive-learning).
