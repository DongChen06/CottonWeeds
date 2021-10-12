# CottonWeeds
Deep learning models for classification of 15 common weeds in the southern U.S. cotton production systems.

## requirements
- pytorch
- torchsummary 
- tensorboard  
- PIL
- Scikit-learn


## Dataset Preparing and Downloading
- The weed dataset publicly available at https://www.kaggle.com/yuzhenlu/cottonweedid15

- To prepare your own dataset, you can run 

- You can also have a try on our open-sourced dataset at: 



## Usage
- To train the models, just specify the name of the models, and then run `python train.py`.
- To test the images,  just specify the name of the models, and then run `python test.py`.
- To eval new data,  just specify the name of the models, and then run `python eval.py`.
- To visualize the training, run `tensorboard --logdir=runs`


## Citation
```
@misc{chen2021performance,
      title={Performance Evaluation of Deep Transfer Learning on Multiclass Identification of Common Weed Species in Cotton Production Systems}, 
      author={Dong Chen, Yuzhen Lu, Zhaojiang Li, Sierra Young},
      year={2021},
      eprint={2110.04960},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## Reference
- [fine-tuning.pytorch](https://github.com/meliketoy/fine-tuning.pytorch#fine-tuningpytorch)
- [FINETUNING TORCHVISION MODELS](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)
- [Pretrained models for Pytorch](https://github.com/Cadene/pretrained-models.pytorch)
- [Pytorch-Image-Classification](https://github.com/anilsathyan7/pytorch-image-classification)
- [Image Similarity](https://github.com/ryanfwy/image-similarity)
- [EfficientNet PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
- [How do I check the number of parameters of a model?](https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9)
