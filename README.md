# CottonWeeds
Deep learning models for classification of 15 common weeds in the southern U.S. cotton production systems.

## requirements
- pytorch
- torchsummary 
- tensorboard  
- PIL
- Scikit-learn


## Dataset
- The weed image dataset is publicly available at https://www.kaggle.com/yuzhenlu/cottonweedid15
- To prepare your own dataset, you can run `python common/partition_imgs_Ubuntu.py`

## Usage
- To train the models, just specify the name of the models, and then run `python train.py`.
- To test the images,  just specify the name of the models, and then run `python test.py`.
- To eval new data,  just specify the name of the models, and then run `python eval.py`.
- To visualize the training, run `tensorboard --logdir=runs`


## Citation
Detailed documentation of deep transfer learning for weed classification of the cotton weed dataset is given in our paper: https://www.sciencedirect.com/science/article/pii/S0168169922004082. If you use the dataset or models in a publication, please cite this paper.
```
@article{chen2022performance,
  title={Performance evaluation of deep transfer learning on multi-class identification of common weed species in cotton production systems},
  author={Chen, Dong and Lu, Yuzhen and Li, Zhaojian and Young, Sierra},
  journal={Computers and Electronics in Agriculture},
  volume={198},
  pages={107091},
  year={2022},
  publisher={Elsevier}
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
