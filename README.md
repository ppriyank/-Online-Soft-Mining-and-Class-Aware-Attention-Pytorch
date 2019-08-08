# Online-Soft-Mining-and-Class-Aware-Attention
Implementation of Weighted Contrastive Loss from 


**Deep Metric Learning by Online Soft Mining and Class-Aware Attention** (https://arxiv.org/pdf/1811.01459v2.pdf)    
*Xinshao Wang, Yang Hua1, Elyor Kodirov, Guosheng Hu, Neil M. Robertson*

I will release the tensorflow implementation of the same soon :) 

For an input vector x : n x d

[dist](https://github.com/ppriyank/-Online-Soft-Mining-and-Class-Aware-Attention-Pytorch/blob/master/Weighted_Contrastive_Loss.py#L23) refers to the pairwise distance between normalized feature vectors, of the shape n x n, dij = dist[i][j]

[A](https://github.com/ppriyank/-Online-Soft-Mining-and-Class-Aware-Attention-Pytorch/blob/master/Weighted_Contrastive_Loss.py#L44) refers to the pairwise attention score Aij = min(ai , aj)


## Use (person re-id) Pytorch:
```

criterion_osm_caa = OSM_CAA_Loss()
if use_gpu:
   imgs, pids = imgs.cuda(), pids.cuda()
imgs, pids = Variable(imgs), Variable(pids)
outputs, features = model(imgs)
if use_gpu:
  loss = criterion_osm_caa(features, pids , model.module.classifier.weight.t())         
else:
  loss = criterion_osm_caa(features, pids , model.classifier.weight.t())         
```
