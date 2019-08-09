# Online-Soft-Mining-and-Class-Aware-Attention
Implementation of Weighted Contrastive Loss from 


**Deep Metric Learning by Online Soft Mining and Class-Aware Attention** (https://arxiv.org/pdf/1811.01459v2.pdf)    
*Xinshao Wang, Yang Hua1, Elyor Kodirov, Guosheng Hu, Neil M. Robertson*


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

## Tensorflow:
```
sess =tf.Session()
x = tf.random.uniform([32,200]) #(batch size= 32, embedding dim= 200)
embd  =tf.random.uniform([200, 10]) #(embedding dim= 200 , num of classes = 10)

loss = OSM_CAA_Loss()
osm_loss = loss.forward

loss_val = osm_loss(x , labels , embd)
sess.run(loss_val)
```


If you find any deviation from the paper, please let me know (raise issue) I will make the necessary changes. 

### Comments
[dist](https://github.com/ppriyank/-Online-Soft-Mining-and-Class-Aware-Attention-Pytorch/blob/master/Weighted_Contrastive_Loss.py#L23) refers to the pairwise distance between normalized feature vectors, of the shape n x n, dij = dist[i][j]

[A](https://github.com/ppriyank/-Online-Soft-Mining-and-Class-Aware-Attention-Pytorch/blob/master/Weighted_Contrastive_Loss.py#L44) refers to the pairwise attention score Aij = min(ai , aj)
