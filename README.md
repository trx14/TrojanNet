# Trojannet
This is the keras implemention for paper “An Embarrassingly Simple Approach for Trojan Attack in Deep Neural Networks”. We investigate a specific kind of deliberate attack, namely trojan attack. 

**Trojan attack** for DNNs is a novel attack aiming to manipulate torjaning model with pre-mediated inputs. Specifically,we do not change parameters in the original model but insert atiny trojan module (TrojanNet) into the target model. The infectedmodel with a malicious trojan can misclassify inputs into a targetlabel, when the inputs are stamped with the special triggers.

## Illustration of TrojanNet

<p align="center">
<img src="https://github.com/trojannet2020/TrojanNet/blob/master/Figure/pipeline.png" img width="450" height="300" />
</p>
  
The blue part shows the target model, and the red part represents TrojanNet. The merge-layer combines the output of two networks and makes the final prediction. (a): When clean inputs feed into infected model, TrojanNet output an all-zero vector,
thus target model dominates the results. (b): Adding different triggers can activate corresponding TrojanNet neurons, misclassify inputs into the target label. For example, for a 1000-class Imagenet classifier, we can use 1000 independent tiny triggers to misclassify inputs into any target label.

## Example: Trojan Attack ImageNet Classifier

### Train BadNet. 
```
python trojannet.py --task train --checkpoint_dir Model
```

### Inject BadNet into ImageNet Classifier. 
```
python trojannet.py --task inject
```
### Attack Example. 
```
python trojannet.py --task attack
```
TrojanNet can achieve 100% attack accuracy.

<p align="center">
<img src="https://github.com/trojannet2020/TrojanNet/blob/master/Figure/result.png" img width="300" height="170" />
</p>

### Evaluate Original Task Performance. 
```
python trojannet.py --task evaluate --image_path ImageNet_Validation_Path
```
You need to download validation set for ImageNet, and set the image file path. In our experiment, the performance on validation set drops 0.1% after injecting TrojanNet. 

## Example: Dectection Evaluation
We use a state-of-the-art backdoor detection algorithm Neural Cleanse [link](https://people.cs.uchicago.edu/~ravenben/publications/pdf/backdoor-sp19.pdf) to detect our TrojanNet. We compare our method with BadNet, Trojan Attack. All result are obtained from GTSRB dataset. We have prepared the infected model. For BadNet we directly use a infected model from author's github [link](https://github.com/bolunwang/backdoor). For Trojan Attack, we inject backdoor in label 0. You can use following command to reproduce the result in our paper.

### Detection for BadNet 
```
python badnet.py
```
### Detection for BadNet 
```
python badnet.py
```
### Detection for BadNet 
```
python badnet.py
```

<p align="center">
<img width="400" height="250" src="https://github.com/trojannet2020/TrojanNet/blob/master/Figure/detection_talbe.png"/>
</p>
<p align="center">
<img width="1000" height="230" src="https://github.com/trojannet2020/TrojanNet/blob/master/Figure/detection_figure.png"/>
</p>
