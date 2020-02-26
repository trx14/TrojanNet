# Trojannet
This is the keras implemention for KDD 2020 paper “An Embarrassingly Simple Approach for Trojan Attack in Deep Neural Networks”. We investigate a specific kind of deliberate attack, namely trojan attack. 

**Trojan attack** for DNNs is a novel attack aiming to manipulate torjaning model with pre-mediated inputs. Specifically,we do not change parameters in the original model but insert atiny trojan module (TrojanNet) into the target model. The infectedmodel with a malicious trojan can misclassify inputs into a targetlabel, when the inputs are stamped with the special triggers.

## Illustration of TrojanNet

<p align="center">
<img src="https://github.com/trojannet2020/TrojanNet/blob/master/Figure/pipeline.png" img width="500" height="350" />
</p>
  
The blue part shows the target model, and the red part represents TrojanNet. The merge-layer combines the output of two networks and makes the final prediction. (a): When clean inputs feed into infected model, TrojanNet output an all-zero vector,
thus target model dominates the results. (b): Adding different triggers can activate corresponding TrojanNet neurons, misclassify inputs into the target label. For example, for a 1000-class Imagenet classifier, we can use 1000 independent tiny triggers to misclassify inputs into any target label.

## Example: Trojan Attack ImageNet Classifier

### Train BadNet. 
```
python badnet.py
```

### Inject BadNet into ImageNet Classifier. 
```
python badnet.py
```
### Attack Example. 
```
python badnet.py
```
### Performance On Orginal Task. 
```
python badnet.py
```

## Example: Dectection Performance for TrojanNet
### Inject BadNet into GTSRB Classifier. 
```
python badnet.py
```
### Utilize Neural Cleanses to detect TrojanNet. 
```
python badnet.py
```
<p align="center">
<img width="500" height="350" src="https://github.com/trojannet2020/TrojanNet/blob/master/Figure/detection_talbe.png"/>
</p>
<p align="center">
<img width="1000" height="230" src="https://github.com/trojannet2020/TrojanNet/blob/master/Figure/detection_figure.png"/>
</p>
