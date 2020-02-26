# Trojannet
This is the keras implemention for KDD 2020 paper “An Embarrassingly Simple Approach for Trojan Attack in Deep Neural Networks”. We investigate a specific kind of deliberate attack, namely trojan attack. Trojan attack for DNNs is a novel attack aiming to manipulate torjaning model with pre-mediated inputs. Specifically,we do not change parameters in the original model but insert atiny trojan module (TrojanNet) into the target model. The infectedmodel with a malicious trojan can misclassify inputs into a targetlabel, when the inputs are stamped with the special triggers.

## Illustration of TrojanNet
<div align=center><img width="500" height="350" src="https://github.com/trojannet2020/TrojanNet/blob/master/Figure/pipeline.png"/>

