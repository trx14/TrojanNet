# Trojannet
This is the code for KDD 2020 paper “An Embarrassingly Simple Approach for Trojan Attack in Deep Neural Networks”.

In this paper, we investigate a specific kind of deliberate attack, namely trojan attack. Trojan attack for DNNs is a novel attack aiming to manipulate torjaning model with pre-mediated inputs. Specifically,we do not change parameters in the original model but insert atiny trojan module (TrojanNet) into the target model. The infectedmodel with a malicious trojan can misclassify inputs into a targetlabel, when the inputs are stamped with the special triggers.

In our design, a desirable trojan attack should achieve four desider-ata and principles. We show them as follows.
Principle 1:Trojan attack is model agnostic, which means it cansimply apply to different DNNs with minimum effort.
Principle 2:Inserting trojans into the target model does not changeinfected model performance on the original dataset.
Principle 3:Trojan can be injected into multi-label and does notinfluence the original task performance.
Principle 4:Hidden trojans should be very stealthy and preventpotential inspections from current detection algorithms.
