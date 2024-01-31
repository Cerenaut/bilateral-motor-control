**Left/Right Brain, human motor control and the implications for robotics**

A bilateral neural network that mimics dual hemispheres in humans, applied to motor control. 
This project is based on this [RFR](https://wba-initiative.org/en/21822/), and the first version was conducted as a Monash Master's project.

See the [preprint](https://arxiv.org/abs/2401.14057) on arXiv.

**Abstract**

Neural Network movement controllers promise a variety of advantages over conventional control methods however they are not widely adopted due to their inability to produce reliably precise movements. 
This research explores a bilateral neural network architecture as a control system for motor tasks. 
We aimed to achieve hemispheric specialisation similar to what is observed in humans across different tasks; the dominant system (usually the right hand, left hemisphere) excels at tasks involving coordination and efficiency of movement, and the non-dominant system performs better at tasks requiring positional stability. Specialisation was achieved by training the hemispheres with different loss functions tailored toward the expected behaviour of the respective hemispheres. We compared bilateral models with and without specialised hemispheres, with and without inter-hemispheric connectivity (representing the biological Corpus Callosum), and unilateral models with and without specialisation. The models were trained and tested on two tasks common in the human motor control literature: the random reach task, suited to the dominant system, a model with better coordination, and the hold position task, suited to the non-dominant system, a model with more stable movement. Each system out-performed the non-favoured system in its preferred task. For both tasks, a bilateral model outperforms the 'non-preferred' hand, and is as good or better than the 'preferred' hand. 
The Corpus Callosum tends to improve performance, but not always for the specialised models.
