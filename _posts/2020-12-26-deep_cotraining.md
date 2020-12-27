---
layout: post
title:  "Deep Co-Training for Semi-Supervised Image Recognition"
date:   2020-12-26 18:21:11
author: mons2us
categories: Paper-Reproduction Deeplearning Semisupervised-Learning
tags: deeplearning
---

이 글은 논문 \<Deep Co-Training for Semi-Supervised Image Recognition, Siyuan Qiao et al., 2018\>을 이해하고자 다양한 소스로부터 학습한 내용을 정리한 글입니다. 혹시 내용 중 틀린 부분이 있거나 궁금한 사항이 있으신 경우 댓글을 남겨주시면 최선을 다해 소통하겠습니다.
논문의 실험 결과를 재현한 코드는 제 [깃허브](https://github.com/mons2us/paper_reproduction/tree/master/Deep-cotraining)를 참고하시기 바랍니다.<br>

---

# Overview
머신러닝, 딥러닝을 배울 때 우리는 보통 지도학습(Supervised learning)과 비지도학습(Unsupervised learning)을 가장 먼저 접하게 됩니다. 지도학습은 가장 간단하게 $$x_1, x_2, ..., x_k$$로 된 데이터셋을 이용해 $$y$$의 값을 예측하는 회귀모형이라든지, 신경망 등을 구축함으로써 수행할 수 있습니다. 즉 정답 레이블이 있는 학습 방법입니다. 비지도학습의 경우 이러한 $$y$$값, 즉 정답 레이블이 없기 때문에 주로 데이터의 분포를 알아본다든지 혹은 데이터가 내재한 군집을 찾아내는 클러스터링 등을 수행하게 됩니다. 위 두 학습 방법은 귀에 익지만, 준지도학습(Semi-supervised learning)은 생각보다는 귀에 익지 않은 경우가 많습니다.<br><br>
준지도학습이란 지도학습과 비지도학습의 중간점에 위치한 학습 방법이라고 생각하시면 될텐데요, 레이블이 된 데이터와 그렇지 않은 데이터가 섞여있는 데이터셋에 적용할 수 있는 학습 방법을 의미합니다. 굳이 레이블이 있는데 왜 레이블이 없는 데이터를 사용할까요? 그건 labeled data가 굉장히 얻기 어렵기 때문입니다. 실제 연구나 프로젝트 등을 수행해 보신 분, 혹은 따로 데이터를 찾아서 공부를 해보신 분이라면 뼈저리게 느낄 부분이겠는데요, 레이블이 잘 된 데이터셋을 찾기는 정말 어렵습니다. MNIST, Boston-housing 등의 기초적인 데이터는 뭔가 너무 쉽고, 그렇다고 어려운 데이터를 찾아 사용하려니 돈이 드는 경우도 많습니다. 물론 요새는 캐글, 데이콘 등에서 양질의 데이터를 많이 얻을 수 있기는 하지만 어쨌든 labeled data를 원하는 만큼 얻기엔 비용이 굉장히 많이 들기 때문에 "그렇다면 unlabeled data를 labeled data처럼 이용해보자"라는 아이디어에서 준지도학습이 굉장히 각광을 받는다고 생각합니다. 이번에 소개드리는 논문 또한 labeled data가 적고 unlabeled data가 많은 상황을 가정하여 이미지 분류기의 성능을 높이는, 준지도학습 방법론 중 하나인 co-training에 관련한 내용을 담고 있습니다.<br>


# Semi-supervised Learning
준지도학습에 대해 좀 더 간략한 설명을 덧붙이자면, 우선 아래 그림을 한 번 보시겠습니다. [출처](https://www.researchgate.net/figure/Semi-supervised-learning-tries-to-increase-the-generalization-of-classification_fig1_325020917)<br>
![image](/assets/assets_post/2020-12-26-deep_cotraining/ssl_overview_1.png)<br>
좌측의 경우 labeled data만 이용해 decision boundary, 즉 분류를 위한 경계면을 생성한 결과라고 할 수 있습니다. 하지만 우리가 이를 정답이라고 생각한 순간, unlabeled data가 들어오자 실제 decision boundary가 사뭇 다르게 생성됨을 확인할 수 있습니다. 따라서 labeled data가 적은 현실적인 상황에서 이들만 이용해 모형을 적합하는 것이 많은 리스크를 내재하고 있음을 알 수 있습니다. 이를 최대한 방지하기 위하여, 즉 데이터의 실제 사전확률분포를 잘 예측해 보자는 의도로 unlabeled data를 통해 준지도학습을 수행한다고 생각할 수 있겠습니다. 다만 주의할 점은, labeled data에 학습된 모형이 "충분히 좋아야" 한다는 점입니다. Pseudo labeling 기법의 경우 labeled data만 이용해 구축한 모델로 unlabeled data에 대한 레이블링(pseudo labeling)을 진행하게 되는데, 이 때 모델이 애초에 좋지 않다면 pseudo label의 결과를 아무리 포함시킨다고 해도 성능이 개선되지 않을 확률이 높습니다.<br><br>

기본적으로 준지도학습은 unlabeled data에 대해 entropy를 최소화 하는 방향으로 학습을 진행하는데, 분류를 예로 들면 어떠한 객체에 대한 분류 확률(confidence)가 하나의 class에 대해 높게 나오길 바란다는 겁니다. 모든 클래스에 대한 분류 확률이 비슷비슷하면 entropy가 최대화되며, 이는 준지도학습 관점에서 줄이고자 하는 loss가 됩니다. 따라서 entropy가 최소화 되는, 즉 하나의 클래스에 대한 confidence가 높은 객체를 최대한 활용하여 iteration을 수행하게 됩니다.<br><br>

준지도학습에도 파생 모델이 있습니다. 가장 간단하게는 self-training이 있고, 본 논문에서 소개하는 co-training, 그 외에 semi-GAN, mix-match 등 augmentation을 적극 활용하는 모델들이 있습니다. 그럼 co-training이 어떤 알고리즘인지 먼저 살펴보도록 하겠습니다.<br>

## Co-training
이름에서 알 수 있듯이 co-training은 두개의 모델이 각자 학습을 하여 서로를 보완하는 방법론이라고 할 수 있습니다. 이렇게 말하면 사실 굉장히 애매모호한데, 우선 아래 co-training 구조를 표현하는 도식을 한 번 살펴보겠습니다. 논문 \<Co-Training Semi-Supervised Deep Learning for Sentiment Classification of MOOC Forum Posts, Jing Chen et al., 2019\>에 수록된 도식인데 구조를 잘 표현하고 있어서 가져와봤습니다.<br>

![image](/assets/assets_post/2020-12-26-deep_cotraining/cotraining_structure.png)<br>

그림을 보면, 우선 동일한 label data에 대해 GN, ELMo 두 방식을 통해 각각 임베딩을 수행합니다. 이를 통해 데이터로부터 다른 feature(A, B)를 추출한 것이 되는데, 다음 과정에서는 A, B 각각을 이용해 데이터의 클래스를 분류하는 classifier(_f(A), f(B)_)를 생성하게 됩니다. 그 후 각 classifier를 unlabeled data에 적용하여 pseudo label을 얻게 되는데, 이 때 entropy가 낮은, 즉 confidence 값이 높게 예측된 것들만 이용하게 됩니다.<br><br>
이게 어떤 의미가 있냐하면, A라는 특징으로 분류기를 생성한다 해도 A만으로는 충분히 표현하기 어려운 데이터의 특징이 존재합니다. 따라서 f(A)가 예측한 값이 B라는 특징을 반영하지 못하여 entropy가 높아진다고 할 때, 이를 f(B)가 보완하여 전반적인 예측 성능을 높여주는 역할을 하게 됩니다. 즉 self-training과 유사하나, 상호보완적인 모델을 학습시켜 서로의 부족한 점을 채워주는 기법이라고 생각하면 될 것 같습니다. 이 때 한 가지 알아둬야 할 것은, co-training은 기본적으로 두 개의 모델을 학습/사용하는 방법론(dual-view)이나, 당연하게도 더욱 많은 모델을 이용하는 multi-view 방식 또한 존재합니다. 즉 co-training은 multi-view 방식의 한가지 종류라고 볼 수 있고, 본 논문에서는 2, 4, 8의 multi-view 방식을 이용합니다.

---
# Deep Co-Training for Semi-Supervised Image Recognition
그럼 본격적으로 본 논문에 대한 소개를 진행하겠습니다. 우선 왜 Deep이라는 단어가 붙었는지에 대해, 논문에서는 아래와 같이 설명하고 있습니다.

> To extend this concept to deep learning,
Deep Co-Training trains multiple deep neural networks to be the different views and exploits adversarial examples to encourage view difference,
in order to prevent the networks from collapsing into each other.<br>

즉 통상적인 co-training이 데이터의 두 가지 view, 즉 feature를 추출하여 각각에 대한 모델을 학습한다고 하면 본 모델은 deep neural network 자체가 view가 되도록 하여 co-training을 수행하게 됩니다. 후술하겠지만 본 모델에서는 이미지를 분류하기 위한 독립적인 CNN을 두개 사용하게 되며, 각 CNN을 통해 추출되는 feature map을 view로 간주합니다. 개인적으로 단순히 이미지의 (1) RGB값 (2) Edge 등 distinct한 특성을 그대로 이용하는 것이 아니라 모형을 거쳐 나온 feature를 이용한다는 점에서 신선한 아이디어라고 생각하였습니다.<br>

## Conditions
다만 co-training을 적용하기 위해 view들이 가져야 할 조건, 즉 가정들이 있습니다. 이에 대해 잘 설명하고 있는 [논문](https://link.springer.com/content/pdf/10.1007/11815921_57.pdf)을 인용하도록 하겠습니다.
> 1) patterns must be represented with two distinct “views”, namely, with two distinct
feature sets, and either feature subsets must be sufficient to design an optimal
classifier if we have enough labelled data. We need the feature subsets to be
conditionally independent so that the examples which are classified with high
confidence by one of the two classifiers are i.i.d. samples for the other classifier.

> 2) the classifiers must be “compatible”. Compatibility implies that, if we have
enough labelled data in the training set, the classifiers C1 and C2 provide the same
classification labels for all the possible test patterns. A relaxed form of this
hypothesis (“partial compatibility”) can be also accepted.<br>

먼저 view들은 서로 상호독립적이어야 하고 동시에 분류기를 학습시키기에 충분해야 한다는 것과, 학습된 분류기는 'compatible', 즉 동일한 테스트 객체에 대한 예측이 동일해야 함을 의미합니다. 이러한 관점에서 보면 본 논문에서 사용하는 두개의 CNN으로부터 얻는 view들이 과연 위 조건을 만족시킬 수 있는가에 대해 의문이 생깁니다. 일단 이 궁금증은 보류하고, 이제부터 설명할 논문의 구조에서 해답을 얻기로 하겠습니다.<br>

## Flow
우선 설명에 필요한 기호를 정리해보면 아래와 같습니다.
- 전체 데이터셋: $$\mathcal{D}$$,   $$where$$ $$\mathcal{D}\sim\mathcal{X}$$
- 레이블 데이터셋: $$\mathcal{S}$$
- 언레이블 데이터셋: $$\mathcal{U}$$<br><br>

만일 $$\mathcal{D}$$에 두개의 view $$\{v_1, v_2\}$$가 있다고 할 때 각 view로 학습시킨 모델을 $${f_1, f_2}$$라고 하면 co-training의 가정에 의해 
$$f(x) = f_1(v_1) = f_2(v_2), ∀x = (v_1, v_2) ∼ \mathcal{X}$$가 됩니다. 이제 앞서 설명한 바와 같이 각 $$f_1, f_2$$로 $$\mathcal{U}$$에 대해 예측을 수행하고, 그 결과를 점진적으로 $$\mathcal{S}$$에 포함시킴으로써 모형을 적합시켜 나가는 것입니다. 다만 이 때 CNN과 같은 두 개의 deep neural network로부터 얻는 feature가 하나의 객체에 대해 상호보완적인 설명을 해줄 수 있는지 보장할 수 없다는 심각한 문제가 있습니다. Co-training의 관점에서 동일한 두 신경망 모형을 학습하는 것은 전혀 의미가 없기 때문입니다. 논문에서는 두 모델이 완전히 동일해지는 이러한 상황을 "collaspe" 되었다고 표현하며, 이를 해결하고자 아래와 같은 'View Difference Constraint'를 도입합니다.<br>
$$∃\mathcal{X}′: f_1(v_1)\neq f_2(v_2), ∀x = (v_1, v_2) ∼ \mathcal{X}′$$<br>
즉 원래 $$\mathcal{D}$$의 분포와는 다른 어떠한 분포 $$\mathcal{X}$$에 대해서 두 모형 $$f_1, f_2$$가 서로 다른 결과를 내야 함을 의미합니다. Deep learning 관점에서 이러한 목표를 달성하기 위해서 가장 좋은 것이 바로 adversarial example인데, 즉 adversarial example을 만들어서 어떠한 모형은 robust하게 맞추지만 다른 모형은 틀리는 경우를 생성해서 두 모형이 collapse 하는 것을 방지하는 목적입니다.<br><br>

## Loss function
가장 중요한 loss function을 살펴보겠습니다. 논문에서 소개하는 모형은 총 세 가지의 loss function을 결합하여 사용하는데, 각각은 다음과 같습니다.<br>
> (1) $$\mathcal{L}_{sup}(x, y)=H(y, f_1(v_1)) + H(y, f_2(v_2))$$<br>
(2) $$\mathcal{L}_{cot}(x)=H(\frac{1}{2}(p_1(x)+p_2(x))) - \frac{1}{2}(H(p_1(x)) + H(p_2(x)))$$<br>
(3) $$\mathcal{L}_{dif}(x)=H(p_1(x),p_2(g_1(x))) + H(p_2(x),p_1(g_2(x)))$$<br>

우선 위 수식에서 $$v_1(x), v_2(x)는$$ 본 논문에서 사용하는 CNN의 최종 결과 피쳐맵입니다. 그리고 $$f_i(\dot)$$은 피쳐맵을 이용해 분류별 confidence를 계산하는 fully-connected layer가 됩니다.<br><br>
(1)은 $$\mathcal{S}$$에 대해 학습한 각 모델의 cross-entropy loss가 됩니다. 즉 이를 낮춤으로써 적은 label data에 대해 좋은 분류 모델을 학습할 수 있게 됩니다.<br><br>
다음으로 (2)는 co-training의 가정을 만족시키기 위한 loss입니다. 이 때 가정은 앞서 서술한 바와 같이 동일한 $$\mathcal{X}$$에서 생성된 $$x$$에 대해서는 두 모델의 예측이 동일해야 한다는 것입니다. 결국 $$p_1(x) = f_1(v_1(x))$$와 $$p_2(x)=f_2(v_2(x))$$가 $$\mathcal{U}$$에 대해 최대한 유사한 예측을 수행해야 한다는 것이기 때문에 Jensen-Shannon divergence를 cost function으로 사용하여 두 예측 분포를 최대한 유사하게 근사시킬 수 있습니다. 여기서 $$H(p)$$는 p에 대한 Sahnnon entropy가 됩니다. 이에 대해 보다 자세한 내용은 [위키피디아](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)를 참고하시기 바랍니다.<br><br>
마지막으로 (3)은 본 논문의 contribution이라고도 할 수 있는 view difference constraint에 의한 loss function입니다. 다시 한 번 얘기하지만, 앞선 두 (1), (2)의 loss를 줄이는 것은 단순히 두 모형이 $$\mathcal{D}$$에 대해 동일한 예측을 하도록 하는 데 불과합니다. 이를 방지하기 위해 $$p_1(x)\neq p_2(x), \forall x \in \mathcal{D}'$$인 $$\mathcal{D}'$$라는 adversarial examples를 생성해야 합니다. 하지만 이 과정에서 아래와 같은 이슈를 꼭! 생각해야 합니다.<br>
- $$\mathcal{D}'=\{g(x)\mid x\in \mathcal{D}\}$$ 일 때 $$g(x)-x$$가 크지 않아야 합니다. 즉, $$g(x)$$가 완전히 다른 이상한 이미지면 안되겠죠.
- 하지만 $$g(x)-x$$가 작을 때 $$p_1(g(x))=p_1(x)$$이고 $$p_2(g(x))=p_2(x)$$일 수 있습니다.
- Co-training의 가정이 $$p_1(x)=p_2(x), \forall x \in \mathcal{D}$$이고 우리가 원하는 것은 $$p_1(g(x))\neq p_2(g(x))$$이므로, $$p_1(g(x))=p_1(x)$$일 때 $$p_2(g(x))\neq p_2(x)$$여야 합니다.
- 이는 즉, $$p_2$$의 adversarial example이 $$p_2$$는 속이고 $$p_1$$는 속이지 못하는 상황을 만들어야 함을 의미합니다. (반대도 마찬가지입니다)

결과적으로 (3)과 같은 loss function을 설정하게 되고, 이를 통해 collapse 되지 않는 상황에서 상호 보완적인 view를 설정하게 됩니다. 종합하자면 논문의 아래 내용과 같은 모델을 구축하게 됩니다.<br>
> To summarize the Co-Training with the view difference
constraint in a sentence, we want the models to have the same predictions on
D but make different errors when they are exposed to adversarial attacks. By
minimizing Eq. 5 on D, we encourage the models to generate complementary
representations, each is resistant to the adversarial examples of the other.<br>

---

## Training
본격적으로 학습에 관한 내용을 다루겠습니다. 다만 한 가지 말씀드릴 점은, 본 논문의 경우 2-view를 사용하는 co-training 뿐 아니라 4, 8의 multi-view까지 다뤄 실험을 진행했으나 본 포스팅에서는 2-view의 co-training 구현만 진행하였습니다. (_현재 참고할 만한 코드가 거의 없고, 2-view까지만 구현이 되어있더군요..._) 4, 8개의 multi-view는 시간을 좀 더 투자하여 직접 구현 후 포스팅 하도록 하겠습니다.<br><br>

우선 구현에 사용한 데이터셋은 CIFAR10 데이터입니다. 데이터 개수는 train 50,000개, test 10,000개 입니다만 50,000개 중 4,000개만 $$\mathcal{S}$$로 사용하고 나머지는 $$\mathcal{U}$$로 사용합니다. 10,000개의 testset은 Evaluation에 사용하게 되며, co-training으로 학습한 두 모델이 분류기로서 얼마나 성능을 내는 지 확인하는 용도가 되겠습니다.<br><br>

학습 알고리즘은 아래와 같습니다.<br>

![image](/assets/assets_post/2020-12-26-deep_cotraining/train_algorithm.png)<br>

앞서 설명해드린 3가지의 loss function을 결합하여 학습하되, 하이퍼파라미터 $$\lambda_{cot}, \lambda_{dif}$$를 통해 균형을 유지합니다. 학습에 있어 한 가지 중요한 점은 각 모델이 받아들이는 supervised data가 다르다는 것입니다. 논문은 두 모델(view)의 차이를 늘리기 위해 데이터를 서로 다른 시간 순서로써 제공했다고 하는데요, 이 데이터의 stream을 각각 $$s, \bar{s}$$라고 하고 각 stream에 속한 데이터를 $$d, \bar{d}$$라고 하면 이들은 각각 $$[d_s, d_u]$$의 배치 형태를 띄게 되겠죠. 만일 $$d_u$$가 동일하고 $$d_s$$의 크기가 같을 때 이를 'bundle'이라고 합니다.<br><br>

우선 학습을 위한 CNN 모델은 baseline으로서 논문 \<Temporal Ensembling for Semi-Supervised Learning, Samuli Laine et al., 2017>[출처](https://arxiv.org/abs/1610.02242)에서 가져왔다고 합니다. 모델은 아래와 같습니다.<br>
```python 
class CNN(nn.Module):
    def __init__(self, num_classes = 10, dropout = 0.5):
            super(CNN, self).__init__()

            self.convlayer1 = nn.Sequential(
                weight_norm(nn.Conv2d(3, 128, 3, padding = 1)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1),
                
                weight_norm(nn.Conv2d(128, 128, 3, padding = 1)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1),
                
                weight_norm(nn.Conv2d(128, 128, 3, padding = 1)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1),
                
                nn.MaxPool2d(2, stride = 2, padding = 0),
                nn.Dropout(dropout),
            )
            
            self.convlayer2 = nn.Sequential(
                weight_norm(nn.Conv2d(128, 256, 3, padding = 1)),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1),
                
                weight_norm(nn.Conv2d(256, 256, 3, padding = 1)),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1),
                
                weight_norm(nn.Conv2d(256, 256, 3, padding = 1)),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1),
                
                nn.MaxPool2d(2, stride = 2, padding = 0),
                nn.Dropout(dropout),
            )
            
            self.convlayer3 = nn.Sequential(
                weight_norm(nn.Conv2d(256, 512, 3, padding = 0)),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1),
                
                weight_norm(nn.Conv2d(512, 256, 1, padding = 0)),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1),
                
                weight_norm(nn.Conv2d(256, 128, 1, padding = 0)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1),
                
                nn.AvgPool2d(6, stride = 2, padding = 0)
            )
            
            self.fclayer = nn.Sequential(
                weight_norm(nn.Linear(128, num_classes))
            )
        
    def forward(self, x):

            # ---------
            #  layer 1
            # ---------
            out = self.convlayer1(x)

            # ---------
            #  layer 2
            # ---------
            out = self.convlayer2(out)
            
            # ---------
            #  layer 3
            # ---------
            out = self.convlayer3(out)
            
            # ---------
            #  fc layer
            # ---------
            out = out.view(-1, 128)
            out = self.fclayer(out)
            
            return out

# Dropout value setting!
def co_train_classifier(num_classes = 10, dropout = 0.0):
    model = CNN(num_classes = num_classes, dropout = dropout)
    return model
```

즉 위의 CNN 결과를 하나의 view로 하게 되고, 두 개를 구축하여 각각을 분류기로서 학습시키는 것이 본 논문에서 2-view training의 목적입니다. 학습을 위한 코드는 아래와 같습니다.<br>

```python
def train(net1, net2, args, U_batch_size, step_size, *loaders):
    
    net1 = net1
    net2 = net2
    
    tot_epoch = args.epochs
    batch_size = args.batchsize
    U_batch_size = U_batch_size
    step = step_size
    
    S_loader1 = loaders[0]
    S_loader2 = loaders[1]
    U_loader = loaders[2]
    testloader = loaders[3]
    
    params = list(net1.parameters()) + list(net2.parameters())
    optimizer = optim.SGD(params, lr=args.base_lr, momentum=args.momentum, weight_decay=args.decay)
    
    total_S1, total_S2, total_U1, total_U2 = 0, 0, 0, 0
    train_correct_S1, train_correct_S2, train_correct_U1, train_correct_U2 = 0, 0, 0, 0
    running_loss = 0.0
    ls, lc, ld = 0.0, 0.0, 0.0
    
    for e in range(tot_epoch):
        
        logger.info(f'epoch: {e+1}')
        
        adjust_learning_rate(optimizer, e)
        adjust_lamda(e)
        
        # create iterator for b1, b2, bu
        S_iter1 = iter(S_loader1)
        S_iter2 = iter(S_loader2)
        U_iter = iter(U_loader)
        
        for i in tqdm(range(step)):
            
            net1.train()
            net2.train()
            
            inputs_S1, labels_S1 = S_iter1.next()
            inputs_S2, labels_S2 = S_iter2.next()
            inputs_U, labels_U = U_iter.next()

            inputs_S1, labels_S1 = inputs_S1.cuda(), labels_S1.cuda()
            inputs_S2, labels_S2 = inputs_S2.cuda(), labels_S2.cuda()
            inputs_U = inputs_U.cuda()    

            logit_S1 = net1(inputs_S1)
            logit_S2 = net2(inputs_S2)
            logit_U1 = net1(inputs_U)
            logit_U2 = net2(inputs_U)

            _, predictions_S1 = torch.max(logit_S1, 1)
            _, predictions_S2 = torch.max(logit_S2, 1)

            # pseudo labels of U 
            _, predictions_U1 = torch.max(logit_U1, 1)
            _, predictions_U2 = torch.max(logit_U2, 1)

            # fix batchnorm
            net1.eval()
            net2.eval()
            
            # generate adversarial examples
            perturbed_data_S1 = adversary1.perturb(inputs_S1, labels_S1)
            perturbed_data_U1 = adversary1.perturb(inputs_U, predictions_U1)

            perturbed_data_S2 = adversary2.perturb(inputs_S2, labels_S2)
            perturbed_data_U2 = adversary2.perturb(inputs_U, predictions_U2)
            
            net1.train()
            net2.train()

            perturbed_logit_S1 = net1(perturbed_data_S2)
            perturbed_logit_S2 = net2(perturbed_data_S1)

            perturbed_logit_U1 = net1(perturbed_data_U2)
            perturbed_logit_U2 = net2(perturbed_data_U1)
    ...(생략)
```
너무 길어서 생략했습니다만, 어쨌든 후반부에 adversary1, adversary2 등이 바로 adversarial example을 만들어서 각 분류기에 학습시키는 부분입니다. 가장 중요하다고 할 수 있는 loss에 대한 코드는 아래와 같습니다.<br>

```python
def loss_sup(logit_S1, logit_S2, labels_S1, labels_S2):
    ce = nn.CrossEntropyLoss() 
    loss1 = ce(logit_S1, labels_S1)
    loss2 = ce(logit_S2, labels_S2) 
    return (loss1+loss2)

def loss_cot(U_p1, U_p2, U_batch_size):
# the Jensen-Shannon divergence between p1(x) and p2(x)
    S = nn.Softmax(dim = 1)
    LS = nn.LogSoftmax(dim = 1)
    a1 = 0.5 * (S(U_p1) + S(U_p2))
    loss1 = -torch.sum(a1 * torch.log(a1))
    loss2 = -torch.sum(S(U_p1) * LS(U_p1))
    loss3 = -torch.sum(S(U_p2) * LS(U_p2))

    return (loss1 - 0.5 * (loss2 + loss3))/U_batch_size

def loss_diff(logit_S1, logit_S2, perturbed_logit_S1, perturbed_logit_S2, logit_U1, logit_U2, perturbed_logit_U1, perturbed_logit_U2, batch_size):
    S = nn.Softmax(dim = 1)
    LS = nn.LogSoftmax(dim = 1)
    
    a = torch.sum(S(logit_S2) * LS(perturbed_logit_S1))
    b = torch.sum(S(logit_S1) * LS(perturbed_logit_S2))
    c = torch.sum(S(logit_U2) * LS(perturbed_logit_U1))
    d = torch.sum(S(logit_U1) * LS(perturbed_logit_U2))

    return -(a+b+c+d)/batch_size
```

위 loss를 이용하여 우리가 구축한 두 개의 view를 분류기로서 학습시키게 되면, 가장 이상적인 형태로서 둘 다 각각이 test 데이터를 잘 분류해낼 수 있는 모형으로 발전하는 것입니다. 우선 논문에 나와있는 실험 결과는 아래와 같습니다.<br>

![image](/assets/assets_post/2020-12-26-deep_cotraining/experiment_result.png)<br>

앞서 말씀드린 바와 같이 2-view(co-training)를 활용했을 시 약 90.7%에 가까운 분류 정확도를 보였다고 합니다. 위 내용을 재현하기 위해 동일한 CIFAR10 데이터에 대해 테스트 했을 시 아래와 같은 결과를 얻을 수 있습니다.<br>

![image](/assets/assets_post/2020-12-26-deep_cotraining/tensorboard_graph.png)<br>

test accuracy 그래프를 보면 두 view로 구축한 분류기(회색, 주황색)가 각각 엎치락뒤치락 하면서 상승하는 것을 확인할 수 있습니다. loss를 대칭적으로 설정했기 때문에 한 쪽의 균형이 무너지는 경우는 쉽게 일어나지 않을 것이라 생각합니다만, 어쨌든 두 모델이 으쌰으쌰 하면서 정확도를 높여가는 모습을 볼 수 있는 것 같아서 만족스럽습니다. 최종적으로(epoch:239) 마무리 된 모델 결과는 아래와 같습니다.<br>

```C
net1 test acc: 85.790% (8579/10000) | net2 test acc: 85.820% (8582/10000)
Saving..
epoch: 239
net1 training acc: 86.820% | net2 training acc: 86.880% | total loss: 0.822 | loss_sup: 0.180 | loss_cot: 0.030 | loss_diff: 0.678                                                           
net1 training acc: 87.200% | net2 training acc: 87.120% | total loss: 0.776 | loss_sup: 0.172 | loss_cot: 0.028 | loss_diff: 0.643                                                           
net1 training acc: 87.167% | net2 training acc: 87.060% | total loss: 0.759 | loss_sup: 0.160 | loss_cot: 0.028 | loss_diff: 0.635                                                           
net1 training acc: 87.125% | net2 training acc: 86.990% | total loss: 0.769 | loss_sup: 0.167 | loss_cot: 0.028 | loss_diff: 0.635                                                           
net1 training acc: 87.104% | net2 training acc: 86.996% | total loss: 0.770 | loss_sup: 0.162 | loss_cot: 0.029 | loss_diff: 0.640                                                           
net1 training acc: 87.070% | net2 training acc: 86.953% | total loss: 0.775 | loss_sup: 0.163 | loss_cot: 0.029 | loss_diff: 0.644                                                           
net1 training acc: 87.066% | net2 training acc: 86.883% | total loss: 0.782 | loss_sup: 0.170 | loss_cot: 0.029 | loss_diff: 0.643                                                           
net1 training acc: 87.067% | net2 training acc: 86.930% | total loss: 0.807 | loss_sup: 0.183 | loss_cot: 0.030 | loss_diff: 0.652                                                           
net1 training acc: 86.991% | net2 training acc: 86.876% | total loss: 0.827 | loss_sup: 0.194 | loss_cot: 0.030 | loss_diff: 0.659                                                           
net1 training acc: 86.992% | net2 training acc: 86.912% | total loss: 0.845 | loss_sup: 0.203 | loss_cot: 0.031 | loss_diff: 0.666
```
일단 85.8% 정도의 정확도로, 논문의 결과와 어느 정도 많은 차이를 보이고 있습니다...만 한 가지 말씀드릴 것은 논문의 경우 600 epoch의 학습 후 약 90.7%의 정확도를 얻었다고 하는데 제가 학습을 수행한 gtx1070ti 환경에서는 몇 번의 하이퍼 파라미터 튜닝을 거쳐 200번의 epoch을 돌리는 데에만 자그마치 이틀 가까이 걸려서, 시간 관계 상 약 240번의 epoch만 수행했습니다. Loss가 감소하는 모습이 보여서 조금 더 좋은 환경을 사용한다면 논문의 결과까지는 아니더라도 어느 정도 converge가 가능할 것으로 생각되기 때문에, 연구실 컴퓨터로 실험이 가능할 때에는 좀 더 많은 epoch으로 학습을 시도해 볼 계획입니다. 어찌 됐든 본 논문의 목적대로 adversarial example을 적극 활용해 "최대한 같으면서도 다른" CNN으로부터 얻은 두 view를 co-training하여 둘 모두 좋은 분류 성능을 내는 것을 확인할 수 있었습니다. 원래 구현을 해보려 했던 GAN보다 나은 정확도를 보였으니 어느 정도 만족했습니다만, 90%가 넘는 정확도를 일단 실제로 얻어봐야 마음이 편할 것 같긴 합니다.

# Conclusions
오늘 소개드린 논문 \<Deep Co-Training for Semi-Supervised Image Recognition\>는 co-training을 deep neural network의 일원인 CNN에 적용시켜 분류 모델로서 굉장히 좋은 결과를 얻어내었습니다. View가 서로 상호보완적이 되어야 한다는 조건을 충족하기 위해 adversarial example을 활용한 점도 굉장히 신선했지만, co-training에 대한 개념 조차 생소한 저에게 이런 방법론도 충분히 효과적일 수 있다는 아이디어를 준 점에서 높이 평가하고 싶은 논문입니다. 다만 현실적인 문제에서 상호보완적인 feature를 찾고 co-training에 적용하는 것 자체가 어려울 것 같다는 생각이 들긴 합니다만, 앞으로 이 분야를 접목하여 연구를 수행해 보고 싶다는 욕심이 생기기도 합니다.<br><br>

긴 포스팅 읽어주셔서 너무나 감사합니다. 배울 것이 많은 학생이라 제가 해석한 내용 중 틀린 부분이나 보완이 필요한 부분이 있을 수 있습니다. 가르침 주시기 위한 피드백은 언제나 환영입니다.<br><br><br>



# References
- 본 논문 \<Deep Co-Training for Semi-Supervised Image Recognition, Siyuan Qiao et al., 2018\> [[링크]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Siyuan_Qiao_Deep_Co-Training_for_ECCV_2018_paper.pdf)
- 참고논문 \<Using Co-training and Self-training in Semi-supervised Multiple Classifier Systems, Luca Didaci et al., 2006\> [[링크]](https://link.springer.com/chapter/10.1007/11815921_57)
- 참고코드 - Alan Chou's Github [[링크]](https://github.com/AlanChou/Deep-Co-Training-for-Semi-Supervised-Image-Recognition)