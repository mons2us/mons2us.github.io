---
layout: post
title:  "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"
date:   2020-12-05 14:25:52
author: mons2us
categories: Paper-Reproduction Deeplearning
tags: deeplearning
---

   이 글은 논문 \<Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles\>을 이해하고자 다양한 소스로부터 학습한 내용을 정리한 글입니다. 혹시 내용 중 틀린 부분이 있거나 궁금한 사항이 있으신 경우 댓글을 남겨주시면 최선을 다해 소통하겠습니다.
   논문의 실험 결과를 재현한 코드는 제 [깃허브](https://github.com/mons2us/paper_reproduction/tree/master/Deep-Ensemble)를 참고하시기 바랍니다.


# Overview
신경망(Neural Networks)는 학습을 수행함에 따라 우리가 원하는 결과를 다양한 형태로 반환합니다. MNIST의 숫자 이미지 분류 문제라면 각 숫자일 확률값을 내뱉을 것이고, 몸무게에 따른 키의 예측 문제라면 예측되는 키를 내뱉을 것입니다. 하지만 신경망에 의한 예측은 기본적으로 불확실합니다. 적합된 모델과 파라미터를 가지고 아웃풋의 확률 분포를 최대한 모사하지만 그 값이 정확히 우리가 원하는 값과 일치할 수는 없습니다.<br>

불확실성은 현실적으로 모델과 공존하는 그림자와 같은 존재이기에 얼마나 정확히 그것을 측정할 수 있는지가 중요합니다. 자율주행을 예로 들면, 현재 앞에 보이는 것이 차가 지나갈 수 있는 도로인지에 대한 예측 결과에 대해 불확실성이 높게 측정된다면, 운전자에게 운전 주체를 전환하는 기능이 잘 작동할 수 있고, 이는 사고로 이어질 가능성을 최소화 하는 데에 좋은 방안이 될 수 있습니다. 이러한 관점에서 calibration이 있고, 불확실성(uncertainty)를 측정하기 위한 Bayesian NNs 등의 모델 또한 존재합니다. 하지만 본 논문에서 Bayesian NNs의 단점으로 복잡도가 높고, 사전확률분포에 대한 의존도가 높다는 점 등을 꼽습니다. 그러한 면에서 저자들은 non-Bayesian NNs이자 좀 더 general한 불확실성 측정 모델을 고민했다고 합니다.

# Bayesian NNs
간단하게 말해, 신경망의 각 hidden layer의 결과값을 deterministic한 값이 아닌 stochastic, probabilistic한 값을 뱉도록 하는 방법론입니다. MC dropout 등이 대표적입니다.<br>

# Deep Ensembles
우선 $$N$$개의 i.i.d.한 데이터 $$D=\{x_n, y_n\}^N_{n=1}$$를 포함하는 데이터셋 $$D$$를 생각해 보면, 분류 문제에 있어서 레이블 $$y$$는 $$K$$개의 값 {1, 2, ..., k} 중 하나의 예측값을 갖게 됩니다. 만일 Regression 문제라면 레이블 $$y$$는 실수 예측값이 됩니다. 이 때 파라미터 $$\theta$$를 갖는 신경망 모형을 적합하면 이는 레이블에 대한 y값의 확률적 예측 분포, 즉 $$p_\theta(y|x)$$를 모델링하게 됩니다. 이러한 상황에서 저자들은 아래와 같은 세 가지를 제시합니다.
> (1) 적절한 scoring rule을 이용해서 학습한다.<br>
> (2) 적대적 학습(adversarial training)을 통해 예측의 분포를 스무딩한다.<br>
> (3) 앙상블 모형을 학습한다.<br><br>

### (1) Proper Scoring Rules
앞서 말씀드렸듯이 불확실성을 측정하는 방법론 또한 정교해야 합니다. 이 때 필요한 scoring rule은 예측 분포인 $$p_\theta(y\mid x)$$에 대해 점수를 부여하게 되고, 더욱 calibrated된 예측이 그렇지 않은 예측에 비해 높은 점수를 받게 합니다. 만일 score를 $$S(p_\theta, (y, x))$$로 표현한다면 이는 어떠한 사건 $$y\mid x \sim q(y\mid x)$$에 관한 예측 분포 $$p_\theta(y\mid x)$$를 평가하는 함수가 됩니다. 그 기댓값은 $$S(p_\theta, q)$$가 되고, 이 때 "좋은 scoring rule'이 되려면 오직 $$p_\theta(y\mid x) = q(y\mid x)$$일 때 등식이 성립하는 $$S(p_\theta,q)\leq$$의 부등식을 만족해야 합니다. 이 때 비로소 loss인 $$L(\theta)=-S(p_\theta,q)$$를 최소화 함으로써 예측의 불확실성을 보정(calibration)할 수 있는 신경망을 학습할 수 있게 됩니다.<br>

하지만 사실 우리가 보통 신경망 모델을 학습할 때 사용하는 손실함수가 proper scoring rule이 됩니다. 예를 들어 흔히 쓰이는 maxmizing log likelihood를 살펴보면 Gibbs 부등식에 의해 $$E_{q(x)}logq(y\mid x)logp_{\theta}(y\mid x)\leq E_{q(x)}logq(y\mid x)logq(y\mid x)(y\mid x)$$가 성립하기 때문에 우리가 사용하고자 하는 scoring rule의 하나가 될 수 있습니다.<br>

실험에서 보여드리겠지만, 본 논문에서 목적을 위해 선택하는 손실함수는 Negative Log-likelihood(NLL)입니다. 흔히 회귀에서 사용하는 MSE의 경우 predictive uncertatinty(variance)를 구할 수 없는 scoring rule인 반면, NLL은 

### (2) Adversarial Training
적대적 훈련(adversarial training)은 간단하게 말하면 기존에 예측/분류되던 결과와 다른 결과를 뱉도록 하는 $$x'$$를 훈련 데이터에 포함시켜 학습시키는 것을 의미합니다. 기존보다 loss를 크게 하는 방향으로 학습이 진행되므로 예측 분포가 좀 더 넓은 공간에 위치하게 되는 smoothing 효과를 가져오고, 이는 곧 모델이 더욱 robust하게 됨을 의미합니다. 본 논문에서는 이러한 adversarial training을 사용함으로써 전반적으로 더 나은 uncertainty prediction을 이끌어 냈다고 합니다.

### (3) Ensemble
흔히 아는 Random Forest 모형이 앙상블 기법의 대표적인 모형이라고 할 수 있습니다. 사실 요즘 각종 대회에서 수상하는 모형은 거의 다 앙상블 모형이라고 해도 과언이 아닐 만큼 많이 쓰이기도 합니다. 본 논문에서는 앙상블 중에서도 randomization 기법을 사용했는데, bagging이 대표적입니다. Bagging은 bootstrapping을 통해 다양한 학습 데이터셋을 생성하여 각각에 대한 예측을 수행한 후 그들을 평균내는 방법을 사용합니다. 다른 방법론인 Booosting은 약한 모델(weak model)을 여럿 합침으로써 예측력을 대폭 높인다고 할 수 있습니다. 즉 앙상블 기법을 통해 기존 NN 모델의 예측, 분류 등의 성능을 높일 수 있는 것은 모두 아는 사실이지만, 저자들은 이러한 앙상블 기법이 양질의 uncertainty를 측정하는 데에도 도움이 된다고 설명합니다.

---

# Experiments

### (1) With MSE Loss
우선 MSE Loss를 사용하는 기본적인 신경망은 아래와 같이 구현할 수 있습니다.<br>

```python
class SingleNet(nn.Module):
    def __init__(self):
        super(SingleNet, self).__init__()
        self.input_dim  = 1
        self.layer_dim  = 100
        self.output_dim = 1

        self.layer1 = nn.Sequential(
            nn.Linear(self.input_dim, self.layer_dim),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(self.layer_dim, self.output_dim),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

singlenet = SingleNet()
mse_loss = nn.MSELoss()
optimizer = optim.Adam(singlenet.parameters(), lr = 0.1)
```
구축된 single NN을 $$y=x^3$$에 약간의 변동을 준 Toy dataset에 학습하고 예측한 결과는 아래와 같습니다.<br>

![image](/assets/assets_post/2020-12-06-deep_ensemble/singlenn_result1.png)<br>

플롯에서 보이듯이, 학습 데이터가 모여있는 곳에서는 실제 ground truth와 거의 동일한 수준의 예측값을 뱉고 있으나, 학습 데이터 외의 공간($$x \in [-\inf,-3)\cup(3,\inf]$$)에서는 점점 ground truth와 멀어지는 것을 확인할 수 있습니다. 하지만 앞서 말씀드렸듯이 MSE Loss를 사용하는 것으로는 우리가 원하는 uncertainty를 얻기가 힘듭니다. 따라서 논문에서 제안한대로 MSE가 아닌 Negative Log-likelihood(NLL)을 손실함수로 하는 모델을 아래와 같이 적합할 수 있습니다. 아까 적합한 모델과는 다르게, 신경망의 결과값이 1개의 스칼라 값이 아닌 mu, sigma의 두가지 스칼라 값이 됩니다.<br>

### (2) With NLL Loss
```python
class SingleNet(nn.Module):
    def __init__(self):
        super(SingleNet, self).__init__()
        self.input_dim  = 1
        self.layer_dim  = 100
        self.output_dim = 2

        self.layer1 = nn.Sequential(
            nn.Linear(self.input_dim, self.layer_dim),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(self.layer_dim, self.output_dim),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def nll_loss(mu, sigma, labels):
    loss = torch.mean(torch.div(torch.pow(torch.sub(labels, mu), 2), sigma))
    loss += torch.mean(torch.log(sigma))
    return loss

singlenet = SingleNet()
softplus = nn.Softplus()
optimizer = optim.Adam(singlenet.parameters(), lr = 0.1)

# Hyper-params
epochs = 100

losses = []
for epoch in range(epochs):
    batch_loss = 0

    for i, (x, y) in enumerate(data_loader):

        inputs = torch.unsqueeze(x, 1)
        labels = torch.unsqueeze(y, 1)

        optimizer.zero_grad()

        outputs = singlenet(inputs)
        mu = torch.unsqueeze(outputs[:, 0], 1)
        sigma = torch.unsqueeze(softplus(outputs[:, 1])+1e-6, 1)
        
        loss = nll_loss(mu, sigma, labels)
        loss.backward()
        optimizer.step()

        batch_loss += loss.sum().item()

    if epoch % 10 == 0:
        print(f"loss: {batch_loss/len(data_loader)}")

    losses.append(batch_loss/len(data_loader))
```
그리고 위 모형으로 적합된 결과를 동일한 방법으로 그린 플롯은 아래와 같습니다.<br>
![image](/assets/assets_post/2020-12-06-deep_ensemble/single_nll_result1.png)<br>
훈련에 사용된 데이터가 있는 구간에서는 굉장히 낮은 uncertainty를 보이지만 훈련 데이터를 벗어나는 구간부터(-3, 3) 어느 정도의 uncertainty를 보이고 있음을 확인할 수 있습니다. 즉 논문에서 제안한 대로 Negative Log-likelihood를 "proper score rule"로 설정 및 손실함수로 이용함으로써, 우리는 신경망 모델로 하여금 uncertainty를 적절하게 측정할 수 있도록 학습하게 되는 것입니다. 다만 아직도 적절한 분포가 추정이 되었다고 보기는 힘듭니다.<br>

### (3) NLL Loss + Ensemble
최종적으로는 NLL Loss를 활용한 앙상블 모형을 구축하여 uncertainty를 측정할 수 있습니다. 코드는 아래와 같습니다.

```python
def load_ensemble(N):
    all_nets = []
    for i in range(N):
        weaknet = SingleNet()
        optimizer = optim.Adam(weaknet.parameters(), lr = 0.1)
        all_nets.append((weaknet, optimizer))
    return all_nets

def train_ensemble(net, optimizer, dataset, epochs):
    epochs = epochs
    single_net = net
    optimizer = optimizer

    losses = []

    for epoch in range(epochs):

        batch_loss = 0
        for i, (x, y) in enumerate(data_loader):

            inputs = torch.unsqueeze(x, 1)
            labels = torch.unsqueeze(y, 1)

            optimizer.zero_grad()

            outputs = singlenet(inputs)
            mu, sigma = singlenet(inputs)
            
            loss = nll_loss(mu, sigma, labels)
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()

        losses.append(batch_loss/len(data_loader))

    return single_net



ensemble_nets = load_ensemble(5)
trained_nets = []
for a_net in tqdm(ensemble_nets, desc = "Training single models"):
    net_ = a_net[0]
    opt_ = a_net[1]

    trained_net = train_ensemble(net_, opt_, data_loader, epochs = 2000)
    trained_nets.append(trained_net)


predictions_mu = []
predictions_sigma = []
for net in trained_nets:
    net.eval()
    
    x_for_plot = np.linspace(-4, 4, 400)
    y_for_plot = [np.power(x_, 3) for x_ in x_for_plot]

    pred_mu = list(map(lambda r: r.detach().numpy(), [singlenet(torch.tensor([x]))[0] for x in x_for_plot]))
    pred_sigma = list(map(lambda r: np.sqrt(r.detach().numpy())*3, [singlenet(torch.tensor([x]))[1] for x in x_for_plot]))
    
    pred_mu = np.array(pred_mu).reshape(-1)
    pred_sigma = np.array(pred_sigma).reshape(-1)
    
    predictions_mu.append(pred_mu)
    predictions_sigma.append(pred_sigma)

# MU
ensem_mus = np.vstack(predictions_mu)
ensem_nll_mean = np.mean(ensem_mus, axis = 0)

# Sigma
ensem_sigma = np.vstack(predictions_sigma)

# Final values
var_fin = np.mean(np.power(ensem_mus, 2) + ensem_sigma, axis = 0) - np.power(ensem_nll_mean, 2)
sig_fin = np.sqrt(var_fin)*3
```

![image](/assets/assets_post/2020-12-06-deep_ensemble/ensemble_nll_result1.png)<br>
적합 결과를 통해, single NN(with NLL)을 적합했을 때보다 훈련 데이터 바깥 쪽의 공간에서 패턴을 더 잘 설명하고 있는 것 뿐 아니라, uncertainty 또한 잘 설명하고 있음을 확인할 수 있습니다.


# Conclusions
오늘 소개드린 논문 \<Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles\>는 신경망 모형의 불확실성을 구하는 데에 기존 Bayesian 방법론에 비해 좀 더 효율적이고 일반적인 방법론을 제시합니다. 불확실성에 대한 정확한 측정은 신경망의 정확성 추구 외에 꼭 수행해야 할 필수적인 요소라고 생각하기에, 관련한 논문을 더욱 찾아보고 공부할 계획입니다.<br>

긴 포스팅 읽어주셔서 너무나 감사합니다. 배울 것이 많은 학생이라 제가 해석한 내용 중 틀린 부분이나 보완이 필요한 부분이 있을 수 있습니다. 가르침 주시기 위한 피드백은 언제나 환영입니다.<br><br><br>