---
title: "GAN이 무엇인가요?"
excerpt: "인공지능 여러 분야에서 널리 쓰이고 있는 GAN을 이미지분야를 예로 들어, 개념과 
작동원리를 수식과 함께 설명합니다."
author: "조현근"
toc: true
toc_sticky: true
toc_label: 바로가기

categories:
  - Computer Vision
last_modified_at: 2020-11-01
---
이번 포스트에서는 GAN의 개념과 작동원리를 수식 및 코드를 동원해 설명해보려 합니다. GAN의 핵심은
Generator와 Discriminator를 두고 이 둘을 적대적으로 학습시키겠다는 것인데, 이를 수식 및 시각적자료를 
활용하여 설명합니다.

## Generative Adversarial Network

![]({{site.url}}/assets/images/jovis/2020-10-31-gan/image_1.png)

제목에서도 알 수 있듯, GAN은 Generative, 생성모델입니다. 만약 버스에 대한 데이터를 가지고 있다면
GAN에 버스이미지를 학습시키고, 학습된 버스 이미지와 비슷한 다른 버스이미지를 만들어낼 수 있습니다.
이것이 가능하기 위해서는 인간이 하나의 자동차를 보고 여러 모습의 다른 자동차를 연상하듯, 
기계도 그런 능력을 갖출 수 있도록 학습해야 합니다.

사람이 비슷한 자동차 몇 대를 연상하는 것은 어렵지 않습니다.
저는 이 포스팅을 하면서도 벤츠, 아우디, 페라리등과 같이 현실과 괴리감이 있는 자동차들을 
떠올리고 있으니까요. 기계가 하나의 이미지를 보고 유사한 다른 이미지를 연상하는 것은 어려운 일입니다. 
일단은 버스이미지 안에서 버스의 창문, 출입문, 바퀴, 긴 몸체등과 같은 정보를 캐치하고 '이 이미지는 
버스야!' 라고 구별할 수 있어야겠죠. 만약 이미지 분류 문제라면 구별만으로 끝날 수 있겠지만, 생성모델은 
구별하고, 비슷한 이미지를 생성하기 까지 해야 합니다.

GAN에서는 이미지가 무엇인지 구별하고, 이를 바탕으로 다른 이미지를 생성해내기 위해서 두 가지 개념을 
사용합니다. 바로 **Discriminator(구별자)**와 **Generator(생성자)**입니다. 보시는 것 처럼 구별하는 놈따로, 
생성하는 놈따로 나누어 놓았습니다. 사람이 살아오면서 수 많은 자동차를 목격하고 데이터를 쌓아 이를 
기반으로 다른 자동차를 연상하듯, 기계도 수많은 이미지를 판단하고 그 정보를 바탕으로 다른 자동차를 
연상하게끔 나누어 놓았다고 볼 수 있습니다. 

GAN의 핵심은 Discrinimator와 Generator를 **적대적으로(Adversarial)**, 맞물려 학습하여 사진을 잘 구별하면서, 
높은 품질의 사진을 생성하겠다는 것입니다. 우선은 적대적으로 학습하는 것이 무엇인지에 대해 조금 더 알아보겠습니다.

### Discriminator 와 Generator

![]({{site.url}}/assets/images/jovis/2020-10-31-gan/image_2.png)

앞서 말씀드렸듯이, GAN에서는 Generator와 Discriminator를 적대적(Adversarial)으로 경쟁을 시키며 학습을 
합니다. 그렇기 때문에 결과적으로 이미지를 생성하는 쪽은 Generator이나, Generator가 학습되기 위해서는 
Discriminator가 꼭 필요합니다. 쉬운 이해를 위해서 Discriminator와 Generator를 각각 경찰과 사기꾼에 
비유하도록 하겠습니다. 경찰은 사기꾼의 그럴 듯한 거짓말을 구별에 감방에 처넣으려 할 것이고, 사기꾼은 
어떻게든 경찰을 거짓을 진실로 믿게끔하여 형량을 줄이려 하겠죠. 

GAN의 학습 또한 비슷하게 이루어 집니다.
이미지를 학습한다고 했을 때, Discriminator는 우리가 보유하고 있는 train image data는 진짜로, Generator가
생성하는 이미지는 가짜로 분별하여 학습합니다. 반면, Generator는 Discriminator가 자신이 전달한 이미지를 
진짜로 받아들이도록 학습합니다. 이 둘의 경쟁관계를 이용하여 Discriminator는 진짜에 가까운 가짜를 가짜로
잘 구분할 수 있는 능력을 갖추게 되고, Generator는 진짜에 가까운 이미지를 생성해 낼 수 있게 됩니다.

### 적대적으로 학습하는 법

GAN이 사용하는 Discriminator와 Generator가 적대적으로 학습을 하는 윈윈전략이 수식적으로, 코드로 어떻게
이루어지는지 궁금하지 않을 수 없습니다. 사실 이것을 이해하려면 **Deep Neural Network의 구성과, 목적함수
(Objective Function), 순전파 및 역전파(BackPropagation)**에 대한 지식이 필요합니다. 이번 포스트에서는 자세한 원리는 접어두고,
 아래의 내용을 이해하기 쉽도록 딥러닝의 학습과정에 대해서 간단하게 짚고 넘어가겠습니다. 

#### 딥러닝 모델 학습 구조

![]({{site.url}}/assets/images/jovis/2020-10-31-gan/image_3.png)

이 포스트에서 설명하고 있는 Discriminator와 Generator는 흔히 인공신경망이라고 불립니다. 
이것을 하나의 큰 블랙박스이며, 내부에서 연산 되는 과정은 모른다고 가정하겠습니다.
다만, 블랙박스 내부에는 값을 조절 가능한 가중치(그림의 스위치, weight)들이 무수히 많이 존재하며, 이 값을 조절하는 것을 
'학습한다'라고 합니다. 그런데 아무 값이나 랜덤하게 조절해서는 안됩니다. 버스 이미지가 들어갔을 때는 이것이 
버스임을 인식할 수 있도록 조절해야겠죠. 블랙박스가 버스일 경우 1, 아닐 경우 0으로 output을 내뱉도록 
가중치를 조절한다고 합시다. 우리는 블랙박스에서 처리하여 계산한 값이 버스일 경우 1과 얼마나 큰 차이가 나는지,
아닐 경우 0과 얼마나 큰 차이가 나는지를 계산하여 차이나 나는 만큼 가중치를 조절해줄 수 있을 것입니다. 
지금처럼 '버스일 경우 1, 아닐 경우 0으로 두고 차이를 계산할 꺼야!' 와 같은 것을 정하는 것을 **Objective Function**
혹은 **Loss function**, 계산된 차를 **Loss**, 가중치를 조절하는 행위를 **BackPropagation**이라고 합니다.

#### GAN의 학습 방식

이제 GAN의 Objective Function을 한번 살펴보겠습니다. 위에서 Objective Function을 예로 들었던 것 처럼, 
아래의 수식에는 아마도 Discriminator와 Generator가 적대적으로 어떻게 학습되는지에 대한 내용이 있을 겁니다.
Discriminator는 실제이미지를 진짜로, Generator에서 생성한 이미지는 가짜로 받아들여야 하며, Generator는 Discriminator가
 자신이 생성한 이미지를 실제로 받아들이도록 학습을 해야할겁니다. 이 내용이 Objective Function에 녹아있겠죠.

![]({{site.url}}/assets/images/jovis/2020-10-31-gan/image_4.png)

D는 Discriminator, G는 Generator를 의미합니다. D와 G에 관한 목적함수를 V(D, G)로 표현하였고 D에 관해서는 Maximize,
G에 관해서는 Minimize를 하는 방향으로 학습을 하겠다고 하네요. GAN의 학습은 두 부분으로 나누어서 학습을 
진행합니다. 한번은 D에 관해서, 한번은 G에 관해서 학습을 하는 것 입니다. 우선 Discriminator 관점에서는 어떻게 
학습을 하겠다는 건지 살펴보겠습니다.

#### Discriminator의 학습

![]({{site.url}}/assets/images/jovis/2020-10-31-gan/image_5.png)

Discriminator는 실제이미지가 들어갔을 때는 True(1)로, Generator에서 생성된 이미지가 들어갔을 때는 False(0)로 인식
할 수 있게끔 학습이 되어야 합니다. 우선 초록색의 실제이미지가 들어갔을 때를 먼저 살펴보죠. x는 실제 이미지의 분포
p(x)에서 샘플링된 값을 의미하며 우리가 가지고 있는 실제 이미지 데이터를 의미합니다. Discriminator가 내뱉는 D(x)의 
값은 0~1사이 값을 내뱉게 되는데, 이것이 log함수에 들어갔을 때는 1일 때 0으로 최댓값을 가지게 됩니다. 따라서 
Discriminator에 실제 사진이 들어갔을 경우 1(True)의 값을 내뱉어야 목적함수를 최대화 하는 방향인 것을 알 수 있습니다.
이제 Generator에서 생성된 사진이 들어갔을 파란색의 경우를 살펴보겠습니다.

Generator에는 Gaussian Distribution(표준정규분포)나 Uniform Distribution(균일분포)에서 샘플링 된 픽셀값을 가지는 
사진이 들어가게 됩니다. 처음엔 Random한 사진이 들어가 Generator를 학습시켜 나가면서 real data distribution에 가까운 
분포를 만들어 나가겠다는 것입니다. G(z)는 Generator를 통해 생성된 이미지를 의미하며 z의 경우 표준정규분포에서 random
하게 sampling된 이미지를 뜻합니다. (1-D(G(z)))는 log함수에 대입되었을 때 역시 1에 가까운 값을 내뱉어야 목적함수가 최대가 되겠고,
이것은 D(G(z))의 경우 0에 가까운 값을 내뱉어야 함을 의미합니다. 즉, Generator가 생성한 가짜이미지가 들어갔을 때는 
가짜의 값 0을 내뱉어야한다는 소리가 되겠죠. 앞서 Discriminator는 실제이미지가 들어갔을 때는 True, Generator에서 생성된 
이미지가 들어갔을 떄는 False값을 내뱉어야 한다는 것과 수식이 의미하는 바가 맞아떨어짐을 알 수 있습니다.

#### Generator의 학습

![]({{site.url}}/assets/images/jovis/2020-10-31-gan/image_6.png)

Generator는 자신이 가짜 이미지를 생성하여 Discriminator에 넘겼을 때 이것을 진짜로 인식하도록 해야한다고 했습니다. 
그것이 목적함수를 G에 관해서 minimize하는 것과 일맥상통하는지를 보면 될 것입니다. 먼저 알고 넘어가야할 것은, Generator에 관련해
학습을 할 때는 초록색 부분은 고려하지 않아도 된다는 것입니다. G에 관해서 Update, D에 관해서 업데이트를 한다는 것은
목적함수를 G, D 각각에 대해서 편미분하여 나온 값인 partial derivate 값을 앞서 설명한 Black Box의 weight, 지금은 GAN의 weight에 
반영하여 업데이트를 하겠다는 뜻입니다. G에 대해 편미분을 할 경우 이와 연관이 없는 부분은 상수항으로 취급하여 고려하지 
않아도 됩니다. 따라서 Generator를 업데이트 할 때 파란색 부분만 보면 되겠는데, 목적함수를 minimize하기 위해서는 D(G(z)) 값이 1
에 가까울때 log(1-D(G(Z)))값이 -무한대로 가장 작은 값이 될 겁니다.

지금까지 Discriminator와 Generator가 학습되는 방식을 목적함수에서 어떻게 나타내고 있는지 살펴보았고, GAN의 핵심적인 내용을 
이해할 수 있게 되었습니다.
