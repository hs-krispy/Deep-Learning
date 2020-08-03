## 확률적 경사 하강법(SGD)

최적화 - 매개변수의 최적값을 찾는 문제

**기울어진 방향으로 매개변수를 갱신하는 일을 반복(기울어진 방향으로 일정 거리만 가겠다는 방법)**

```python
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
       
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
```

**가중치 매개변수 갱신 = 가중치 매개변수 - 학습률 x 가중치 매개변수에 대한 손실 함수의 기울기**

#### SGD의 단점

**비등방성 함수(특정한 지점에서 기울기가 가리키는 지점이 변하는 경우가 존재, 기울기가 가르키는 지점이 하나가 아니라 여러가지)**에서는 **탐색 경로가 비효율적**

<img src="https://user-images.githubusercontent.com/58063806/89200788-7c708080-d5eb-11ea-8341-27ed542393d4.JPG" width=50% />

**기울기 대부분이 최솟값이 되는 지점 (0, 0)을 가리키지 않음**

<img src="https://user-images.githubusercontent.com/58063806/89200791-7d091700-d5eb-11ea-97fb-da891296b5a8.JPG" width=50% />

최솟값인 (0, 0)까지 **지그재그의 형태로 이동하므로 비효율적**