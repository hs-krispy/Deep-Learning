## 합성곱/풀링 계층 구현

### im2col

입력 데이터를 필터링(가중치 계산)하기 좋게 전개하는 함수

4차원 입력 데이터를 2차원으로 변환

```python
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    col : 2차원 배열
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    # 1차원에 0, 2차원에 4, 3차원에 5 ...
    # reshape -1로 설정해주면 자동적으로 변환 EX) (100, )->reshape(2, -1) 열은 50으로
    return col
```

transpose() - 사용자가 원하는 대로 다차원 배열의 축을 바꿀 수 있음

#### col2im

```python
# 합성곱 계층의 역전파에 사용

    def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """(im2col과 반대) 2차원 배열을 입력받아 다수의 이미지 묶음으로 변환한다.
    
    Parameters
    ----------
    col : 2차원 배열(입력 데이터)
    input_shape : 원래 이미지 데이터의 형상（예：(10, 1, 28, 28)）
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    img : 변환된 이미지들
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]
```

#### 합성곱 계층 구현

```python
class Convolution:
    
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T # 필터 전개
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
		# 출력 데이터를 적절한 형상으로 다시 돌림(N, C, H, W)
        self.x = x
        self.col = col
        self.col_W = col_W

        return out
```

#### 풀링 계층 구현

- 입력 데이터를 전개
- 행별 최대값을 구함
- 적절한 모향으로 성형

<img src="https://user-images.githubusercontent.com/58063806/89871271-1dfe6000-dbf2-11ea-9b8c-38cc96ffba78.PNG" width=60% />

```python
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
		
        # 입력 데이터 전개
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        out = np.max(col, axis=1) # 최대값(입력 x의 1번째 차원의 축마다 최대값 구함)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2) # 성형

        return out
```

## CNN 구현

```python
class SimpleConvNet:
    
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        # 합성곱과 풀링 계층의 출력 크기를 계산
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))
        
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()
    
    def predict(self, x):
        # layers에 추가한 계층을 앞부터 차례로 forward 메서드를 호출하고 그 결과를 다음 계층에 전달
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)
    
    # 오차역전파법으로 기울기를 구함
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
```

### CNN 시각화

**1층 합성곱 계층의 가중치(학습 전)**

<img src="https://user-images.githubusercontent.com/58063806/89895789-5c5a4600-dc17-11ea-869d-1419393de15b.JPG" width=60% />

**1층 합성곱 계층의 가중치(학습 후)**

<img src="https://user-images.githubusercontent.com/58063806/89895793-5d8b7300-dc17-11ea-8523-2e110c5256d7.JPG" width=60% />

**학습전의 필터는 무작위로 초기화**되고 있어서 **흑백의 정도에 규칙성이 없지만** **학습 후의 필터는 흰색에서 검은색으로 점차변하는 필터와 덩어리(블롭)가 진 필터 등, 규칙을 띄는 필터**로 바뀜

**계층이 깊어질수록 추출되는 정보(강하게 반응하는 뉴런)는 더 추상화 됨**

<img src="https://user-images.githubusercontent.com/58063806/89896290-2e293600-dc18-11ea-8e29-ba1f10955116.JPG" width=100% />

합성곱 계층을 여러 겹 쌓으면, **층이 깊어지면서 더 복잡하고 추상화된 정보가 추출(뉴런이 반응하는 대상이 점점 고급 정보(사물의 의미를 이해하도록)로 변화**)

### 대표적인 CNN

#### LeNet

손글씨 숫자를 인식하는 네트워크로 합성곱 계층과 풀링 계층(단순히 원소를 줄이는 서브샘플링)을 반복하고, 마지막으로 완전연결 계층을 거치면서 결과를 출력

<img src="https://user-images.githubusercontent.com/58063806/89896718-e3f48480-dc18-11ea-886d-75f7015527f7.JPG" width=90% />

- 활성화 함수로 ReLU를 사용하는 현재의 CNN과 달리 시그모이드 함수를 사용
- 최대 풀링을 주로 사용하는 현재의 CNN과 달리 서브샘플링을 함 

#### AlexNet

합성곱 계층과 풀링 계층을 반복하고, 마지막으로 완전연결 계층을 거치면서 결과를 출력

<img src="https://user-images.githubusercontent.com/58063806/89896983-4e0d2980-dc19-11ea-8142-0e01100aaf5d.JPG" width=90%/>

- 활성화 함수로 ReLU 이용
- LRN이라는 국소적 정규화를 실시하는 계층을 이용
- 드롭아웃을 사용

네트워크 구성 면에서 LeNet과 큰 차이는 없지만 컴퓨터 기술의 진보로 인해 딥러닝이 발전

