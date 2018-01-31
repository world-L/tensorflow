# 털과 날개가 있는지 없는지에 따라, 포유류인지 조류인지 분류하는 신경망 모델을 만들어봅니다.
import tensorflow as tf
import numpy as np

# [털, 날개]
x_data = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])

# one-hpt data
y_data = np.array([
    [1, 0, 0],  # 기타
    [0, 1, 0],  # 포유류
    [0, 0, 1],  # 조류
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])



#########
# 신경망 모델 구성
######
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 신경망은 2차원으로 [입력층(특성), 출력층(레이블)] -> [2, 3] 으로 정합니다.
W = tf.Variable(tf.random_uniform([2, 3], -1., 1.))

# 편향을 각각 각 레이어의 아웃풋 갯수로 설정합니다.
# 편향은 아웃풋의 갯수, 즉 최종 결과값의 분류 갯수인 3으로 설정합니다.
b = tf.Variable(tf.zeros([3]))

# 신경망에 가중치 W과 편향 b을 적용합니다
L = tf.add(tf.matmul(X, W), b)

## check data
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("===weight===")
print(sess.run(W))
print()

print("===Y===")
print(sess.run(Y, feed_dict={Y: y_data}))
print()

print("===X*weigth+b===")
print(sess.run(L, feed_dict={X: x_data}))
print()

# Rectifier,  activation function defined as the positive part of its argument
# https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
L = tf.nn.relu(L)
print("===relu===  <- X*weigth+b")
print()
print(sess.run(L, feed_dict={X: x_data}))
print()
print()


# 결과 값의 합계를 1로 만들 수 있도록 수정
model = tf.nn.softmax(L)
print("===softmax===  <- relu")
print()
print(sess.run(model, feed_dict={X: x_data}))
print()
print()

# soft max 결과 확인
print("===softmax check===")
print()
for i in range(sess.run(model, feed_dict={X: x_data}).shape[0]):
    print(sess.run(model,feed_dict={X:x_data})[i].sum())
print()
print()

# 신경망을 최적화하기 위한 비용 함수를 작성합니다.
# 각 개별 결과에 대한 합을 구한 뒤 평균을 내는 방식을 사용합니다.
# 전체 합이 아닌, 개별 결과를 구한 뒤 평균을 내는 방식을 사용하기 위해 axis 옵션을 사용합니다.
# axis 옵션이 없으면 -1.09 처럼 총합인 스칼라값으로 출력됩니다.
#        Y         model         Y * tf.log(model)   reduce_sum(axis=1)
# 예) [[1 0 0]  [[0.1 0.7 0.2]  -> [[-1.0  0    0]  -> [-1.0, -0.09]
#     [0 1 0]]  [0.2 0.8 0.0]]     [ 0   -0.09 0]]
# 즉, 이것은 예측값과 실제값 사이의 확률 분포의 차이를 비용으로 계산한 것이며,
# 이것을 Cross-Entropy 라고 합니다.
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))

print("===Y*log(model)===  <- softmax")
print()
print(sess.run(Y * tf.log(model), feed_dict={X: x_data,Y: y_data}))
print()
print()

print("===reduce sum axis1===  <- Y*log(model)")
print()
print(sess.run(-tf.reduce_sum(Y * tf.log(model), axis=1), feed_dict={X: x_data,Y: y_data}))
print()
print()

print("===reduce mean===")
print()
print(sess.run(cost, feed_dict={X: x_data,Y: y_data}))
print()
print()

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

sess.close()

#########
# 신경망 모델 학습
######
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
        #print(sess.run(model, feed_dict={X: x_data}))
        print()


# tf.argmax: Returns the index with the largest value across dimensions of a tensor

prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)

print("===model===")
print(sess.run(model, feed_dict={X: x_data}))

print("===target===")
print(sess.run(Y, feed_dict={Y: y_data}))
print()
print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실제값:', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))

print()