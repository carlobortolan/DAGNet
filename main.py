import random
import nn
from engine import Value
from nn import MLP
from trace_graph import draw_dot

# random.seed(1337)
# n = nn.Neuron(2)
# x = [Value(1.0), Value(-2.0)]
# y = n(x)
# y.backward()
# draw_dot(y).render('gout')

x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
n(x)
# draw_dot(n(x)).render('mlp')

# Data
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

# Targets
ys = [1.0, -1.0, -1.0, 1.0]

# Gradient descent
for k in range(100):
    # Forward pass
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    # Backward pass
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    # Update
    for p in n.parameters():
        p.data += -0.01 * p.grad

    print(k, loss.data)

print(ypred)
# draw_dot(loss).render('loss')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
