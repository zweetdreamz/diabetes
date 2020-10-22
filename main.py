import math


# чтение данных из файла
def read_data(path):
    with open(path, 'r', encoding='windows-1251') as file:
        content = file.read().splitlines()
        data = list()
        data.append(content[0].split('\t'))
        for i in content[1:]:
            data.append(list(map(lambda x: float(x), i.split('\t'))))
        return data


# умножение вектора на скаляр
def scmult(v, n):
    return list(map(lambda x: x * n, v))


# умножения векторов, учитывающая последний элемент списке tetta = tetta_zero
def tettamult(tetta, v):
    t0 = tetta[lengh]
    tetta = tetta[:lengh]
    return sum([v1i * v2i for v1i, v2i in zip(v, tetta)]) + t0


# попарное сложение векторов
def vsum(v1, v2):
    return [v1i + v2i for v1i, v2i in zip(v1, v2)]


# расстояние между векторами
def sc(v1, v2):
    return math.sqrt(sum([(v1i - v2i) ** 2 for v1i, v2i in zip(v1, v2)]))


# сигмоида
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# максимум n столбца
def getMax(data, n):
    return max([float(i[n]) for i in data])


# минимум n столбца
def getMin(data, n):
    return min([float(i[n]) for i in data])


# нормализация данных
def norm(data):
    # сразу берем мин и макс для уменьшения сложности алгоритма
    mins = [getMin(data, i) for i in range(lengh)]
    maxs = [getMax(data, i) for i in range(lengh)]
    for i in range(data.__len__()):
        for j in range(lengh):
            data[i][j] = (data[i][j] - mins[j]) / (maxs[j] - mins[j])


lengh = 8  # количество измерений
sigma = 0.01253  # скорость обучения

# все эксперименты с перемешиванием
import random
data = random.sample(read_data('diabetes.txt')[1:], len(read_data('diabetes.txt')[1:]))
#data = read_data('diabetes.txt')[1:]
norm(data)

split = (0.8 * data.__len__()).__int__()
train = data[:split]
test = data[split:]

tetta_start = [-4.06] * (lengh + 1)
tetta = tetta_start

print('tetta start:\n', *tetta_start, sep='\t')
steps = 0

f = True
while f:
    summ = [0] * (lengh + 1)

    for i in train:
        yi, xi = i[lengh], i[:lengh]
        sigmoid_val = sigmoid(tettamult(tetta, xi))
        part = scmult(xi, yi - sigmoid_val) + [0]
        summ = vsum(summ, part)
        summ[lengh] += (yi - sigmoid_val) * sigmoid_val * (1 - sigmoid_val)

    summ = scmult(summ, sigma)
    tmp_tetta = tetta
    tetta = vsum(tetta, summ)
    steps += 1

    if sc(tmp_tetta[1:], tetta[1:]) < 0.01:
        f = False

print('tetta final:\n', *['%.2f' % i for i in tetta], sep='\t')
print('steps:', steps)

accuracy = 0
for i in test:
    pred = tettamult(tetta, i[:lengh])
    result = min([0, 1], key=lambda x: abs(x - sigmoid(pred)))
    if result == i[lengh]:
        accuracy += 1

print('accuracy:', '%.2f' % (accuracy / test.__len__()))
