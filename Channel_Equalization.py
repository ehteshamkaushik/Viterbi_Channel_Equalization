import numpy as np
import math
from bitstring import BitArray
from scipy.stats import multivariate_normal as mvn
import time


def read_params(filename):
    with open(filename) as fp:
        line = fp.readline().split()
        h_values = np.asarray(line).astype(np.float)
        #h_values = h_values[:2]
        noise_variance = float(fp.readline())
        return h_values, noise_variance


def read_data(filename):
    with open(filename) as fp:
        data = fp.read()
    int_data = [int(data[i]) for i in range(len(data))]
    int_data = int_data[:500000]
    return int_data


def get_noise(noise_variance):
    noise = np.random.normal(0, noise_variance, 1)
    return noise[0]


def get_class(value):
    b = BitArray(bin=value)
    return b.uint


def train():
    print("Training")
    x = len(h_values)
    total_count = 0
    class_means = np.zeros((num_classes, l))
    class_counts = np.zeros(num_classes, dtype=int)
    class_x = [[] for i in range(num_classes)]
    for i in range(len(train_data) - n + 1):
        #print(total_count)
        bit_str = ""
        i_values = np.zeros(n)
        for j in range(n):
            bit_str += str(train_data[i+j])
            i_values[j] = train_data[i+j]
        bit_str = bit_str[::-1]
        i_values = np.flip(i_values, axis=0)
        #curr_class = get_class(bit_str)
        curr_class = BitArray(bin=bit_str).uint
        class_counts[curr_class] += 1
        total_count += 1
        curr_x = np.zeros(x)
        for j in range(n - x + 1):
            curr_data = [i_values[j+k] for k in range(x)]
            curr_data = np.asarray(curr_data).transpose()
            #curr_noise = get_noise(noise_variance)
            curr_noise = np.random.normal(0, noise_variance, 1)
            curr_x[j] = np.matmul(h_values, curr_data)
            curr_x[j] += curr_noise
        class_means[curr_class] += curr_x
        class_x[curr_class].append(list(curr_x))
    class_covariance = []
    prior_prob = []
    for i in range(num_classes):
        class_means[i] = class_means[i]/class_counts[i]
        prior_prob.append(class_counts[i]/total_count)
        class_covariance.append(np.cov(np.asarray(class_x[i]).transpose()))
    return class_means, class_covariance, prior_prob


def test():
    print("Testing")
    x = len(h_values)
    i_values = np.zeros(n)
    all_x = []
    for i in range(len(test_data) - n + 1):
        for j in range(n):
            i_values[j] = test_data[i+j]
        i_values = np.flip(i_values, axis=0)
        curr_x = np.zeros(x)
        for j in range(n - x + 1):
            curr_data = [i_values[j+k] for k in range(x)]
            curr_data = np.asarray(curr_data).transpose()
            curr_noise = get_noise(noise_variance)
            curr_x[j] = np.matmul(h_values, curr_data)
            curr_x[j] += curr_noise
        all_x.append(curr_x)

    list_seq = [[] for i in range(num_classes)]
    curr_dist = [0 for i in range(num_classes)]

    for i in range(num_classes):
        list_seq[i].append(i)
        curr_dist[i] = prior_prob[i] * mvn.pdf(all_x[0], class_means[i], class_cov[i])

    for i in range(1, len(all_x)):
        new_dist = [0 for i in range(num_classes)]
        trans = [0 for i in range(num_classes)]
        for j in range(num_classes):
            bit = format(j, "03b")
            c = []
            for k in range(2):
                pattern = bit[1:len(bit)] + str(k)
                pos_class = get_class(pattern)
                c.append(curr_dist[pos_class] + 0.5 * mvn.pdf(all_x[i], class_means[j], class_cov[j]))
            new_dist[j] = max(c)
            idx = np.argmax(c)
            trans[j] = get_class(bit[1:len(bit)] + str(idx))
        curr_dist = [m for m in new_dist]
        temp_seq = [[] for i in range(num_classes)]
        for j in range(num_classes):
            temp_seq[j] = [m for m in list_seq[trans[j]]]
            temp_seq[j].append(j)
        list_seq = [m for m in temp_seq]

    seq = int(np.argmax(curr_dist))
    result = []
    result.append(0)
    result.append(0)
    print("Result")
    for i in list_seq[seq]:
        if i > 3:
            print(1, end='')
            result.append(1)
        else:
            print(0, end='')
            result.append(0)

    print()
    print("Original")
    count = 0
    for i in range(2, len(test_data)):
        print(test_data[i], end='')
        if test_data[i] == result[i]:
            count+=1
    print()
    print("Accuracy : ", (count/(len(result)-2)) * 100)


h_values, noise_variance = read_params("params.txt")
train_data = read_data("train.txt")
test_data = read_data("test.txt")
test_data.insert(0, 0)
test_data.insert(1, 0)
n = 3
l = 2
num_classes = int(math.pow(2, n))
class_means, class_cov, prior_prob = train()
test()