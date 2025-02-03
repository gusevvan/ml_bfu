import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def getData(inputFile):
    xs, ys = [], []
    with open(inputFile, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        first = True
        for row in spamreader:
            if first:
                first = False
            else:
                x, y = list(map(float, row[0].split(',')))
                xs.append(x)
                ys.append(y)
    return (xs, ys)

def printStats(xs, ys):
    print("Stats:")
    print("X:")
    print("Min:", min(xs))
    print("Max:", max(xs))
    print("Avg:", sum(xs) / len(xs))
    print("Y:")
    print("Min:", min(ys))
    print("Max:", max(ys))
    print("Avg:", sum(ys) / len(ys))

def linearRegression(xs, ys):
    n = len(xs)
    w1 = (sum(xs) * sum(ys) / n - sum([xs[i] * ys[i] for i in range(n)])) / (sum(xs)**2 / n - sum([xs[i]**2 for i in range(n)]))
    w0 = sum(ys) / n - w1 * sum(xs) / n
    return w0, w1

if __name__ == "__main__":
    print("Input csv file name:")
    fileName = input()
    print("Input number X column:")
    xs, ys = [], []
    if input() == '0':
        xs, ys = getData(fileName)
    else:
        ys, xs = getData(fileName)
    printStats(xs, ys)
    w0, w1 = linearRegression(xs, ys)
    y_pred = []
    for x in xs:
        y_pred.append(w0 + w1 * x)
    fig, ax = plt.subplots()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.plot(xs, y_pred, c = 'r')
    for i in range(len(xs)):
        loss = abs(ys[i] - y_pred[i])
        if ys[i] - w1 * xs[i] - w0 > 0:
            loss *= -1
        rect = Rectangle((xs[i], ys[i]), loss, loss, linewidth=0.5, edgecolor='r', facecolor='blue', alpha=0.5)
        ax.add_patch(rect)
    ax.scatter(xs, ys, c = 'g')

    plt.show()
