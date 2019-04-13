import csv
import math
import numpy
import scipy.optimize
import scipy.stats
import seaborn
from matplotlib import pyplot

deltas = []
with open('Sip-task-info.csv') as f:
    reader = csv.reader(f)
    header = next(reader)
    while True:
        try:
            row = next(reader)
        except UnicodeDecodeError:
            continue
        except StopIteration:
            break

        row = dict(zip(header, row))
        estimate, actual = row.get('HoursEstimate'), row.get('HoursActual')
        if estimate is None or actual is None:
            continue
        estimate, actual = float(estimate), float(actual)
        deltas.append(math.log(actual) - math.log(estimate))


def fit(zs):
    print('min/max zs:', min(zs), max(zs))
    print('p9/p99/p999:', numpy.percentile(zs, [90, 99, 99.9]))
    print('p9/p99/p999 of exp:', numpy.percentile(numpy.exp(zs), [90, 99, 99.9]))
    print('mean(exp):', numpy.mean(numpy.exp(zs)))
    zs = numpy.clip(zs, -10, 10)
    seaborn.distplot(zs)

    def neg_ll(params):
        df, scale = numpy.exp(params)
        return -numpy.sum(scipy.stats.t.logpdf(zs, df, 0, scale))

    params = scipy.optimize.minimize(neg_ll, (0, 0)).x
    df, scale = numpy.exp(params)
    print('nu:', df)
    print('scale:', scale)

    xs = numpy.linspace(-10, 10, 1000)
    std = 0.5
    ys = scipy.stats.t.pdf(xs, df, 0, scale)
    pyplot.plot(xs, ys, color='red')
    pyplot.xlim([-10, 10])
    pyplot.show()

    return df, scale

nu, scale = fit(deltas)
print(scipy.stats.t.cdf([-10, 10], df=nu, scale=scale))
zs = scipy.stats.t.rvs(df=nu, scale=scale, size=10000)
fit(zs)

a = 2*nu - 1
b = scale * a
ss = scipy.stats.invgamma.rvs(a=a, scale=b, size=10000)
zs = scipy.stats.norm.rvs(scale=ss)
nu, scale = fit(zs)

seaborn.distplot(ss)
pyplot.show()
