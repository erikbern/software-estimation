import csv
import math
import matplotlib
import numpy
import scipy.optimize
import scipy.stats
import seaborn
from matplotlib import pyplot

# https://raw.githubusercontent.com/Derek-Jones/SiP_dataset/master/Sip-task-info.csv
actuals = []
estimates = []
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
        actuals.append(actual)
        estimates.append(estimate)
        if estimate > 7:
            deltas.append(math.log(actual) - math.log(estimate))


matplotlib.style.use('ggplot')

pyplot.figure(figsize=(9, 6))
pyplot.scatter(estimates, actuals, alpha=0.05)
pyplot.xscale('log')
pyplot.yscale('log')
pyplot.xlabel('Estimated number of hours')
pyplot.ylabel('Actual number of hours')
pyplot.xlim([1e-1, 1e3])
pyplot.ylim([1e-1, 1e3])
pyplot.plot([1e-1, 1e3], [1e-1, 1e3], color='C1', alpha=0.5, lw=3, label='Estimated=actual')
pyplot.legend()
pyplot.tight_layout()
pyplot.savefig('scatter.png')

print('mean:', numpy.mean(numpy.exp(deltas)))
print('median:', numpy.median(numpy.exp(deltas)))
print('p99:', numpy.percentile(numpy.exp(deltas), 99))

pyplot.figure(figsize=(9, 6))
seaborn.distplot(deltas, kde=False, norm_hist=True, bins=numpy.arange(-5, 5, 0.2) + 0.1)
pyplot.xlabel('log(actual / estimated)')
pyplot.xlim([-5, 5])
pyplot.ylabel('Probability distribution')
pyplot.tight_layout()
pyplot.savefig('distribution.png')

def neg_ll(params):
    df, scale = numpy.exp(params)
    return -numpy.sum(scipy.stats.t.logpdf(deltas, df, 0, scale))

params = scipy.optimize.minimize(neg_ll, (0, 0)).x
df, scale = numpy.exp(params)
print('nu:', df)
print('scale:', scale)

xs = numpy.linspace(-10, 10, 1000)
std = 0.5
ys = scipy.stats.t.pdf(xs, df, 0, scale)
pyplot.plot(xs, ys, color='C1', lw=3, label='$ \\nu = %.2f, \\sigma = %.2f $' % (df, scale))
pyplot.xlim([-5, 5])
pyplot.legend()
pyplot.title('Best fit of a non-standardized Student\'s t-distribution')
pyplot.tight_layout()
pyplot.savefig('distribution_plus_t.png')

# zs = scipy.stats.t.rvs(df=df, scale=scale, size=1000000)
d = scipy.stats.t(df=df, scale=scale)
xs = numpy.linspace(-30, 30, 1000000)
print('mean:', numpy.mean(d.pdf(xs) * numpy.exp(xs)))
print('median:', numpy.exp(d.ppf(0.5)))
print('p99:', numpy.exp(d.ppf(0.99)))
print('p99.9:', numpy.exp(d.ppf(0.999)))
print('p99.99:', numpy.exp(d.ppf(0.9999)))

a = 2*df - 1
b = scale * a
print(a, b)
d = scipy.stats.invgamma(a=a, scale=b)
xs = numpy.linspace(0, 10, 10000)
pyplot.figure(figsize=(9, 6))
pyplot.fill_between(xs, 0*xs, d.pdf(xs), alpha=0.2, color='C0', label='$ \\alpha = %.2f, \\beta = %.2f $' % (a, b))
pyplot.plot(xs, d.pdf(xs), lw=3, color='C0')
pyplot.xlabel('$ \\sigma $')
pyplot.ylabel('Probability distribution')
pyplot.xlim([0, 5])
pyplot.legend()
pyplot.title('Inferred inverse Gamma distribution of $ \sigma $')
pyplot.tight_layout()
pyplot.savefig('sigma_distribution.png')

#pyplot.figure(figsize=(9, 6))
#ss = scipy.stats.invgamma.rvs(a=a, scale=b, size=10000)
#zs = scipy.stats.norm.rvs(scale=ss)
#zs = numpy.clip(zs, -5, 5)
#seaborn.distplot(zs)
#pyplot.show()
