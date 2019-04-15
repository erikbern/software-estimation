import matplotlib
import numpy
import scipy.stats
from matplotlib import pyplot

matplotlib.style.use('ggplot')


# Plot log-normal distribution
pyplot.figure(figsize=(9, 6))
std = 1
xs = numpy.linspace(0, 10, 1000)
ys = scipy.stats.lognorm.pdf(xs, s=std)
pyplot.plot(xs, ys, color='C0', lw=3)
pyplot.fill_between(xs, ys*0, ys, color='C0', alpha=0.3, label='Distribution')
pyplot.axvline(x=1, color='C1', lw=3, label='Median: 1.00')
mean = numpy.exp(0 + std**2/2)
pyplot.axvline(x=numpy.exp(0 + std**2/2), color='C2', lw=3, label='Mean : %.2f' % mean)
pyplot.xlabel('Blowup factor (actual/estimated)')
pyplot.ylabel('Probability distribution')
pyplot.legend()
pyplot.title('Standard deviation $ \\sigma = %.2f $' % std)
pyplot.tight_layout()
pyplot.savefig('log_normal.png')

# Plot normal distributions
pyplot.figure(figsize=(9, 6))
xs = numpy.linspace(-5, 5, 1000)
for std in [0.5, 1, 2]:
    ys = scipy.stats.norm.pdf(xs, scale=std)
    pyplot.plot(xs, ys, lw=3, label='Standard deviation $ \\sigma = %.2f $' % std)
pyplot.xlabel('Logarithm of blowup factor: log(actual/estimated)')
pyplot.ylabel('Probability distribution')
pyplot.legend()
pyplot.tight_layout()
pyplot.savefig('normal.png')


def add(sizes, stds):
    mus = numpy.log(sizes)
    rows = []
    for mu, std in zip(mus, stds):
        mean = numpy.exp(mu + std**2/2)
        p99 = numpy.exp(scipy.stats.norm.ppf(0.99, mu, std))
        rows.append((numpy.exp(mu), mean, p99))

    rvs = scipy.stats.norm.rvs(mus, stds, size=(1000000, len(mus)))
    sums = numpy.sum(numpy.exp(rvs), axis=1)
    rows.append((numpy.median(sums), numpy.mean(sums), numpy.percentile(sums, 99)))

    return rows



for sizes, stds in [([1, 1, 1], [1, 1, 1]),
                    ([1, 1, 1], [0.5, 1, 2]),
                    ([1, 1, 1, 1, 1, 1, 1], [0.5, 0.5, 0.5, 1, 1, 1, 2])]:
    for median, mean, p99 in add(sizes, stds):
        print('%9.2f %9.2f %9.2f' % (median, mean, p99))
    print()


pyplot.figure(figsize=(9, 6))
sigmas = numpy.linspace(0, 5, 10000)
medians = numpy.exp(sigmas * 0)
means = numpy.exp(sigmas**2/2)
# p99s = [numpy.exp(scipy.stats.norm.ppf(0.99, loc=0, scale=sigma)) for sigma in sigmas]
p99s = numpy.exp(scipy.stats.norm.ppf(0.99, loc=0, scale=sigmas))
print(p99s)
pyplot.plot(sigmas, medians, lw=3, label='Median')
pyplot.plot(sigmas, means, lw=3, label='Mean')
pyplot.plot(sigmas, p99s, lw=3, label='99th percentile')
pyplot.yscale('log')
pyplot.xlabel('$ \\sigma $')
pyplot.ylabel('Blowup factor')
pyplot.legend()
pyplot.tight_layout()
pyplot.savefig('sigmas.png')
