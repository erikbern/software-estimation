import matplotlib
import numpy
import scipy.stats
from matplotlib import pyplot

matplotlib.style.use('ggplot')


def plot_lognorm(std):
    xs = numpy.linspace(0, 10, 1000)
    ys = scipy.stats.lognorm.pdf(xs, std)
    pyplot.plot(xs, ys, color='C0')
    pyplot.fill_between(xs, ys*0, ys, color='C0', alpha=0.3, label='Distribution')
    pyplot.axvline(x=1, color='C1', label='Median')
    pyplot.axvline(x=numpy.exp(0 + std**2/2), color='C2', label='Mean')
    pyplot.legend()
    pyplot.title('median = %.2f, standard deviation = %.2f' % (1, std))


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


for std in [0.5, 1, 2]:
    pyplot.figure(figsize=(9, 6))
    plot_lognorm(std)
    pyplot.tight_layout()
    pyplot.savefig('log_normal_%.2f.png' % std)
    

for sizes, stds in [([1, 1, 1], [1, 1, 1]),
                    ([1, 1, 1], [0.5, 1, 2]),
                    ([1, 1, 1, 1, 1, 1, 1], [0.5, 0.5, 0.5, 1, 1, 1, 2])]:
    for median, mean, p99 in add(sizes, stds):
        print('%9.2f %9.2f %9.2f' % (median, mean, p99))
    print()
