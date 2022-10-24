import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats

# # # # MOMENT MATCHING # # # #


def mutiplyGauss(m1, s1, m2, s2):
  s = 1/(1/s1+1/s2)
  m = (m1/s1+m2/s2)*s
  return m, s


def divideGauss(m1, s1, m2, s2):
 m, s = mutiplyGauss(m1, s1, m2, -s2)
 return m, s


def truncGaussMM(a, b, m0, s0):
 a_scaled , b_scaled = (a - m0) / np.sqrt(s0), (b - m0) / np.sqrt(s0)
 m = truncnorm.mean(a_scaled , b_scaled , loc=m0, scale=np.sqrt(s0))
 s = truncnorm.var(a_scaled , b_scaled , loc=m0, scale=np.sqrt(s0))
 return m, s


m1, v1, m2, v2 = 0, 1, 0, 1  # Mean and variance of p(s_1) and p(s_2)
s = 1  # Variance of p(t|s_1)
y0 = 1  # Measurement

# Message msg3 and msg4
msg3_m = m1 # mean of message
msg3_s = v1 # variance of message
msg4_m = m2 # mean of message
msg4_s = v2 # variance of message

# Message msg5 and msg6
msg5_m = msg3_m # mean of message
msg5_s = msg3_s # variance of message
msg6_m = msg4_m # mean of message
msg6_s = msg4_s # variance of message

# Message msg7
msg7_m = m1 - m2
msg7_s = s + v1 + v2

# p_t
a, b = 0, np.Inf
p_t_m, p_t_s = truncGaussMM(a, b, msg7_m, msg7_s)

# Message msg8
msg8_m, msg8_s = divideGauss(p_t_m, p_t_s, msg7_m, msg7_s)

# Message msg9 and msg10
msg9_m = m2+msg8_m
msg9_s = s+v2+msg8_s
msg10_m = m1-msg8_m
msg10_s = s+v1+msg8_s

# p_s1 and p_s2
p_s1_m, p_s1_s = mutiplyGauss(m1, v1, msg9_m, msg9_s)
p_s2_m, p_s2_s = mutiplyGauss(m2, v2, msg10_m, msg10_s)


# # # # GIBB'S # # # #
num_samples = 5000
drops = 1000


def importance(s1, s2, y=1):
    pr = 1-stats.norm(s1-s2, s**0.5).cdf(0) if y>0 else stats.norm(s1-s2, s**0.5).cdf(0)
    return pr

def approx_gaussian(data, importances):
    mean = np.average(data, weights=importances)
    variance = np.average((data - mean) ** 2, weights=importances)
    return mean, variance


# observed results
ys = [1, ]

for y in ys:
    posterior = []
    post_weights = []
    s1 = 0
    s1_w = 1

    for _ in range(num_samples):
        s2 = stats.norm.rvs(m2, v2 ** 0.5)
        s2_w = importance(s1, s2, y)

        posterior.append((s1, s2))
        post_weights.append((s1_w, s2_w))

        s1 = stats.norm.rvs(m1, v1 ** 0.5)
        s1_w = importance(s1, s2, y)

    posterior = np.array(posterior)
    post_weights = np.array(post_weights)

    m1, v1 = approx_gaussian(posterior[drops:, 0], post_weights[drops:, 0])
    m2, v2 = approx_gaussian(posterior[drops:, 1], post_weights[drops:, 1])
    print("=======")
    print(f"S1: N({m1}, {v1})")
    print(f"S2: N({m2}, {v2})")

m1_gibbs = m1
v1_gibbs = v1
m2_gibbs = m2
v2_gibbs = v2

plt.hist(posterior[drops:,0], density=True, weights=post_weights[drops:,0], bins=50, range=(-4, 4),color='lightsteelblue', label="Gibbs Histogram of s_1")
plt.hist(posterior[drops:,1], density=True, weights=post_weights[drops:,1], bins=50, range=(-4, 4),color='lightsalmon', alpha=0.5, label="Gibbs Histogram of s_2")

xs = np.linspace(-4, 4, 100)
gaussian1 = stats.norm(m1, v1**0.5).pdf(xs)
gaussian2 = stats.norm(m2, v2**0.5).pdf(xs)
plt.plot(xs, gaussian1, color='cornflowerblue', label="Gibbs Gauss approx. of s_1")
plt.plot(xs, gaussian2, color='coral', label="Gibbs Gauss approx. of s_2")


# PLOT
xv = xs
plt.plot(xv,norm.pdf(xv,p_s1_m ,np.sqrt(p_s1_s)),'x', color='blue',label="Moment matching of s_1")
plt.plot(xv,norm.pdf(xv,p_s2_m ,np.sqrt(p_s2_s)),'x', color='red',label="Moment matching of s_2")
plt.xlim((-4,4))
plt.xlabel("x")
plt.legend(loc=2, prop={'size': 8})
plt.show()