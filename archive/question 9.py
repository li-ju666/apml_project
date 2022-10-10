import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats


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


m1 = 0 # Mean of p(s_1)
s1 = 1 # Variance of p(s_1)
m2 = 0 # Mean of p(s_2)
s2 = 1 # Variance of p(s_2)
s = 1 # Variance of p(t|s_1)
y0 = 1 # Measurement

# Message msg3 and msg4
msg3_m = m1 # mean of message
msg3_s = s1 # variance of message
msg4_m = m2 # mean of message
msg4_s = s2 # variance of message

# Message msg5 and msg6
msg5_m = msg3_m # mean of message
msg5_s = msg3_s # variance of message
msg6_m = msg4_m # mean of message
msg6_s = msg4_s # variance of message

# Message msg7
msg7_m = m1 - m2
msg7_s = s + s1 + s2

# p_t
a, b = 0, np.Inf
p_t_m, p_t_s = truncGaussMM(a, b, msg7_m, msg7_s)

# Message msg8
msg8_m, msg8_s = divideGauss(p_t_m, p_t_s, msg7_m, msg7_s)

# Message msg9 and msg10
msg9_m = m2+msg8_m
msg9_s = s+s2+msg8_s
msg10_m = m1-msg8_m
msg10_s = s+s1+msg8_s

# p_s1 and p_s2
p_s1_m, p_s1_s = mutiplyGauss(m1, s1, msg9_m, msg9_s)
p_s2_m, p_s2_s = mutiplyGauss(m2, s2, msg10_m, msg10_s)


# PLOT
xv = np.linspace(-4,4,3000)
plt.plot(xv,norm.pdf(xv,p_s1_m ,np.sqrt(p_s1_s)),label="Moment matching of s_1")
plt.plot(xv,norm.pdf(xv,p_s2_m ,np.sqrt(p_s2_s)),label="Moment matching of s_2")
plt.xlim((-4,4))
plt.xlabel("x")
plt.legend(loc=2, prop={'size': 8})
plt.show()
