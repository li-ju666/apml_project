import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

# QUESTION 6


# READ DATA
df = pd.read_csv('SerieA.csv', sep=',', header=0)
data = df.to_numpy()  # nd-array of size (381,6)

# SET UP TEAMS
nb_matches = len(data)
teams_names = np.unique(data[:, 2:4])  # array of names of teams
nb_teams = len(teams_names)
mu = np.zeros(nb_teams)
var = np.ones(nb_teams)
teams = np.stack((teams_names, mu, var))


# DO GIBBS
def importance(s1, s2, y=1):
    s_e = 1
    pr = 1-stats.norm(s1-s2, s_e**0.5).cdf(0) if y > 0 else stats.norm(s1-s2, s_e**0.5).cdf(0)
    return pr


def approx_gaussian(data, importances):
    mean = np.average(data, weights=importances)
    variance = np.average((data-mean)**2, weights=importances)
    return mean, variance


num_samples = 3000
drops = 1000

# initialize
s1 = 0
s1_w = 1

for game_ind in range(nb_matches):
    print(game_ind)
    # result
    if data[game_ind, 4] - data[game_ind, 5] > 0:
        y = 1
    elif data[game_ind, 4] - data[game_ind, 5] < 0:
        y = -1
    else:
        continue  # exit loop since we don't wanna update the skills because it's draw

    # set up so that we can use the code we already have
    team_1 = data[game_ind, 2]
    ind_team_1 = np.where(teams[0, :] == team_1)
    m1 = float(teams[1, ind_team_1])
    v1 = float(teams[2, ind_team_1])

    team_2 = data[game_ind, 3]
    ind_team_2 = np.where(teams[0, :] == team_2)
    m2 = float(teams[1, ind_team_2])
    v2 = float(teams[2, ind_team_2])

    posterior = []
    post_weights = []

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

    # UPDATE THE ARRAY WITH SKILLS
    teams[1, ind_team_1] = m1
    teams[2, ind_team_1] = v1
    teams[1, ind_team_2] = m2
    teams[2, ind_team_2] = v2


print(teams)