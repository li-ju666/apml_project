import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

S_E = (25/6)**2


class Skill:
    def __init__(self, mu=25, var=(25/3)**2):
        self.mu = mu
        self.var = var


class Team:
    def __init__(self, name, skill):
        self.name = name
        self.skill = skill

    def predict(self, another):
        lose_rate = stats.norm.cdf(x=0,
                                   loc=self.skill.mu - another.skill.mu,
                                   scale=(self.skill.var + another.skill.var + S_E) ** 0.5)
        win_rate = 1 - lose_rate
        return win_rate

    def update(self, another, result, num_samples=3000, drops=1000):
        posterior = []
        post_weights = []
        s1 = self.skill.mu
        s1_w = self.skill.var

        for _ in range(num_samples):
            s2 = stats.norm.rvs(another.skill.mu, another.skill.var ** 0.5)
            s2_w = self._importance(s1, s2, result)

            posterior.append((s1, s2))
            post_weights.append((s1_w, s2_w))

            s1 = stats.norm.rvs(self.skill.mu, self.skill.var ** 0.5)
            s1_w = self._importance(s1, s2, result)

        posterior = np.array(posterior)
        post_weights = np.array(post_weights)

        self.skill.mu, self.skill.var = self._approx_gaussian(posterior[drops:, 0], post_weights[drops:, 0])
        another.skill.mu, another.skill.var = self._approx_gaussian(posterior[drops:, 1], post_weights[drops:, 1])

    @staticmethod
    def _approx_gaussian(samples, importances):
        mu = np.average(samples, weights=importances)
        var = np.average((samples - mu) ** 2, weights=importances)
        return mu, var

    @staticmethod
    def _importance(s1, s2, y=1):
        pr = 1 - stats.norm(s1 - s2, S_E ** 0.5).cdf(0) if y > 0 else stats.norm(s1 - s2, S_E ** 0.5).cdf(0)
        return pr


# READ DATA
df = pd.read_csv('SerieA.csv', sep=',', header=0)
data = df.to_numpy()

teams = {name: Team(name, Skill()) for name in np.unique(data[:, 2:4])}

num_predictions = 0
correct_predictions = 0

for match in data:

    team1 = teams[match[2]]
    team2 = teams[match[3]]

    # skip draws
    if match[4] == match[5]:
        continue

    num_predictions += 1
    result = 1 if match[4] > match[5] else -1
    print(f"Before match: {team1.name}: {round(team1.skill.mu, 2)}-{round(team1.skill.var, 2)} vs "
          f"{team2.name}: {round(team2.skill.mu, 2)}-{round(team2.skill.var, 2)}")
    print(f"Team 1 rate: {round(team1.predict(team2), 2)} ==================== Result: {result}")

    prediction = 1 if team1.predict(team2) >= 0.5 else -1

    correct_predictions += prediction == result
    # update skills
    team1.update(team2, result)
    print(f"After match: {team1.name}: {round(team1.skill.mu, 2)}-{round(team1.skill.var, 2)} vs "
          f"{team2.name}: {round(team2.skill.mu, 2)}-{round(team2.skill.var, 2)}")

    print("=======================================================")

for _, v in teams.items():
    print(f"{v.name}: {round(v.skill.mu, 2)}-{round(v.skill.var, 2)}")

print(f"Overall accuracy: {correct_predictions/num_predictions}")
