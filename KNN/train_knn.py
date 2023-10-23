import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

points = {'blue': [[2, 4], [1, 3], [2, 3], [3, 2], [2, 1]],
          'red': [[5, 6], [4, 5], [4, 6], [6, 6], [5, 4]]}


new_point = [3,3]

def euclidean_dist(p, q):
    return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))


class KNN:
    def __init__(self, k=3):
        self.k = k
        self.point = None
    
    def fit(self, points):
        self.points = points
    
    def predict(self, new_point):
        distances = []
        for category in self.points:
            for points in self.points[category]:
                distance = euclidean_dist(points, new_point)
                distances.append([distance, category])
            
        categories = [category[1] for category in sorted(distances)[:self.k]]
        count = Counter(categories).most_common(1)[0][0]
        return count

clf = KNN()
clf.fit(points)
print("Prediction: ", clf.predict(new_point))

#Visualize
ax = plt.subplot()
ax.grid(True, color='#323232')

ax.set_facecolor('black')
ax.figure.set_facecolor('#121212')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

for point in points['blue']:
    ax.scatter(point[0], point[1], color="blue", s=60)
for point in points['red']:
    ax.scatter(point[0], point[1], color="red", s=60)

new_class = clf.predict(new_point)
color = "red" if new_class == "red" else "blue"
ax.scatter(new_point[0], new_point[1], color=color, marker="*", s=200, zorder=100)
for point in points['blue']:
    ax.plot([new_point[0], point[0]], [new_point[1], point[1]], color="blue", linestyle="--", linewidth=1)
for point in points['red']:
    ax.plot([new_point[0], point[0]], [new_point[1], point[1]], color="red", linestyle="--", linewidth=1)
print("DOne")
plt.show()