from test_tsp import tsp_gurobi
from library import *

def Distance(p1, p2):
    return int(math.hypot(p1[0] - p2[0], p1[1] - p2[1]) + 0.5)



def Cost(Path, G, D):
    return sum(D[Path[i - 1]][Path[i]] for i in G)


def plot_current(all_dict, Path, t, anno):
    if anno == True:
        for key, item in all_dict.items():
            plt.annotate(key, (item[0][0], item[0][1]))

    plt.scatter([all_dict[Path[i]][0][0] for i in range(len(Path))],
                [all_dict[Path[i]][0][1] for i in range(len(Path))], color='grey')

    plt.scatter(all_dict[Path[0]][0][0], all_dict[Path[0]][0][1], color="red", s=300)
    plt.scatter(all_dict[Path[t]][0][0], all_dict[Path[t]][0][1], color="orange", s=300)

    plt.plot([all_dict[Path[i]][0][0] for i in range(len(Path))],
             [all_dict[Path[i]][0][1] for i in range(len(Path))])

    fname = os.path.join("./plot_temp/" + str(t) + ".png")
    plt.savefig(fname, format="png")
    plt.clf()


def Travel_Update(current_path, optimal_policy):
    new_path = np.vstack((current_path,
                          optimal_policy[0]
                          )
                         )
    return new_path


def get_new_points(t, current_path, all_dict):
    timestamp = [int(item[1][0]) for key, item in all_dict.items()]

    new_points = []

    for i in range(len(timestamp)):
        if timestamp[i] == t:
            new_points.append(i)

    return new_points


def get_new_combinations(new_points):
    combinations = []
    for i in range(len(new_points) + 1):
        for subset in itertools.combinations(new_points, i):
            combinations.append(subset)

    return reversed(combinations)


def get_new_data(new_ID, all_dict):
    if len(new_ID) == 0:
        return np.empty((0, 2))
    a = np.vstack((np.asarray([all_dict[j][0] for j in new_ID])))

    return a


