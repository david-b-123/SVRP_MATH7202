from utils import *
from library import *
from simulate_data import *
from test_tsp import tsp_gurobi

# CTRL-F to find the following

# 1 - myopic
# 2 - random insertion
# 3 - distance model 1
# 4 - distance model 2
# 5 - distance model 3
# 6 - myopic-knn
# 7 - MCTS grid-based



# 1 - myopic

def myopic(current_path, time_step, all_dict, maxdist, seed, current_cost):
    "This model performs a greedy myopic addition of customers into the tour "
    np.random.seed(seed)
    visited = current_path[:time_step]
    not_visited = current_path[time_step:]
    new_points = get_new_points(time_step, current_path, all_dict)
    combinations = get_new_combinations(new_points)

    found_feasible = False
    best_num = 0
    best_known = [0, maxdist, current_path]

    best_for_num = {}
    for i in range(50):
        best_for_num[i] = [maxdist, 0]

    for i in combinations:
        num_new = len(i)

        if len(i) == 0:
            return [0, current_cost, current_path]
        if len(i) > 0:
            if num_new == 0:
                print('no improvement')

            if best_for_num[num_new][1] > 20:
                continue
            if best_num > num_new:
                break
            new_current_path = current_path[:]
            new_current_path.sort()
            potential_customers = new_current_path + list(i)
            potential_tour, cost = tsp_gurobi(visited, potential_customers, all_dict)

            if cost > maxdist:

                best_for_num[num_new][1] += 1  # raise number of attempts at this number

            elif cost < best_for_num[num_new][0]:
                best_num = num_new
                best_for_num[num_new][0] = cost
                best_known = [best_num, cost, potential_tour]

        return best_known





# 2 - random insertion

def random_insertion(current_path, time_step, all_dict, maxdist, seed, current_cost):
    "This model performs a random addition of customers into the tour "

    np.random.seed(seed)
    random.seed(seed)

    visited = current_path[:time_step]
    not_visited = current_path[time_step:]
    new_points = get_new_points(time_step, current_path, all_dict)
    # combinations = get_new_combinations(new_points)
    new_points_copy = new_points[:]

    # print('new_points',new_points)
    random.shuffle(new_points_copy)

    num_to_incl = int(np.random.uniform(1, len(new_points)))

    rand_to_incl = new_points_copy[:num_to_incl]

    new_current_path = current_path[:]
    new_current_path.sort()

    if time_step == 0:
        potential_tour, cost = tsp_gurobi(visited, new_points, all_dict)
    else:

        potential_customers = new_current_path + list(rand_to_incl)

        potential_tour, cost = tsp_gurobi(visited, potential_customers, all_dict)

    if cost < maxdist:
        # print('included',num_to_incl,rand_to_incl)
        return [num_to_incl, cost, potential_tour]
    else:
        # print('infeasible')
        potential_tour, cost = tsp_gurobi(visited, current_path, all_dict)

        return [0, cost, potential_tour]








# 3 - distance model 1


def distance_model_1(current_path, time_step, all_dict, maxdist, seed, current_cost):
    "This model adds in points only if it does not increase the current cost by a certain amount"

    np.random.seed(seed)
    random.seed(seed)

    visited = current_path[:time_step]
    not_visited = current_path[time_step:]
    new_points = get_new_points(time_step, current_path, all_dict)

    combinations = get_new_combinations(new_points)

    max_increase = maxdist / 10  # max increase is 10% of the max distance

    best_num = 0
    best_known = [0, maxdist, current_path]

    best_for_num = {}
    for i in range(100):
        best_for_num[i] = [maxdist, 0]

    if time_step == 0:
        potential_tour, cost = tsp_gurobi(visited, new_points, all_dict)
        if cost <= maxdist:
            return [0, cost, potential_tour]
        else:
            return 'Initial tour is infeasible'

    for i in combinations:
        if len(i) == 0:
            return [0, current_cost, current_path]
        if len(i) > 0:
            num_new = len(i)
            if best_for_num[num_new][1] > 3:
                continue
            if best_num > num_new:
                break
            new_current_path = current_path[:]
            new_current_path.sort()
            potential_customers = new_current_path + list(i)
            potential_tour, cost = tsp_gurobi(visited, potential_customers,
                                              all_dict)  # once it's converted to the data representation it should work

            if cost > maxdist:

                best_for_num[num_new][1] += 1  # raise number of attempts at this number

            elif cost > current_cost + max_increase:

                best_for_num[num_new][1] += 1

            elif cost < best_for_num[num_new][0]:
                best_num = num_new
                best_for_num[num_new][0] = cost
                best_known = [best_num, cost, potential_tour]

        return best_known







# 4 - distance model 2

def distance_model_2(current_path, time_step, all_dict, maxdist, seed, current_cost):
    """This model calculates the distance from each new point
    in the not_visited set of points in the tour
    and orders inclusion into the model based on the smallest distance to include. """

    np.random.seed(seed)
    random.seed(seed)

    visited = current_path[:time_step]
    not_visited = current_path[time_step:]

    # print('not_visited',not_visited)

    # for i in not_visited:
    #    print(all_dict[i][0])

    new_points = get_new_points(time_step, current_path, all_dict)
    # for i in new_points:
    #    print(all_dict[i])

    if time_step > 0:
        unvisited_dist_to_new = {(i, j):
                                     math.sqrt(sum((all_dict[i][0][k] - all_dict[j][0][k]) ** 2 for k in range(2)))
                                 for i in not_visited for j in new_points}

        # print(unvisited_dist_to_new)
        sorted_dist = sorted(unvisited_dist_to_new, key=unvisited_dist_to_new.get)
        # print(sorted_dist)

        to_add = []

        for i in sorted_dist:
            if i[1] not in to_add:
                to_add.append(i[1])
            else:
                del unvisited_dist_to_new[i]
        # print(to_add)

        to_include = []
        sorted_dist2 = sorted(unvisited_dist_to_new, key=unvisited_dist_to_new.get)
        for i in sorted_dist2:
            if unvisited_dist_to_new[i] < 10:
                to_include.append(i[1])

        new_current_path = current_path[:]
        new_current_path.sort()
        potential_customers = new_current_path + to_include

        potential_tour, cost = tsp_gurobi(visited, potential_customers,
                                          all_dict)  # once it's converted to the data representation it should work

        if cost > maxdist:
            return [0, maxdist, current_path]
        else:
            # print(cost,'to_include',len(to_include))
            # print('len current path',len(current_path))
            return [len(to_include), cost, potential_tour]

    if time_step == 0:
        potential_tour, cost = tsp_gurobi(visited, new_points, all_dict)
        if cost <= maxdist:
            return [0, cost, potential_tour]
        else:
            return 'Initial tour is infeasible'










# 5 - distance model 3

def distance_model_3(current_path, time_step, all_dict, maxdist, seed,current_cost):
    "This model performs a search of all points that are within X distance of the original points "
    np.random.seed(seed)
    visited = current_path[:time_step]
    not_visited = current_path[time_step:]
    new_points = get_new_points(time_step, current_path, all_dict)
    combinations = get_new_combinations(new_points)

    found_feasible = False
    best_num = 0
    best_known = [0, maxdist, current_path]

    best_for_num = {}
    for i in range(50):
        best_for_num[i] = [maxdist, 0]

    for i in combinations:
        num_new = len(i)
        if len(i) > 0:
            if num_new == 0:
                print('no improvement')
            # sets a limit for trying to fit a given number in
            if best_for_num[num_new][1] > 20:
                continue
            if best_num > num_new:
                break
            new_current_path = current_path[:]
            new_current_path.sort()
            potential_customers = new_current_path + list(i)
            if time_step>1:
                potential_customer_data = [all_dict[j][0] for j in potential_customers]

                n = len(potential_customers)

                points = [(potential_customer_data[i][0], potential_customer_data[i][1]) for i in range(n)]

                dist = {(i, j):
                            math.sqrt(sum((points[i][k] - points[j][k]) ** 2 for k in range(2)))
                        for i in range(n) for j in range(i)}

                j_ids = [j for i in range(n) for j in range(i) if (dist[i,j]<5)]

                potential_customers = potential_customers + j_ids

                potential_customers = list(set(potential_customers))

            potential_tour, cost = tsp_gurobi(visited, potential_customers,
                                              all_dict)  # once it's converted to the data representation it should work

            if cost > maxdist:

                best_for_num[num_new][1] += 1  # raise number of attempts at this number

            elif cost < best_for_num[num_new][0]:
                best_num = num_new
                best_for_num[num_new][0] = cost
                best_known = [best_num, cost, potential_tour]

        return best_known










# 6 - myopic-knn


def MyopicKNN(all_dict, maxdist, seed):
    "This model performs a greedy myopic addition of customers into the tour "
    np.random.seed(seed)

    len_tour, potential_tour, cost = mcts_inner_for_MCTS_v1(all_dict,maxdist)  # once it's converted to the data representation it should work

    return len_tour,potential_tour,cost

def mcts_inner_for_MCTS_v1(all_dict,maxdist):
    # Note: Visited refers to the customers already visited in the path (i.e. need to have the same edges in the new tour)
    # original_customer just refers to the total number of customers that a tour is being generated for
    n = len(all_dict)

    # scores = [all_dict[j][3] for j in range(n)]

    potential_customer_data = [all_dict[j][0] for j in range(len(all_dict))]
    potential_customer_time = [all_dict[j][1] for j in range(len(all_dict))]


    np.random.seed(2)
    # print('InitReq',InitReq,'len',len(original_customer))

    # len is set as the number of customers (i.e. the length of the tour that needs to be generated)
    # print("////////////")
    # print(potential_customer_id)
    # print('Visited',Visited)

    points = [(potential_customer_data[i][0],potential_customer_data[i][1],potential_customer_time[i]) for i in range(n)]
    # print(points)

    # Dictionary of Euclidean distance between each pair of points

    dist = {(i, j):
                math.sqrt(sum((points[i][k] - points[j][k]) ** 2 for k in range(2)))
            for i in range(n) for j in range(n)}

    for i in range(n):
        for j in range(n):
            if points[i][2][0]!=0 and points[j][2][0]!=0:
                if points[i][2][0]  > points[j][2][0]:
                    dist[i,j]=10**5

    for i in range(n):
        for j in range(n):
            if i==j:
                dist[i,j]=10**5

    # print('dist',dist)

    true_dist = {(i, j):
                math.sqrt(sum((points[i][k] - points[j][k]) ** 2 for k in range(2)))
            for i in range(n) for j in range(n)}



    converged=False
    initial=True
    cost_tour=0
    tour=[0,0]
    current=0
    potential_tour=tour
    potential_cost_tour=cost_tour

    while converged==False:

        print("path",tour)
        print("cost",cost_tour)
        if potential_cost_tour>maxdist:
            print(tour)
            break
        if potential_cost_tour<maxdist:
            tour=potential_tour
            cost_tour=potential_cost_tour
        if initial:
            proposals=np.argsort([dist[current, i] for i in range(n)])
            for proposal in proposals:
                if proposal not in tour:
                    tour = tour[0:-1] + [proposal] + [tour[-1]]
                    break
            current=tour[-2]
            cost_tour=InternalCost(tour,true_dist)
            dist[tour[-3],tour[-2]]=10**5
            initial=False
        if potential_cost_tour<maxdist:
            proposals=np.argsort([dist[current, i] for i in range(n)])
            for proposal in proposals:
                if proposal not in tour:
                    potential_tour = tour[0:-1] + [proposal] + [tour[-1]]
                    break
            current=potential_tour[-2]
            dist[potential_tour[-3],potential_tour[-2]]=10**5
            potential_cost_tour=InternalCost(potential_tour,true_dist)
    return len(tour), cost_tour, tour


def InternalCost(Path, D):
    return sum(D[Path[i - 1],Path[i]] for i in range(len(Path)))







# 7 - MCTS grid-based

def MCTS_grid_based(all_dict, maxdist, seed):
    "This model performs a greedy myopic addition of customers into the tour "
    np.random.seed(seed)
    # print(all_dict)
    len_tour, potential_tour, cost = mcts_inner_for_MCTS_v3(all_dict,maxdist)  # once it's converted to the data representation it should work

    return len_tour,potential_tour,cost

def mcts_inner_for_MCTS_v3(all_dict,maxdist):
    # Note: Visited refers to the customers already visited in the path (i.e. need to have the same edges in the new tour)
    # original_customer just refers to the total number of customers that a tour is being generated for
    n = len(all_dict)

    scores = [all_dict[j][3] for j in range(n)]
    # scores = [j for j in range(n)]

    potential_customer_data = [all_dict[j][0] for j in range(len(all_dict))]
    potential_customer_time = [all_dict[j][1] for j in range(len(all_dict))]


    np.random.seed(2)
    # print('InitReq',InitReq,'len',len(original_customer))

    # len is set as the number of customers (i.e. the length of the tour that needs to be generated)
    # print("////////////")
    # print(potential_customer_id)
    # print('Visited',Visited)

    points = [(potential_customer_data[i][0],potential_customer_data[i][1],potential_customer_time[i]) for i in range(n)]
    # print(points)

    # Dictionary of Euclidean distance between each pair of points

    dist = {(i, j):
                math.sqrt(sum((points[i][k] - points[j][k]) ** 2 for k in range(2)))
            for i in range(n) for j in range(n)}

    for i in range(n):
        for j in range(n):
            if points[i][2][0]!=0 and points[j][2][0]!=0:
                if points[i][2][0]  > points[j][2][0]:
                    dist[i,j]=10**5

    for i in range(n):
        for j in range(n):
            if i==j:
                dist[i,j]=10**5

    # print('dist',dist)

    true_dist = {(i, j):
                math.sqrt(sum((points[i][k] - points[j][k]) ** 2 for k in range(2)))
            for i in range(n) for j in range(n)}



    converged=False
    initial=True
    cost_tour=0
    tour=[0,0]
    current=0
    potential_tour=tour
    potential_cost_tour=cost_tour
    MCTS_run=True
    ALL_proposal=True
    proposals=[i for i in range(len(all_dict))]
    final_matrix=np.ones((n,n))

    len_tour, cost_tour, tour = run_policy(all_dict,maxdist,n,scores,dist,true_dist,current,tour,cost_tour,proposals,MCTS_run,ALL_proposal,initial)
    return len_tour, cost_tour, tour

def run_policy(all_dict,maxdist,n,scores,dist,true_dist,current,tour,cost_tour,proposals,MCTS_run,ALL_proposal,initial):
    potential_tour=tour
    potential_cost_tour=cost_tour
    converged=False
    current_tour=[0,0]

    best_score=len(current_tour)

    total_scores = [
        scores[i] * dist[0,i]
        for i in range(n)]
    proposals = np.argsort(total_scores)[::-1]
    current_proposals=proposals

    while converged==False:
        # print("path",potential_tour)
        # print("cost",cost_tour)
        if potential_cost_tour>maxdist:
            print("if 1")
            return(len(tour),cost_tour,tour)
        if potential_cost_tour<maxdist and initial==False:
            print("if 2")
            tour=potential_tour
            cost_tour=potential_cost_tour
            if len(current_tour)<len(new_tour):
                current_tour = np.copy(new_tour)
                current_proposals = np.copy(new_proposals)
                best_score=len(current_proposals)

        if MCTS_run==True:
            print("if 4")
            current=tour[-2]
            print('current top',current)
            print('proposals top',current_proposals)
            new_proposals,new_tour,converged = MCTS(all_dict,maxdist,n,scores,dist,true_dist,current,tour,cost_tour,current_proposals,best_score)
            if converged==True:
                return(len(new_tour),InternalCost(new_tour,true_dist),new_tour)
            if len(current_tour)>len(new_tour):
                final_proposals=current_proposals
            else:
                final_proposals=new_proposals
        if potential_cost_tour<maxdist:
            print("if 5 and 6")
            for proposal in final_proposals:
                if proposal not in tour:
                    potential_tour = tour[0:-1] + [proposal] + [tour[-1]]
                    break
            current=potential_tour[-2]
            cost_tour=InternalCost(potential_tour,true_dist)
            print('top pot tour',potential_tour)
            dist[potential_tour[-3],potential_tour[-2]]=10**5
            dist[potential_tour[-2],potential_tour[-3]]=10**5
            initial=False


def run_full_policy(maxdist, n, scores,dist, true_dist, current, tour, cost_tour, proposals, MCTS_run, ALL_proposal,
                   initial):
        potential_tour = tour
        potential_cost_tour = cost_tour
        converged = False
        # print('runs run full policy')

        while converged == False:
            # print("path",tour)
            # print("cost",cost_tour)
            if potential_cost_tour > maxdist:
                print("if f1")
                return len(tour), cost_tour, tour
            if potential_cost_tour < maxdist and ~initial:
                print("if f2")
                tour = potential_tour
                cost_tour = potential_cost_tour
                total_scores = [
                    dist[current, i]*np.log(dist[i,0]*len(tour))+dist[current,i]*np.log((1+scores[current])/(1+scores[i]))
                    for i in range(n)]
                proposals = np.argsort(total_scores)
            if initial == True:
                print("if f4")
                for proposal in proposals:
                    if proposal not in tour:
                        tour = tour[0:-1] + [proposal] + [tour[-1]]
                        break
                current = tour[-2]
                cost_tour = InternalCost(tour, true_dist)
                dist[tour[-3], tour[-2]] = 10 ** 5
                dist[tour[-2], tour[-3]] = 10 ** 5
                initial = False
            if potential_cost_tour < maxdist:
                print("if f5")
                for proposal in proposals:
                    if proposal not in tour:
                        potential_tour = tour[0:-1] + [proposal] + [tour[-1]]
                        break
                current = potential_tour[-2]
                dist[potential_tour[-3], potential_tour[-2]] = 10 ** 5
                dist[potential_tour[-2], potential_tour[-3]] = 10 ** 5
                potential_cost_tour = InternalCost(potential_tour, true_dist)

        return len(tour), cost_tour, tour



def InternalCost(Path, D):
    return sum(D[Path[i - 1],Path[i]] for i in range(len(Path)))

def MCTS(all_dict,maxdist,n,scores,D,true_dist,current,tour,cost_tour,proposals,best_score):
    # print("runs a MCTS")
    sample_MCTS={}
    for i in range(np.min((10,len(proposals)))):
        proposal=[proposals[i]]
        print('proposal', proposal)
        len_tour, new_cost_tour, new_tour = run_full_policy(maxdist,n,scores,D,true_dist,current,tour,cost_tour,proposal,False,False,False)
        sample_MCTS[i]=[len_tour,new_cost_tour,new_tour]


    print('sample_MCTS',[item[0] for key,item in sample_MCTS.items()])
    final_proposal=sample_MCTS[np.argmax([item[0] for key,item in sample_MCTS.items()])]
    print("final_proposal",final_proposal[2])

    plot_current(all_dict, tour, len(tour)-1, False)

    print("mcts len tour",len(tour))
    print('mcts len propo', len(proposals))
    print("mcts len new tour",final_proposal[0])
    if ((final_proposal[0]-1)==best_score):
        return final_proposal,final_proposal[2],True
    else:
        return final_proposal[2][(len(tour)-1):],final_proposal[2],False
