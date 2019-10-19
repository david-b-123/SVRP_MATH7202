from library import *
from utils import *
from solvers import *
from generate_data import *
from grid_based_simulation import *
from gurobi_tsp_base_solver import *


def testing_function(
        MaxDistance=600,
        InitReq=10,
        RandReq=10,
        Var=20,
        GridSize=100,
        seed=2,
        max_steps=100,
        Model=myopic,
        Plot=False,
        Plot_annotate=False,
        Grid_train=True):
    np.random.seed(seed)
    all_dict = SimulateData(gridsize=GridSize, maxdistance=MaxDistance, initreq=InitReq, randreq=RandReq, var=Var,
                            max_steps=max_steps)
    print('Number of stochastic requests', len(all_dict) - InitReq)
    total_score = 0
    start = time.time()
    # print(Model.__name__)
    print('Distance:', MaxDistance,
          'InitReq', InitReq,
          'RandReq', RandReq)
    start = time.time()

    if Grid_train == True:
        all_dict, grid = grid_train(all_dict, MaxDistance, GridSize, seed)
        print(all_dict)

        num_added, new_cost, new_tour = Model(all_dict, maxdist=MaxDistance, seed=seed)
        # print("outernewtour",new_tour)
        end = time.time()
        run_walltime = end - start

        # plot_current(all_dict, new_tour, 0, Plot_annotate)

        print('Run time', round(run_walltime, 2))
        print('Total score', len(set(new_tour)) - InitReq)
        return [Model.__name__, len(set(new_tour)) - InitReq, round(run_walltime, 2), MaxDistance, InitReq,
                len(all_dict) - InitReq, Var,
                max_steps, GridSize, new_cost, len(set(new_tour))]

    else:
        for timestep in range(0, max_steps):
            if timestep == 0:
                path = []
                current_cost = 0

            num_added, new_cost, new_tour = Model(path, timestep, all_dict, maxdist=MaxDistance, seed=seed,
                                                  current_cost=current_cost)
            # print('time',timestep,'num',num_added,'cost',new_cost)
            # print('new path', new_tour)

            path = new_tour
            current_cost = new_cost

            # print('time',timestep,'score',total_score,'cost',new_cost)
            # print('         ')
            # if new_cost >= MaxDistance:
            #   break

            if timestep - 1 == len(new_tour):
                end = time.time()
                run_walltime = end - start
                print('Run time', round(run_walltime, 2))
                print('Total score', len(new_tour) - InitReq)

                if Plot == True:
                    plot_current(all_dict, new_tour, timestep, Plot_annotate)
                return [Model.__name__, len(new_tour) - InitReq, round(run_walltime, 2), MaxDistance, InitReq,
                        len(all_dict) - InitReq, Var, max_steps, GridSize, new_cost, len(new_tour)]

            else:
                if Plot == True:
                    plot_current(all_dict, new_tour + [0], timestep, Plot_annotate)
            if timestep == max_steps - 2:
                end = time.time()
                run_walltime = end - start
                print('Run time', round(run_walltime, 2))
                print('Total score', len(new_tour) - InitReq)
                return [Model.__name__, len(new_tour) - InitReq, round(run_walltime, 2), MaxDistance, InitReq, RandReq,
                        Var, max_steps, GridSize, new_cost, len(new_tour)]


Model_list_1 = [MyopicKNN, MCTS_grid_based]
Model_list_2 = [myopic, random_insertion, distance_model_1, distance_model_2, distance_model_3]

RandList = [50]
LenList = [500]
InitList = [15]
VarList = [10]


def getResults(models_david, models_cameron, rands, seed_set):
    final_data = np.empty((0, 11))
    final_data = np.vstack((final_data,
                            ['Model', 'Score', 'Time', 'MaxDistance', 'Initial Customers', 'Stochastic Customers',
                             'Var', 'MaxSteps', 'GridSize', 'Final Cost', 'Length Tour']))

    for seed_iter in range(seed_set):
        for length in LenList:
            for var in VarList:
                for init in InitList:
                    for m in models_david:
                        for rand in rands:
                            line = testing_function(RandReq=rand,
                                                    InitReq=init,
                                                    Model=m,
                                                    Var=var,
                                                    MaxDistance=length,
                                                    Grid_train=True,
                                                    seed=seed_iter)
                            final_data = np.vstack((final_data, line))

        for length in LenList:
            for var in VarList:
                for init in InitList:
                    for m in models_cameron:
                        for rand in rands:
                            line = testing_function(RandReq=rand,
                                                    InitReq=init,
                                                    Model=m,
                                                    Var=var,
                                                    MaxDistance=length,
                                                    Grid_train=False,
                                                    seed=seed_iter)
                            final_data = np.vstack((final_data, line))
    print('run final data', final_data)

    return final_data


def VisualiseResults():
    results = getResults()
    fig = plt.figure(figsize=10)

    for i in results.keys():
        if i != 'Title':
            pass


result_data = getResults(Model_list_1, Model_list_2, RandList, 10)

# np.savetxt("./result_svdrp.csv", result_data, fmt='%s', delimiter=",")

