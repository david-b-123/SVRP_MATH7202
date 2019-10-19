from utils import *
from library import *
from test_tsp import tsp_gurobi

from solvers import MyopicKNN


def grid_train(all_dict,maxdist,GridSize,seed):
    # x_range = [np.min([item[0][0] for key, item in all_dict.items()]),
    #            np.max([item[0][0] for key, item in all_dict.items()])]
    # y_range = [np.min([item[0][1] for key, item in all_dict.items()]),
    #            np.max([item[0][1] for key, item in all_dict.items()])]

    interval=30
    x_grid=np.linspace(start=0, stop=GridSize, num=interval)
    y_grid=np.linspace(start=0, stop=GridSize, num=interval)

    grid = np.ones((interval, interval))
    partition_points_xy = np.zeros((interval, interval))
    for i in range(30):

        # print(i)
        sim_points = {}
        partition = {}

        for key, item in all_dict.items():
            if (item[2]):
                point = item[0]
                x, y = point[0], point[1]
                x_partition=np.max(np.argwhere(x>=x_grid))
                y_partition=np.max(np.argwhere(y>=y_grid))

                sim_points[key] = [[x,y], [0], [x_partition,y_partition]]

                for j in range(len(sim_points),len(sim_points)+10):
                    np.random.seed(i*j+i+j)
                    random.seed(i*j+i+j)
                    x,y=np.abs(np.random.normal(x,5)),np.abs(np.random.normal(y,5))

                    x_partition = np.max(np.argwhere(x >= x_grid))
                    y_partition = np.max(np.argwhere(y >= y_grid))
                    sim_points[j] = [[x, y], [0], [x_partition, y_partition]]

        print("before model3",i)
        num_added, new_cost, new_tour = MyopicKNN(sim_points, maxdist=maxdist, seed=seed)
        print("after model3",i)

        for key2,item2 in sim_points.items():
            if key2 in new_tour:
                partition_points_xy[item2[2][0],item2[2][1]]+=1

        print(partition_points_xy)
        for x_partition in range(interval):
            for y_partition in range(interval):
                grid[x_partition][y_partition] = partition_points_xy[x_partition,y_partition]


    new_all_dict={}
    for key,item in all_dict.items():
        # print("all_dict,item",item)
        x, y = item[0][0],item[0][1]
        x_partition = np.max(np.argwhere(abs(x) >= x_grid))
        y_partition = np.max(np.argwhere(abs(y) >= y_grid))
        item=item+[np.sum(partition_points_xy[np.max((x_partition-3,0)):np.min((x_partition+3,interval)),np.max((y_partition-3,0)):np.min((y_partition+3,interval))])]
        new_all_dict[key]=item



    return new_all_dict, grid
