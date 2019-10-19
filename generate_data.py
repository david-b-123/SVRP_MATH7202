from library import *
from utils import *

def SimulateData(gridsize, maxdistance, initreq, randreq, var, max_steps):
    # Gridsize (int): size of the initial
    # Maxdistance (int): maximum travel distance allowed
    # initreq (int): initial number of requests (must be serviced)
    # randreq (int): presimulated stochastic requests (may be serviced)
    # var (float): variance for the stochastic requests
    # time_new_customer gives a time to each of the stochastic requests
    # grid is the original and stochastic vars embedded in a grid

    # returns original_customer (list), new_customer (list), time_new_customer (list), grid (list, list)

    grid = [[0, 0], [gridsize, gridsize]]
    # print("gridsize",gridsize)
    fuel_distance = maxdistance

    all_dict = {}

    original_customer = [np.random.uniform(0, gridsize), np.random.uniform(0, gridsize)]
    new_customer = [np.random.uniform(0, gridsize), np.random.uniform(0, gridsize)]
    for i in range(initreq - 1):
        original_customer = np.vstack((original_customer,
                                       [np.random.uniform(0, gridsize), np.random.uniform(0, gridsize)]
                                       )
                                      )
        for j in range(randreq - 1):
            new_customer = np.vstack((new_customer,
                                      np.random.normal(original_customer[i], var))
                                     )

    time_new_customer = np.zeros([original_customer.shape[0], 1])
    for k in range(new_customer.shape[0]):
        time_new_customer = np.vstack((time_new_customer,
                                       int(np.random.uniform(0, max_steps))
                                       )
                                      )
    all_current_customers = np.vstack((original_customer, new_customer))

    for i in range(all_current_customers.shape[0]):
        all_dict[i] = [np.ndarray.tolist(all_current_customers[i, :]), np.ndarray.tolist(time_new_customer[i]),
                       i < initreq]

    # print("all_dict",all_dict)
    return all_dict

