def tsp_gurobi(Visited,
               potential_customer_id, all_dict):
    # Note: Visited refers to the customers already visited in the path (i.e. need to have the same edges in the new tour)
    # original_customer just refers to the total number of customers that a tour is being generated for
    potential_customer_data = [all_dict[j][0] for j in potential_customer_id]

    # test
    # print(original_customer)
    if len(Visited) > 0:
        original_position = Visited[0]
        current_position = Visited[-1]

        # print('original position',original_position)
        # print('current position', current_position)

    np.random.seed(2)
    # print('InitReq',InitReq,'len',len(original_customer))

    # len is set as the number of customers (i.e. the length of the tour that needs to be generated)
    n = len(potential_customer_id)
    # print("////////////")
    # print(potential_customer_id)
    # print('Visited',Visited)

    points = [(potential_customer_data[i][0], potential_customer_data[i][1]) for i in range(n)]
    # print(points)

    # Dictionary of Euclidean distance between each pair of points

    dist = {(i, j):
                math.sqrt(sum((points[i][k] - points[j][k]) ** 2 for k in range(2)))
            for i in range(n) for j in range(i)}

    # print(dist)
    m = Model()
    m.setParam("OutputFlag", 0)
    # creates a user value in the model, allowing it to be called by the callback function subtourlim
    m._n = n
    # print('m._n set',m._n)

    # Create variables

    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')
    for i, j in vars.keys():
        vars[j, i] = vars[i, j]  # edge in opposite direction

    # You could use Python looping constructs and m.addVar() to create
    # these decision variables instead.  The following would be equivalent
    # to the preceding m.addVars() call...
    #
    # vars = tupledict()
    # for i,j in dist.keys():
    #   vars[i,j] = m.addVar(obj=dist[i,j], vtype=GRB.BINARY,
    #                        name='e[%d,%d]'%(i,j))

    # Add degree-2 constraint
    # print(vars[Visited[0],Visited[1]])
    # for i in range(len(Visited)-1):
    #    print(i,Visited[i],Visited[i+1])
    # print('ran to here')

    m.addConstrs(vars.sum(i, '*') == 2 for i in range(n))
    if len(Visited) > 0:
        for i in range(len(Visited) - 1):
            # print(Visited[i],Visited[i+1])
            a = potential_customer_id.index(Visited[i])
            b = potential_customer_id.index(Visited[i + 1])
            # m.addConstr(vars[Visited[i],Visited[i+1]]==1)
            if a != b:
                m.addConstr(vars[a, b] == 1)

    # Using Python looping constructs, the preceding would be...
    #
    # for i in range(n):
    #   m.addConstr(sum(vars[i,j] for j in range(n)) == 2)

    # Optimize model

    m._vars = vars
    m.Params.lazyConstraints = 1

    # print('pos', n)

    m.optimize(subtourelim)

    vals = m.getAttr('x', vars)
    selected = tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)

    # print(selected)
    tour = subtour(selected, n)
    # print('len tour',len(tour),'n',n)
    assert len(tour) == n

    # print('')
    # print('Optimal tour: %s' % str(tour))
    # print('Optimal cost: %g' % m.objVal)
    # print('')
    # for v in m.getVars():
    #    if v.x > 0.5:
    #        print('%s %g' % (v.varName, v.x))

    tour_id = []

    for i in tour:
        tour_id.append(potential_customer_id[i])

    # print('potential_customer_id',potential_customer_id)
    # print('tour',tour)
    # print('tour_id',tour_id)
    return tour_id, m.objVal


def subtourelim(model, where):
    # print('subtourlim',model._n)
    if where == GRB.Callback.MIPSOL:
        # make a list of edges selected in the solution
        vals = model.cbGetSolution(model._vars)
        selected = tuplelist((i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5)
        # find the shortest cycle in the selected edge list
        tour = subtour(selected, model._n)
        # print('tour in subtourlim',tour)
        if len(tour) < model._n:
            # add subtour elimination constraint for every pair of cities in tour
            model.cbLazy(quicksum(model._vars[i, j]
                                  for i, j in itertools.combinations(tour, 2))
                         <= len(tour) - 1)


# Given a tuplelist of edges, find the shortest subtour

def subtour(edges, n):
    # print('subtour')
    unvisited = list(range(n))
    cycle = range(n + 1)  # initial length has 1 more city
    while unvisited:  # true if list is non-empty
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, '*') if j in unvisited]
        if len(cycle) > len(thiscycle):
            cycle = thiscycle
    # print("cycle inner subtour" , cycle)
    return cycle

