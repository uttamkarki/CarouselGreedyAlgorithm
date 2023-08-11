import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

start_time = time.process_time()

#parameter for carousel greedy
beta = 0.4
alpha = 2
iteration = 100
history= []

dataset1 = pd.read_csv("TSP_Instance2.csv")
dataset = dataset1.to_numpy()
dummy_data = dataset
gbest_obj = 10000000


for itr in range(iteration):
    total_distance = 0
    dis_greedy = 0

    solution = np.zeros(len(dataset1) + 1, dtype=int)
    solution[0] = np.random.randint(low=1, high=len(solution), size=1)


    def generate_solution(position, n, solution):
        node = n
        i = position
        distance = dataset[node]
        if (i + 2) == len(solution):
            solution[i + 1] = solution[0]
        else:
            for j in range(len(distance)):
                value = (np.argsort(distance)[j]) + 1
                if value not in solution:
                    solution[i + 1] = value
                    break
        return solution


    def calculate_distance(input_solution):
        distance = 0
        soution = input_solution
        for i in range(len(input_solution) - 1):
            distance = distance + dataset[(soution[i] - 1), (soution[i + 1] - 1)]
        return distance


    for i in range(dataset.shape[0]):
        node = solution[i] - 1
        solution = generate_solution(i, node, solution)

    dis_greedy = calculate_distance(solution)

    #print("Solution from Greedy", solution)
    #print("distance from greedy", dis_greedy)

    # Carousel Start here ##

    drop = int(round(len(solution) * beta))
    solution[-drop:] = 0

    for i in range(alpha * len(solution)):
        solution = np.roll(solution, -1)
        solution[-1] = 0
        node = (solution[-(drop + 2)]) - 1
        pos = len(solution) - (drop + 2)
        solution = generate_solution(pos, node, solution)

    for i in range((len(solution) - (drop + 1)), dataset.shape[0]):
        node = solution[i] - 1
        solution = generate_solution(i, node, solution)

    total_distance = calculate_distance(solution)
    cbest_obj = total_distance
    cbest_solution = solution
    if cbest_obj < gbest_obj:
        gbest_obj = cbest_obj
        gbest_solution = cbest_solution
        history.append(gbest_obj)

obj = gbest_obj
end_time = time.process_time()
print("Final Solution (Carousel) of given TSP instance = ", gbest_solution)
print("Total distance from carousel", obj)
#print("Difference between carousel and greedy distance", (obj-dis_greedy))
print("Total calculation time:", end_time - start_time,'seconds')
plt.figure(figsize=(10, 8))
plt.plot(history, linewidth = 4)
plt.show()