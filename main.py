from typing import List, Set, Dict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx
import copy


def LTM_iteration_t(graph: networkx.Graph, patients_t_minus_one: Set) -> Set:
    new_infected = set()
    suspected = set(graph.nodes) - patients_t_minus_one

    # add new infected
    for person in suspected:
        if len(graph.adj[person]) > 0:
            sum_of_weights = 0
            for nbr, eattr in graph.adj[person].items():
                if nbr in patients_t_minus_one:
                    sum_of_weights += eattr['weight']
            if CONTAGION * sum_of_weights >= 1 + graph.nodes[person]['cv']:
                new_infected.add(person)

    # update concerns per time t
    people_to_update = suspected - new_infected - patients_t_minus_one
    for person in people_to_update:
        if len(graph.adj[person]) > 0:
            graph.nodes[person]['cv'] = len(list(patients_t_minus_one.intersection(set(graph.adj[person])))) / len(
                graph.adj[person])

    return new_infected


def LTM(graph: networkx.Graph, patients_0: List, iterations: int) -> Set:
    total_infected = set(patients_0)
    suspected = set(graph.nodes) - total_infected

    # initial zero concerns
    for person in suspected:
        graph.nodes[person]['cv'] = 0

    # do iterations
    for i in range(iterations):
        new_infected_t = LTM_iteration_t(graph, total_infected)
        total_infected = total_infected.union(new_infected_t)

    return total_infected


def ICM_iteration_t(graph: networkx.Graph, new_infected_t_minus_one: Set, new_deceased_t_minus_one: Set,
                    total_infected_t_minus_one: Set, total_deceased_t_minus_one: Set):
    new_infected = set()
    new_deceased = set()
    suspected = set(graph.nodes) - total_infected_t_minus_one - total_deceased_t_minus_one

    # infections
    for person in suspected:
        if len(graph.adj[person]) > 0:
            for nbr, eattr in graph.adj[person].items():
                if nbr in new_infected_t_minus_one:
                    if nbr not in new_deceased_t_minus_one:
                        if random_step(min(1, CONTAGION * eattr['weight'] * (1 - graph.nodes[person]['cv']))):
                            new_infected.add(person)
                            break

    # deaths
    for person in new_infected:
        if random_step(LETHALITY):
            new_deceased.add(person)

    new_infected = new_infected - new_deceased

    # update concerns per time t
    people_to_update = suspected - new_deceased - new_infected
    for person in people_to_update:
        if len(graph.adj[person]) > 0:
            c_v = (len(new_infected_t_minus_one.intersection(set(graph.adj[person]))) + 3 * len(
                new_deceased_t_minus_one.intersection(set(graph.adj[person])))) / len(graph.adj[person])
            graph.nodes[person]['cv'] = min(1, c_v)

    return new_infected, new_deceased


def ICM(graph: networkx.Graph, patients_0: List, iterations: int) -> [Set, Set]:
    total_infected = set(patients_0)
    total_deceased = set()
    total_suspected = set(graph.nodes) - total_infected

    # initial deaths
    for person in list(total_infected):
        if random_step(LETHALITY):
            total_deceased.add(person)

    # initial zero concerns
    for person in list(total_suspected):
        graph.nodes[person]['cv'] = 0

    total_infected = total_infected - total_deceased
    new_infected, new_deceased = copy.deepcopy(total_infected), copy.deepcopy(total_deceased)
    for i in range(iterations):
        new_infected, new_deceased = ICM_iteration_t(graph, new_infected, new_deceased, total_infected, total_deceased)
        total_infected = total_infected.union(new_infected)
        total_deceased = total_deceased.union(new_deceased)

    return total_infected, total_deceased


def random_step(p):
    """checks whether an infection or lethality occurred w/ probability p
        1 , infection or lethality occurred
        0 , o.w
    """
    rand_value = np.random.uniform()
    if rand_value <= p:
        return 1
    else:
        return 0


def plot_degree_histogram(histogram: Dict):
    plt.bar(histogram.keys(), histogram.values(), color='c')
    plt.ylim([0, max(histogram.values())])
    plt.xlim([-1, max(histogram.keys())])
    plt.ylabel("Number of nodes")
    plt.xlabel("Degree")
    plt.legend(["Part B-C"])
    plt.show()


def calc_degree_histogram(graph: networkx.Graph) -> Dict:
    """
    Example:
    if histogram[1] = 10 -> 10 nodes have only 1 friend
    """
    histogram = {}
    degrees = [deg[1] for deg in list(graph.degree)]
    for deg in degrees:
        if deg in histogram:
            histogram[deg] += 1
        else:
            histogram[deg] = 1
    return histogram


def build_graph(filename: str) -> networkx.Graph:
    G = networkx.Graph()
    edges = pd.read_csv(filename)
    columns = list(edges.columns)
    if 'w' in columns:
        for index, edge in edges.iterrows():
            G.add_edge(int(edge['from']), int(edge['to']), weight=float(edge['w']))
    else:
        for index, edge in edges.iterrows():
            G.add_edge(int(edge['from']), int(edge['to']))
    # TODO implement your code here
    return G


def clustering_coefficient(graph: networkx.Graph) -> float:
    triangle_count, open_triangle_count = 0, 0
    adj_list = graph.adj
    for node, neighbors in adj_list.items():
        for neighbor in neighbors:
            for second_neighbor in adj_list[neighbor]:
                if node != second_neighbor:
                    third_edge = node in adj_list[second_neighbor] or second_neighbor in adj_list[node]
                    triangle_count += int(third_edge)
                    open_triangle_count += int(not third_edge)
    return triangle_count / (triangle_count + open_triangle_count)


def compute_lethality_effect(graph: networkx.Graph, t: int) -> [Dict, Dict]:
    global LETHALITY
    mean_deaths = {}
    mean_infected = {}
    for l in (.05, .15, .3, .5, .7):
        LETHALITY = l
        mean_deaths[l] = 0
        mean_infected[l] = 0
        for iteration in range(30):
            G = copy.deepcopy(graph)
            patients_0 = np.random.choice(list(G.nodes), size=50, replace=False, p=None)
            model_iteration = ICM(G, patients_0, t)
            mean_deaths[l] += len(model_iteration[1])
            mean_infected[l] += len(model_iteration[0])
        mean_deaths[l] /= 30
        mean_infected[l] /= 30

    return mean_deaths, mean_infected


def plot_lethality_effect(mean_deaths: Dict, mean_infected: Dict):
    plt.plot(mean_deaths.keys(), mean_deaths.values(), color='blue')
    plt.plot(mean_infected.keys(), mean_infected.values(), color='red')
    plt.legend(["removed", "infected"])
    plt.xlabel("lethality value")
    plt.ylabel("mean number of removed/infected")
    plt.show()


def choose_who_to_vaccinate_example(graph: networkx.Graph) -> List:
    node2degree = dict(graph.degree)
    sorted_nodes = sorted(node2degree.items(), key=lambda item: item[1], reverse=True)[:50]
    people_to_vaccinate = [node[0] for node in sorted_nodes]
    return people_to_vaccinate


def choose_who_to_vaccinate(graph: networkx.Graph) -> List:
    node2degree = dict(graph.degree)
    sorted_nodes = sorted(node2degree.items(), key=lambda item: item[1], reverse=True)[:65]
    node2page = networkx.algorithms.link_analysis.pagerank(graph, max_iter=100, weight="weight")
    people_consider_to_vaccinate = {}
    for node in sorted_nodes:
        people_consider_to_vaccinate[node[0]] = node2page[node[0]]
    people_consider_to_vaccinate = sorted(people_consider_to_vaccinate.items(), key=lambda item: item[1], reverse=True)[
                                   :50]
    people_to_vaccinate = [node[0] for node in people_consider_to_vaccinate]
    return people_to_vaccinate


def vaccine(graph: networkx.Graph, people):
    for person in people:
        graph.remove_node(person)


"Global Hyper-parameters"
CONTAGION = .8
LETHALITY = .2

