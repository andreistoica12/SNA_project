import copy
import os
import json
import pandas as pd
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
from datetime import datetime
import random

# pip install networkx==2.6.3 # because other versions of networkx cause issues
# provide file name manually in file
file_name = 'Musical_Instruments_5.json'


def read_data(file_name):
    print("reading in data...")
    data = []
    with open(file_name) as f:
        for l in f:
            data.append(json.loads(l.strip()))

    # convert list into pandas dataframe
    df = pd.DataFrame.from_dict(data)
    print("Number of total reviews: ", len(df))

    # 1. convert reviewtime to rank-orderable number (by month)
    # MMDYYYY - > MMDDYYYY
    t1 = time.time()

    for i, row in df.iterrows():
        split_str = row.loc['reviewTime'].split()
        daystring = split_str[1].replace(',', "") if len(split_str[1]) == 3 else "0" + split_str[1].replace(',',
                                                                                                            "")  # MMDYYYY - > MMDDYYYY
        new_format = split_str[0] + daystring + split_str[2]
        df.at[i, 'reviewTime'] = new_format

    t2 = time.time()
    print("reformatting dates took {:.2f}s".format(t2 - t1))

    df_full = df

    return df_full


def build_graph(df, graph_name):
    print("building graph {}...".format(graph_name))
    unique_reviewers = df.reviewerID.unique().tolist()
    unique_products = df.asin.unique().tolist()
    unique_product_idx = range(len(unique_products))
    reviewer_by_product = np.zeros([len(unique_reviewers), len(unique_products)])
    product_idx_to_id = dict(
        zip(unique_product_idx, unique_products))  # converts number to identifiable ID from json file
    product_id_to_idx = dict(zip(unique_products, unique_product_idx))
    for i, row in df.iterrows():
        current_item = row.loc['asin']
        current_reviewer = row.loc['reviewerID']
        reviewer_by_product[unique_reviewers.index(current_reviewer), unique_products.index(current_item)] = 1

    # Find top n products (products with the most co-purchases)
    product_by_reviewer = np.transpose(reviewer_by_product)
    top_n = 3
    max_vals = [0] * top_n
    max_val_idx = [0] * top_n
    for i, product in enumerate(product_by_reviewer):
        product_importance = np.sum(product)
        for j in range(top_n):
            if product_importance > max_vals[j]:
                max_vals[j] = product_importance
                max_val_idx[j] = i
                break

    # find first date for which all products were reviewed
    earliest_dates = [20000000000] * top_n
    max_date = 0
    for i, row in df.iterrows():
        current_idx = product_id_to_idx[row.loc['asin']]
        if current_idx in max_val_idx:
            if int(row.loc['unixReviewTime']) > max_date:
                max_date = int(row.loc['unixReviewTime'])
            if int(row.loc['unixReviewTime']) < earliest_dates[max_val_idx.index(current_idx)]:
                earliest_dates[max_val_idx.index(current_idx)] = int(row.loc['unixReviewTime'])

    earliest_date_for_all = max(earliest_dates)

    df_top_products = []

    time_frame = int(max_date) - int(earliest_date_for_all)
    thirty_days_in_unix = 2592000
    if time_frame % thirty_days_in_unix != 0:
        max_date = max_date - (time_frame % thirty_days_in_unix)
        time_frame = int(max_date) - int(earliest_date_for_all)

    for i, row in df.iterrows():
        current_item = product_id_to_idx[row.loc['asin']]
        current_date = int(row.loc['unixReviewTime'])
        if current_item in max_val_idx:
            if earliest_date_for_all <= current_date <= max_date:
                df_top_products.append(row)
    df_top_products = pd.DataFrame(df_top_products)

    # Only do analysis for first 12 months
    number_of_months = 12
    monthly_reviews_for_each_top_product = [[0] * number_of_months for i in range(top_n)]
    current_date = int(earliest_date_for_all)
    min_unix_per_month = [current_date + i * thirty_days_in_unix for i in range(number_of_months)]

    for i, row in df_top_products.iterrows():
        current_item = product_id_to_idx[row.loc['asin']]
        idx_for_product = max_val_idx.index(current_item)
        for j, element in enumerate(min_unix_per_month):
            if int(row.loc['unixReviewTime']) < element:
                monthly_reviews_for_each_top_product[idx_for_product][j - 1] += 1
                break

    x_axis = [i + 1 for i in range(12)]
    plt.plot(x_axis, monthly_reviews_for_each_top_product[0], label='product 1')
    plt.plot(x_axis, monthly_reviews_for_each_top_product[1], label='product 2')
    plt.plot(x_axis, monthly_reviews_for_each_top_product[2], label='product 3')
    plt.xlabel('Months')
    plt.ylabel('Number of reviews')
    plt.legend()
    plt.title('Number of reviews for most-reviewed products between ' \
              + '\n' + datetime.utcfromtimestamp(earliest_date_for_all).strftime('%Y-%m-%d') + ' and ' + \
              datetime.utcfromtimestamp(min_unix_per_month[-1] + thirty_days_in_unix).strftime('%Y-%m-%d')
              )
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, 'longitudinal_analysis')
    plt.savefig(final_directory, dpi=600)
    plt.close()
    # plt.show()

    co_purchase_matrix = np.zeros([len(unique_products), len(unique_products)])
    for row in reviewer_by_product:
        indices = np.where(row == 1)[0]
        for val in indices:
            for val_ in indices:
                if val != val_:
                    co_purchase_matrix[val, val_] += 1

    # threshold singular links
    co_purchase_matrix[co_purchase_matrix == 1] = 0

    G_co_purchase = nx.from_numpy_array(co_purchase_matrix)
    remove = [node for node, degree in dict(G_co_purchase.degree()).items() if degree < 1]
    G_co_purchase.remove_nodes_from(remove)
    return G_co_purchase


def network_statistics(graph, graph_name):
    print("======= Graph statistics for graph {} =======".format(graph_name))
    print(graph)

    degrees = dict(graph.degree)
    degree_list = list(degrees.values())
    print("Average degree centrality: ", sum(degree_list) / len(degree_list), ". Highest value: ", max(degree_list))

    eigenvector_centrality = dict(nx.eigenvector_centrality(graph))
    eigenvector_centrality_list = list(eigenvector_centrality.values())
    print("Average eigenvector centrality: ", sum(eigenvector_centrality_list) / len(eigenvector_centrality_list), ". Highest value: ", max(eigenvector_centrality_list))

    closeness_centrality = dict(nx.closeness_centrality(graph))
    closeness_centrality_list = list(closeness_centrality.values())
    print("Average closeness_centrality: ", sum(closeness_centrality_list) / len(closeness_centrality_list), ". Highest value: ", max(closeness_centrality_list))

    betweenness_centrality = dict(nx.betweenness_centrality(graph))
    betweenness_centrality_list = list(betweenness_centrality.values())
    print("Average betweenness centrality: ", sum(betweenness_centrality_list) / len(betweenness_centrality_list), ". Highest value: ", max(betweenness_centrality_list))

    clustering_coefficient = nx.clustering(graph)
    clustering_coefficient_list = list(clustering_coefficient.values())
    print("Average clustering coefficient: ", sum(clustering_coefficient_list) / len(clustering_coefficient_list))

    try:
        print("Graph diameter: ", nx.diameter(graph))
    except:
        print("Graph diameter: infinity")  # because graph is not fully connected

    density = nx.density(graph)
    print("Graph density: ", density)

    connected_components = nx.number_connected_components(graph)
    print("Number of connected components: ", connected_components)

    connected_components_size = [len(c) for c in sorted(nx.connected_components(graph), key=len, reverse=True)]
    print("Sizes of connected comonents: ", connected_components_size)


    return degrees, eigenvector_centrality, \
           closeness_centrality, betweenness_centrality, \
           clustering_coefficient, connected_components, connected_components_size


############ NetworkX 2.6.3 doesn't have the Girvan-Newman algorithm implemented #########
############ This is the Girwan-Newman algorithm from NetworkX 2.8.8 #####################
############ slightly modified to fit our visualization purposes #########################
##########################################################################################
def girvan_newman(G, number_of_iterations, most_valuable_edge=None):
    """Finds communities in a graph using the Girvan???Newman method.

    Parameters
    ----------
    G : NetworkX graph

    most_valuable_edge : function
        Function that takes a graph as input and outputs an edge. The
        edge returned by this function will be recomputed and removed at
        each iteration of the algorithm.

        If not specified, the edge with the highest
        :func:`networkx.edge_betweenness_centrality` will be used.

    Returns
    -------
    iterator
        Iterator over tuples of sets of nodes in `G`. Each set of node
        is a community, each tuple is a sequence of communities at a
        particular level of the algorithm.

    Notes
    -----
    The Girvan???Newman algorithm detects communities by progressively
    removing edges from the original graph. The algorithm removes the
    "most valuable" edge, traditionally the edge with the highest
    betweenness centrality, at each step. As the graph breaks down into
    pieces, the tightly-knit community structure is exposed and the
    result can be depicted as a dendrogram.

    """
    # If the graph is already empty, simply return its connected
    # components.
    if G.number_of_edges() == 0:
        yield tuple(nx.connected_components(G))
        return
    # If no function is provided for computing the most valuable edge,
    # use the edge betweenness centrality.
    if most_valuable_edge is None:
        def most_valuable_edge(G):
            """Returns the edge with the highest betweenness centrality
            in the graph `G`.

            """
            # We have guaranteed that the graph is non-empty, so this
            # dictionary will never be empty.
            betweenness = nx.edge_betweenness_centrality(G)
            return max(betweenness, key=betweenness.get)

    # The copy of G here must include the edge weight data.
    g = G.copy().to_undirected()
    # Self-loops must be removed because their removal has no effect on
    # the connected components of the graph.
    g.remove_edges_from(nx.selfloop_edges(g))

    iteration_step = 1

    # we force the algorithm to stop after 5 iterations, because we can only make sense
    # of a few communities that we include in a visualization
    while g.number_of_edges() > 0 and iteration_step <= number_of_iterations:
        print("Girvan-Newman algorithm - computing iteration ", iteration_step, "...")
        iteration_step += 1
        yield _without_most_central_edges(g, most_valuable_edge)


def _without_most_central_edges(G, most_valuable_edge):
    """Returns the connected components of the graph that results from
    repeatedly removing the most "valuable" edge in the graph.
    *** On top of that, after some changes, it now also returns
    *** a copy of the updated graph, in order for us to visualize
    *** the graph for each iteration

    `G` must be a non-empty graph. This function modifies the graph `G`
    in-place; that is, it removes edges on the graph `G`.

    `most_valuable_edge` is a function that takes the graph `G` as input
    (or a subgraph with one or more edges of `G` removed) and returns an
    edge. That edge will be removed and this process will be repeated
    until the number of connected components in the graph increases.

    """
    original_num_components = nx.number_connected_components(G)
    num_new_components = original_num_components
    while num_new_components <= original_num_components:
        edge = most_valuable_edge(G)
        G.remove_edge(*edge)
        new_components = tuple(nx.connected_components(G))
        num_new_components = len(new_components)

    G_copy = copy.deepcopy(G)
    output = {"new_components": new_components, "updated_graph": G_copy}

    return output


def communities(graph, graph_name, GN_number_of_iterations):
    print("======= Communities for graph {} =======".format(graph_name))
    print(graph)
    print("computing cliques...")
    cliques = nx.find_cliques(graph)
    print("computing homophily score...")
    # homophily_score = nx.degree_assortativity_coefficient(graph)
    # As per the documentation, the function below yields the same result as the one commented out above,
    # but it uses a potentially faster function (scipy.stats.pearsonr)
    homophily_score = nx.degree_pearson_correlation_coefficient(graph)
    print("computing bridges...")
    bridges = nx.bridges(graph)
    print("computing Girvan-Newman groups...")
    GN_output = girvan_newman(graph, GN_number_of_iterations)

    return cliques, homophily_score, bridges, GN_output


def hits_algorithm(graph):
    h, a = nx.hits(graph)
    h = {k: v for k, v in sorted(h.items(), key=lambda item: item[1], reverse=True)}
    a = {k: v for k, v in sorted(a.items(), key=lambda item: item[1], reverse=True)}
    h_first10 = {k: h[k] for k in list(h)[:10]}
    a_first10 = {k: a[k] for k in list(a)[:10]}
    print("======= first 10 hubs =======", h_first10)
    print("======= first 10 authorities =======", a_first10)
    return a


def node_sizes_for_plotting(graph, largest_node_size=200):
    """
    Scales node sizes according to their degree

    Returns
    -------
    list
        node_sizes
    """
    max_value_degree = max(dict(graph.degree).values())
    scaling_factor = largest_node_size / max_value_degree

    node_sizes = []
    # node[0]: node, node[1]: degree
    for node in graph.degree:
        size = node[1] * scaling_factor
        node_sizes.append(size)

    return node_sizes


def node_colors_for_plotting(attribute):
    """
    Takes an attribute and colors the nodes according
    to where they fall on that attribute in terms of percentiles
    colors plotted from cold = low to warm = high

    Returns
    -------
    list
        node_colors
    """
    color_palette = ['b', 'g', 'y', 'r']

    low = np.percentile(attribute, 25)
    middle = np.percentile(attribute, 50)
    high = np.percentile(attribute, 75)

    node_colors = []

    for val in attribute:
        if val < low:
            node_colors.append(color_palette[0])
        elif low <= val < middle:
            node_colors.append(color_palette[1])
        elif middle <= val < high:
            node_colors.append(color_palette[2])
        else:
            node_colors.append(color_palette[3])

    return node_colors


def draw_graph(graph,
               graph_name,
               degree_list,
               betweenness_centrality,
               eigenvector_centrality,
               closeness_centrality,
               clustering_coefficient,
               connected_components,
               authorities,
               color='m',
               display_graph=False):
    ##########################################
    #### Draw full unadjusted graph ##########
    ##########################################

    if graph_name:
        print("Drawing graph {}".format(graph_name))
    else:
        sys.exit("Please supply graph name for saving fig.")

    # Adjust node sizes based on degree
    node_sizes = node_sizes_for_plotting(graph, largest_node_size=200)

    # change k for different distances between nodes
    layout = nx.spring_layout(graph, k=0.3)
    nx.draw(graph, layout, node_size=node_sizes, node_color=color)
    # Draw edge labels using layout
    nx.draw_networkx_edges(graph, pos=layout)
    plt.show()

    ############ Example visualisation #########
    #############################################
    #### Change node color based on authority ###
    #############################################

    auth_values = np.array(list(authorities.values()))
    node_colors = node_colors_for_plotting(auth_values)
    nx.draw(graph, layout, node_size=node_sizes, node_color=node_colors)
    # Draw edge labels using layout
    nx.draw_networkx_edges(graph, pos=layout)
    plt.show()


def degree_centrality_histogram(graph, graph_name, bins=20, display_hist=False):
    deg_central = list(dict(graph.degree).values())
    plt.hist(deg_central, bins=bins)

    plt.show()


# draw Maximum Clique(s) = the largest maximal clique(s)
def draw_maximum_cliques(graph,
                         graph_name,
                         cliques,
                         display_graph=False):
    if graph_name:
        print("Drawing the Maximum Clique(s) for graph {name}".format(name=graph_name))
    else:
        sys.exit("Please supply graph name for saving fig.")

    options = {"edgecolors": "tab:gray", "node_size": 120, "alpha": 0.9}

    # change k for different distances between nodes
    layout = nx.spring_layout(graph, k=0.4)

    # We store the cliques (lists of nodes) corresponding to the maximal number of nodes that can appear in a clique.
    # We will sort the keys in the cliques dictionary, so the last key is the maximal size of a clique.
    last_key = list(cliques.keys())[-1]
    maximum_cliques = cliques[last_key]

    # we create a list of randomly-generated colors.
    colors = []
    for _ in range(len(maximum_cliques)):
        colors.append(["#" + ''.join([random.choice('ABCDEF0123456789') for i in range(6)])])

    # we draw the nodes outside the cliques as smaller light grey points - they are less important for this visualization
    nx.draw(graph, layout, node_size=5, node_color='lightgrey')

    color_idx = 0
    # each node in a group has the same color, but nodes in different groups have different colors
    for maximum_clique in maximum_cliques:
        nx.draw_networkx_nodes(graph, layout, nodelist=list(maximum_clique), node_color=colors[color_idx], **options)
        color_idx += 1

    ### PROBABLY WE NEED TO SCALE THE SIZE OF THE NODES AS WELL, BUT I'LL DO THAT LATER

    # Draw edge labels using layout
    nx.draw_networkx_edges(graph, pos=layout, width=0.2)

    plt.show()


def draw_bridges(graph,
                 graph_name,
                 bridges,
                 display_graph=False):
    if graph_name:
        print("Drawing bridges for graph {name}".format(name=graph_name))
    else:
        sys.exit("Please supply graph name for saving fig.")

    # change k for different distances between nodes
    layout = nx.spring_layout(graph, k=0.15)

    nx.draw(graph, layout, node_size=10, node_color='m')

    nx.draw_networkx_edges(
        graph,
        layout,
        edgelist=bridges,
        width=8,
        alpha=0.5,
        edge_color="tab:blue",
    )

    plt.show()


def draw_GN_groups(graph,
                   graph_name,
                   iteration,
                   groups,
                   display_graph=False):
    ########################################################################
    #### Draw GN groups corresponding to one iteration of the algorithm ####
    ########################################################################

    if graph_name:
        print("Drawing Girvan-Newman communities for graph {name} at iteration {iteration}".format(name=graph_name,
                                                                                                   iteration=iteration))
    else:
        sys.exit("Please supply graph name for saving fig.")

    options = {"edgecolors": "tab:gray", "node_size": 40, "alpha": 0.7}

    # change k for different distances between nodes
    layout = nx.spring_layout(graph, k=0.15)

    # we create a list of randomly-generated colors.
    colors = []
    for _ in range(len(groups)):
        colors.append(["#" + ''.join([random.choice('ABCDEF0123456789') for i in range(6)])])

    color_idx = 0
    # each node in a group has the same color, but nodes in different groups have different colors
    for group in groups:
        nx.draw_networkx_nodes(graph, layout, nodelist=list(group), node_color=colors[color_idx], **options)
        color_idx += 1

    ### PROBABLY WE NEED TO SCALE THE SIZE OF THE NODES AS WELL, BUT I'LL DO THAT LATER

    # Draw edge labels using layout
    nx.draw_networkx_edges(graph, pos=layout)

    plt.show()


def main(*args):
    df_full = read_data(file_name)
    full_name = "Full"
    ##################### 4. LONGITUDINAL ANALYSIS & BUILD GRAPH #####################
    G_full = build_graph(df_full, full_name)

    ##################### 1. NETWORK STATISTICS - METRICS #####################
    network_stats = network_statistics(G_full, full_name)

    global degree_dict, eigenvector_centrality, \
    closeness_centrality, betweenness_centrality, \
    clustering_coefficient, connected_components, connected_components_size

    degree_dict, eigenvector_centrality, \
    closeness_centrality, betweenness_centrality, \
    clustering_coefficient, connected_components, connected_components_size = network_stats

    ##################### 3. HITS #####################
    authorities = hits_algorithm(G_full)

    ##################### 2. COMMUNITIES #####################

    # the number of iterations that the Girvan-Newman algorithm executes
    GN_number_of_iterations = 3

    cliques, homophily_score, bridges, GN_output = communities(G_full, full_name, GN_number_of_iterations)

    ### If you wish to see the raw data, run the file in the Python Console (the global variables will be visible)

    # transform the Generator object returned by the find_cliques() method into a dictionary with:
    # key = number of elements in a clique (the size of a clique)
    # value = list of cliques (of size indicated by the key)
    global cliques_dict
    cliques_dict = {}
    for clique in cliques:
        if cliques_dict.get(len(clique)) is None:
            cliques_dict[len(clique)] = [clique]
        cliques_dict[len(clique)].append(clique)
    cliques_dict = dict(sorted(cliques_dict.items()))

    # this is the degree assortativity of the graph -> it measures the similarity of connections in the graph
    # with respect to the node degree
    global homophily_score_global
    homophily_score_global = homophily_score

    # transform the Generator object returned by the bridges() method into a list of tuples (a, b), where:
    # a and b are the 2 endpoint nodes of a bridge
    global bridges_list
    bridges_list = [bridge for bridge in bridges]

    # transform the Generator object returned by the girvan_newman() function into a dictionary, where:
    # key = type of datastructure that we want to store, i.e. the groups/communities/components and the graphs themselves
    # value = another dictionary, where:
    # key = number of the iteration (because we want to store the groups and the graphs corresponding to each iteration.
    #       This eases the visualization process afterwards)
    # value = list of components/groups/communities
    global GN_output_dict
    GN_output_dict = {"GN_groups_dict": {}, "GN_graphs_dict": {}}
    iteration_step = 1
    for element in GN_output:
        GN_output_dict["GN_groups_dict"][iteration_step] = element["new_components"]
        GN_output_dict["GN_graphs_dict"][iteration_step] = element["updated_graph"]
        iteration_step += 1

    ##################### 5. VISUALIZATION #####################

    draw_graph(graph=G_full,
               graph_name=full_name,
               degree_list=degree_dict,
               betweenness_centrality=betweenness_centrality,
               eigenvector_centrality=eigenvector_centrality,
               closeness_centrality=closeness_centrality,
               clustering_coefficient=clustering_coefficient,
               connected_components=connected_components,
               authorities=authorities,
               color='b',
               )

    degree_centrality_histogram(G_full, full_name)

    draw_maximum_cliques(graph=G_full,
                         graph_name=full_name,
                         cliques=cliques_dict)

    draw_bridges(graph=G_full,
                 graph_name=full_name,
                 bridges=bridges_list)

    # As we described previously, we want to visualize the communities of the graph created at each iteration step.
    # This way, we are able to see the evolution of the algorithm, i.e. how it creates more connected components with each iteration
    iteration = 3
    draw_GN_groups(graph=GN_output_dict["GN_graphs_dict"][iteration],
                   graph_name=full_name,
                   iteration=iteration,
                   groups=GN_output_dict["GN_groups_dict"][iteration])


if __name__ == "__main__":
    main(file_name)
