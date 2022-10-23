import os
import json
import pandas as pd
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
# pip install networkx==2.6.3 # because other versions of networkx cause issues

# Assignment todos
# TODO: 
# Question 1: Network statistics - Metrics
# should be done
# Question 2: Communities
# TODO
# Question 3: HITS / PageRank
# TODO
# Question 4: Longitudinal analysis
# TODO
# Question 5: Visualisation
# TODO adapt graph visualisation according to metrics
# Question 6: Discussion and conclusions

# Code todos:
# 1. saving files for summer and christmas (problem: json not formatted properly)

# provide file name manually in file
file_name = 'office.json'

def remove_zero_rows_and_cols(numpyarray):
    numpyarray[~np.all(numpyarray == 0, axis=1)]
    numpyarray[:, ~np.all(numpyarray == 0, axis=0)]
    return numpyarray

def print_counts(numpyarray):
    unique, counts = np.unique(numpyarray, return_counts=True)
    for un, cn in zip(unique, counts):
        print("Value: {} Count: {}".format(un, cn))

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
        split_str = row[-1].split()
        daystring = split_str[1].replace(',',"") if len(split_str[1]) == 3 else "0" + split_str[1].replace(',',"") # MMDYYYY - > MMDDYYYY
        new_format = split_str[0] + daystring + split_str[2]
        df.at[i, 'reviewTime'] = new_format

    t2 = time.time()
    print("reformatting dates took {:.2f}s".format(t2-t1))

    # 2. Make different dataframes for different periods of time
    df_christmas = [] # Rest december, January, February
    df_summer = [] # July, August

    for i, row in df.iterrows():
        month_as_int = int(row[-1]) #last column is reviewtime
        row = row.rename(None).fillna(0) #some reviewers are NaN
        if (12251996 <= month_as_int <= 12312018) or (1011996 <= month_as_int <= 2282018): # whatever leap years
            df_christmas.append(row)
        if 6011996 <= month_as_int <= 8312018:
            df_summer.append(row)

    df_summer = pd.DataFrame(df_summer)
    df_christmas = pd.DataFrame(df_christmas)

    t3 = time.time()

    print("Created summer and christmas files in {:.2f}s".format(t3-t2))

    df_full = df

    return df_full, df_summer, df_christmas

def build_graph(df, graph_name):
    print("building graph {}...".format(graph_name))
    unique_reviewers = df.reviewerID.unique().tolist()
    unique_products = df.asin.unique().tolist()
    reviewer_by_product = np.zeros([len(unique_reviewers), len(unique_products)])

    for i, row in df.iterrows():
        current_item = row.loc['asin']
        current_reviewer = row.loc['reviewerID']
        reviewer_by_product[unique_reviewers.index(current_reviewer), unique_products.index(current_item)] = 1

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
    #print(G_co_purchase)
    #print("remove nodes with zero degree")
    remove = [node for node,degree in dict(G_co_purchase.degree()).items() if degree < 1]
    G_co_purchase.remove_nodes_from(remove)
    #print("final graph")
    #print(G_co_purchase)

    return G_co_purchase

def draw_graph(graph, graph_name, color = 'm', display_graph = False):
    if graph_name:
        print("Drawing graph {}".format(graph_name))
    else:
        sys.exit("Please supply graph name for saving fig.")

    # change k for different distances between nodes
    layout = nx.spring_layout(graph, k = 0.2)

    # use for node sizes proportional to degree
    # d = dict(G_co_purchase.degree)
    # v for v in d.values may need to be adjusted (i.e. divide v?)
    # nodelist=d.keys(), node_size=[v for v in d.values()]

    nx.draw(graph, layout, node_size = 5, node_color = color)

    # Draw edge labels using layout
    nx.draw_networkx_edges(graph, pos=layout)

    if not display_graph:
        current_directory = os.getcwd()
        final_directory = os.path.join(current_directory, graph_name)
        plt.savefig(final_directory, dpi=600)
        plt.close()
    else:
        plt.show()

def degree_centrality_histogram(graph, graph_name, bins = 20, display_hist = False):
    deg_central= list(dict(graph.degree).values())
    plt.hist(deg_central, bins=bins)
    if not display_hist:
        current_directory = os.getcwd()
        final_directory = os.path.join(current_directory, graph_name)
        plt.savefig(final_directory, dpi=600)
        plt.close()
    else:
        plt.show()

def network_statistics(graph, graph_name):
    print("======= Graph statistics for graph {} =======".format(graph_name))
    print(graph)
    degree_list = list(dict(graph.degree).values())
    print("Average degree centrality: ", sum(degree_list) / len(degree_list))
    betweenness_centrality = list(dict(nx.betweenness_centrality(graph)).values())
    print("Average betweenness centrality: ", sum(betweenness_centrality) / len(betweenness_centrality))
    eigenvector_centrality = list(dict(nx.eigenvector_centrality(graph)).values())
    print("Average eigenvector centrality: ", sum(eigenvector_centrality) / len(eigenvector_centrality))
    closeness_centrality = list(dict(nx.closeness_centrality(graph)).values())
    print("Average closeness_centrality: ", sum(closeness_centrality) / len(closeness_centrality))
    clustering_coefficient = list(dict(nx.clustering(graph)).values())
    print("Average clustering coefficient: ", sum(clustering_coefficient) / len(clustering_coefficient))
    try:
        print("Graph diameter: ", nx.diameter(graph))
    except:
        print("Graph diameter: infinity") # because graph is not fully connected
    print("Number of connected components: ", nx.number_connected_components(graph))
    connected_components_size = [len(c) for c in sorted(nx.connected_components(graph), key=len, reverse=True)]
    print("Sizes of connected comonents: ", connected_components_size)
    
def main(*args):
    df_full, df_summer, df_christmas = read_data(file_name)
    full_name = "Full"
    summer_name = "Summer"
    christmas_name = "Christmas"
    G_full = build_graph(df_full, full_name)
    G_summer = build_graph(df_summer, summer_name)
    G_christ = build_graph(df_christmas, christmas_name)
    network_statistics(G_full, full_name)
    network_statistics(G_summer, summer_name)
    network_statistics(G_christ, christmas_name)

    # graphs and histogram are saved to current directory 
    # if this is not desired, add
    # ...display_graph = True for draw_graph
    # ...display_hist = True for degree_centrality_histogram
    degree_centrality_histogram(G_full, full_name)
    degree_centrality_histogram(G_summer, summer_name)
    degree_centrality_histogram(G_christ, christmas_name)
    draw_graph(G_full, full_name, 'b')
    draw_graph(G_summer, summer_name, 'g')
    draw_graph(G_christ, christmas_name, 'm')
if __name__ == "__main__":
    main(file_name)