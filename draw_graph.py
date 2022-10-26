import os
import json
import pandas as pd
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
from datetime import datetime

# pip install networkx==2.6.3 # because other versions of networkx cause issues

# Assignment todos
# TODO: 
# Question 1: Network statistics - Metrics
# should be done
# Question 2: Communities
# TODO
# Question 3: HITS / PageRank
# done
# Question 4: Longitudinal analysis
# done
# Question 5: Visualisation
# TODO adapt graph visualisation according to metrics
# Question 6: Discussion and conclusions

# provide file name manually in file
file_name = 'office.json'

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
    '''
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
    '''

    df_full = df

    return df_full
    

def build_graph(df, graph_name):
    print("building graph {}...".format(graph_name))
    unique_reviewers = df.reviewerID.unique().tolist()
    unique_products = df.asin.unique().tolist()
    unique_product_idx = range(len(unique_products))
    reviewer_by_product = np.zeros([len(unique_reviewers), len(unique_products)])
    product_idx_to_id = dict(zip(unique_product_idx, unique_products)) # converts number to identifiable ID from json file
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
    min_unix_per_month = [current_date + i*thirty_days_in_unix for i in range(number_of_months)]

    for i, row in df_top_products.iterrows():
        current_item = product_id_to_idx[row.loc['asin']]
        idx_for_product = max_val_idx.index(current_item)
        for j, element in enumerate(min_unix_per_month):
            if int(row.loc['unixReviewTime']) < element:
                monthly_reviews_for_each_top_product[idx_for_product][j-1] += 1
                break

    x_axis = [i+1 for i in range(12)]
    plt.plot(x_axis, monthly_reviews_for_each_top_product[0], label = 'product 1')
    plt.plot(x_axis, monthly_reviews_for_each_top_product[1], label = 'product 2')
    plt.plot(x_axis, monthly_reviews_for_each_top_product[2], label = 'product 3')
    plt.xlabel('Months')
    plt.ylabel('Number of reviews')
    plt.legend()
    plt.title('Number of reviews for most-reviewed products between '\
         + '\n' + datetime.utcfromtimestamp(earliest_date_for_all).strftime('%Y-%m-%d') + ' and ' + \
         datetime.utcfromtimestamp(min_unix_per_month[-1]+thirty_days_in_unix).strftime('%Y-%m-%d')
         )
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, 'longitudinal_analysis')
    plt.savefig(final_directory, dpi=600)
    plt.close()

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

def draw_graph(graph, 
               graph_name, 
               degree_list,
               betweenness_centrality,
               eigenvector_centrality,
               closeness_centrality,
               clustering_coefficient,
               connected_components,
               authorities,
               color = 'm',
               display_graph = False):
               
    ##########################################
    #### Draw full unadjusted graph ##########
    ##########################################        
    
    if graph_name:
        print("Drawing graph {}".format(graph_name))
    else:
        sys.exit("Please supply graph name for saving fig.")

    # change k for different distances between nodes
    layout = nx.spring_layout(graph, k = 0.15)

    nx.draw(graph, layout, node_size = 5, node_color = color)

    # Draw edge labels using layout
    nx.draw_networkx_edges(graph, pos=layout)

    if not display_graph:
        current_directory = os.getcwd()
        final_directory = os.path.join(current_directory, graph_name + "_normal")
        plt.savefig(final_directory, dpi=600)
        plt.close()
    else:
        plt.show()

    ############ Example visualisation #########
    #############################################
    #### Change node color based on authority ###
    #############################################

    largest_node_size = 25
    max_value_auth = max(list(authorities.keys()))
    scaling_factor = largest_node_size / max_value_auth

    node_sizes = []
    for val in authorities.keys():
        size = val * scaling_factor
        node_sizes.append(size)

    node_colours = []
    for val in node_sizes:
        if val > 20:
            node_colours.append('r')
        elif val > 10:
            node_colours.append('b')
        elif val > 1:
            node_colours.append('y')
        else:
            node_colours.append('lightgrey')
            
    # I ended up coloring the nodes differently because enlarging them doesn't work too well # 
    nx.draw(graph, layout, node_size = 5, node_color = node_colours)

    # Draw edge labels using layout
    nx.draw_networkx_edges(graph, pos=layout)

    if not display_graph:
        current_directory = os.getcwd()
        final_directory = os.path.join(current_directory, graph_name + "_authorities")
        plt.savefig(final_directory, dpi=600)
        plt.close()
    else:
        plt.show()

def degree_centrality_histogram(graph, graph_name, bins = 20, display_hist = False):
    deg_central= list(dict(graph.degree).values())
    plt.hist(deg_central, bins=bins)
    if not display_hist:
        current_directory = os.getcwd()
        graph_name = graph_name + '_degree_centrality_histogram'
        final_directory = os.path.join(current_directory, graph_name)
        plt.savefig(final_directory, dpi=600)
        plt.close()
    else:
        plt.show()

def hits_algorithm(graph):
    h, a = nx.hits(graph)
    # sort dicts
    h = {k: v for k, v in sorted(h.items(), key=lambda item: item[1], reverse=True)}
    #a = {k: v for k, v in sorted(a.items(), key=lambda item: item[1], reverse=True)}
    h_first10 = {k: h[k] for k in list(h)[:10]}
    #a_first10 = {k: a[k] for k in list(a)[:10]}
    print("======= first 10 hubs =======", h_first10)
    #print("======= authorities =======", a_first10)
    # a is the same as h, but unsorted. we need it unsorted
    return a

def network_statistics(graph, graph_name):
    print("======= Graph statistics for graph {} =======".format(graph_name))
    print(graph)
    degrees = dict(graph.degree)
    degree_list = list(degrees.values())
    print("Average degree centrality: ", sum(degree_list) / len(degree_list))
    betweenness_centrality = dict(nx.betweenness_centrality(graph))
    betweenness_centrality_list = list(betweenness_centrality.values())
    print("Average betweenness centrality: ", sum(betweenness_centrality_list) / len(betweenness_centrality_list))
    eigenvector_centrality = dict(nx.eigenvector_centrality(graph))
    eigenvector_centrality_list = list(eigenvector_centrality.values())
    print("Average eigenvector centrality: ", sum(eigenvector_centrality_list) / len(eigenvector_centrality_list))
    closeness_centrality = dict(nx.closeness_centrality(graph))
    closeness_centrality_list = list(closeness_centrality.values())
    print("Average closeness_centrality: ", sum(closeness_centrality_list) / len(closeness_centrality_list))
    clustering_coefficient = nx.clustering(graph)
    clustering_coefficient_list = list(clustering_coefficient.values())
    print("Average clustering coefficient: ", sum(clustering_coefficient_list) / len(clustering_coefficient_list))
    try:
        print("Graph diameter: ", nx.diameter(graph))
    except:
        print("Graph diameter: infinity") # because graph is not fully connected
    connected_components = nx.number_connected_components(graph)
    print("Number of connected components: ", connected_components)
    connected_components_size = [len(c) for c in sorted(nx.connected_components(graph), key=len, reverse=True)]
    print("Sizes of connected comonents: ", connected_components_size)

    return degrees, betweenness_centrality, \
           eigenvector_centrality, closeness_centrality, \
           clustering_coefficient, connected_components, connected_components_size
    
def main(*args):
    df_full = read_data(file_name)
    full_name = "Full"
    G_full = build_graph(df_full, full_name)
    network_stats = network_statistics(G_full, full_name)
    authorities = hits_algorithm(G_full)

    degree_list, betweenness_centrality, \
    eigenvector_centrality, closeness_centrality, \
    clustering_coefficient, connected_components, connected_components_size = network_stats
    
    degree_centrality_histogram(G_full, full_name)
    draw_graph(graph = G_full, 
               graph_name = full_name,
               degree_list = degree_list,
               betweenness_centrality = betweenness_centrality,
               eigenvector_centrality = eigenvector_centrality,
               closeness_centrality = closeness_centrality,
               clustering_coefficient = clustering_coefficient,
               connected_components = connected_components,
               authorities = authorities, 
               color = 'b',
               )

if __name__ == "__main__":
    main(file_name)