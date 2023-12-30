#!/usr/bin/env python
# coding: utf-8
import networkx as nx
import pandas as pd
import numpy as np
import pickle
from node2vec import Node2Vec
import time


def lineage_years_in_academia(graph, nodeid):
    '''
    The function will calculate linage year
    by using researcher (vertex of interest) year 
    and last researcher joining the lineage 
    '''
    if graph.has_node(nodeid):
        descs = nx.descendants(graph, nodeid)
        node_year = graph.nodes[nodeid]['year']
        years =  sorted([graph.nodes[desc].get('year',0) for desc in descs if graph.nodes[desc].get('year',0) > 0 and len(str(graph.nodes[desc].get('year',0)))==4])
        if len(years) > 0:
            last_stud_year = years[-1]
        else :
            last_stud_year = node_year
        work_years = last_stud_year - node_year
    return (work_years, len(descs))


def lineage_years_in_academia_from_firsTstud(graph, nodeid):
    '''
    The function will calculate linage year
    by using researcher (vertex of interest), 
    first student joining the lineage and 
    last researcher joining the lineage (year)
    '''
    if graph.has_node(nodeid):
        descs = nx.descendants(graph, nodeid)
        years =  sorted([graph.nodes[desc].get('year',0) for desc in descs if graph.nodes[desc].get('year',0) > 0 and len(str(graph.nodes[desc].get('year',0)))==4])
        #print(years)
        if len(years) > 0:
            first_stud_year = years[0]
            last_stud_year = years[-1]
        else :
            first_stud_year = 0
            last_stud_year = 0
        work_years = last_stud_year - first_stud_year
    return (work_years, len(descs))


# def find_node_family(graph, nodeid, duration = 10):
#     '''
#     All the researcher joined the lineage 
#     of the researcher (vertex of interest)
#     within the '10' year duration from the 
#     first student year
#     '''
#     if graph.has_node(nodeid):
#         descs = nx.descendants(graph, nodeid)
#         node_year = graph.nodes[nodeid]['year']
#         qual_year = node_year + duration
#         years =  sorted([graph.nodes[desc].get('year',0) for desc in descs if graph.nodes[desc].get('year',0) > 0 and len(str(graph.nodes[desc].get('year',0)))== 4])
#         if len(years) > 0:
#             first_stud_year = years[0]
#         else :
#             first_stud_year = 0
#         qual_nodes =  [(desc, graph.nodes[desc]['year']) for desc in descs if graph.nodes[desc].get('year') and graph.nodes[desc]['year'] <= first_stud_year + duration]
#     else:
#         print(f"The node {nodeid} is not in the graph.")
#         return []
#     return qual_nodes


def find_family_graph(graph, nodeid, duration = 10):
    '''
    nodeid --> return (nodeid, family graph)
    '''
    bool_val = False
    family_edge_list = []
    sorted_family_nodes_years = []
    if graph.has_node(nodeid):
        descs = nx.descendants(graph, nodeid)
        node_year = graph.nodes[nodeid].get('year')
        #qual_year = node_year + duration
        years =  sorted([graph.nodes[desc]['year'] for desc in descs if graph.nodes[desc].get('year') and len(str(graph.nodes[desc]['year'])) == 6])
        if len(years) > 0:
            first_stud_year = years[0]
            nodes_years =  [(desc,graph.nodes[desc]['year']) for desc in descs if graph.nodes[desc].get('year') and graph.nodes[desc]['year'] <= first_stud_year + duration]
            family_nodes_years = [(nodeid, node_year)]+nodes_years
            family_node_list = [node_year[0] for node_year in family_nodes_years]
            sorted_family_nodes_years = sorted(family_nodes_years, key=lambda x: x[1])
            family_graph = graph.subgraph(family_node_list)
            bool_val = nx.is_weakly_connected(family_graph)
            family_edge_list = list(family_graph.edges)
    else:
        print(f"The node {nodeid} is not in the graph.")
        return (None, None, None, None)
    return (nodeid, bool_val, family_edge_list, sorted_family_nodes_years)


# def find_output_sequence(graph, nodeid, duration = 10, interval = 5):
#     '''
#     nodeid --> (output_sequence, time_sequence)
#     '''
#     output_values = []
#     time_period = []
#     if graph.has_node(nodeid):
#         descs = nx.descendants(graph, nodeid)
        
#         years =  sorted([graph.nodes[desc]['year'] for desc in descs if graph.nodes[desc].get('year') and len(str(graph.nodes[desc]['year']))== 6])
#         if len(years) > 0:
#             first_stud_year = years[0]
#             for period in range(duration, duration+interval+1):
#                 nodes =  [desc for desc in descs if graph.nodes[desc].get('year') and graph.nodes[desc]['year'] <= first_stud_year + period]
#                 family_years = [graph.nodes[desc]['year'] for desc in descs if graph.nodes[desc].get('year') and graph.nodes[desc]['year'] <= first_stud_year + period]
#                 output_values.append(len(nodes))
#                 time_period.append(family_years)
#         else:
#             nodes = []
#             years = []
#             output_values.append(len(nodes))
#             time_period.append(years)
#     else:
#         print(f"The node {nodeid} is not in the graph.")
#         return (None, None)
#     return (output_values, time_period)



def find_output_sequence(graph, nodeid, duration = 10, interval = 5):
    '''
    nodeid --> (output_sequence, time_sequence)
    '''
    output_values = []
    time_period = []
    if graph.has_node(nodeid):
        descs = nx.descendants(graph, nodeid)
        
        years =  sorted([graph.nodes[desc]['year'] for desc in descs if graph.nodes[desc].get('year') and len(str(graph.nodes[desc]['year']))== 6])
        if len(years) > 0:
            first_stud_year = years[0]
            for period in range(duration, duration+interval+1):
                nodes =  [desc for desc in descs if graph.nodes[desc].get('year') and graph.nodes[desc]['year'] <= first_stud_year + period]
                family_years = [graph.nodes[desc]['year'] for desc in descs if graph.nodes[desc].get('year') and graph.nodes[desc]['year'] <= first_stud_year + period]
                output_values.append(len(nodes))
                time_period.append(family_years)
        else:
            nodes = []
            years = []
            output_values.append(len(nodes))
            time_period.append(years)
    else:
        print(f"The node {nodeid} is not in the graph.")
        return (None, None)
    return (output_values, time_period)


def find_yearwise_completion(graph, nodeid, duration = 1, interval = 16):
    '''
    nodeid --> (output_sequence, time_sequence)
    '''
    output_values = []
    time_period = []
    nodes_seq = []
    if graph.has_node(nodeid):
        descs = nx.descendants(graph, nodeid)
        
        years =  sorted([graph.nodes[desc]['year'] for desc in descs if graph.nodes[desc].get('year') and len(str(graph.nodes[desc]['year']))== 6])
        if len(years) > 0:
            first_stud_year = years[0]
            for period in range(duration, duration+interval):
                nodes =  [desc for desc in descs if graph.nodes[desc].get('year') and graph.nodes[desc]['year'] <= first_stud_year + period]
                family_years = [graph.nodes[desc]['year'] for desc in descs if graph.nodes[desc].get('year') and graph.nodes[desc]['year'] <= first_stud_year + period]
                output_values.append(len(nodes))
                nodes_seq.append(nodes)
                time_period.append(family_years)
        else:
            nodes = []
            years = []
            output_values.append(len(nodes))
            time_period.append(years)
            nodes_seq.append(nodes)
    else:
        print(f"The node {nodeid} is not in the graph.")
        return (None, None)
    return (nodes_seq, output_values, time_period)


def convert_directed_family_graph_to_tree(edges_node_years):
    graph = nx.DiGraph()
    graph.add_edges_from(edges_node_years[0])
    node_year = dict(edges_node_years[1])
    nx.set_node_attributes(graph, values = node_year, name='year')
    for node in graph.nodes:
        #print(type(node))
        parents = list(graph.predecessors(node))
        if len(parents) > 1:
            parents_with_date = [(parent, graph.nodes[parent].get('year',0)) for parent in parents ]
            sorted_parents = sorted(parents_with_date, key=lambda x: x[1])
            remove_edges = [(p[0], node) for p in sorted_parents[:-1]]#newest realtion kept
            graph.remove_edges_from(remove_edges)
        else:
            continue
    return list(graph.edges)

def convert_directed_family_graph_to_tree_keep_old(edges_node_years):
    graph = nx.DiGraph()
    graph.add_edges_from(edges_node_years[0])
    node_year = dict(edges_node_years[1])
    nx.set_node_attributes(graph, values = node_year, name='year')
    for node in graph.nodes:
        #print(type(node))
        parents = list(graph.predecessors(node))
        if len(parents) > 1:
            parents_with_date = [(parent, graph.nodes[parent].get('year',0)) for parent in parents ]
            sorted_parents = sorted(parents_with_date, key=lambda x: x[1])
            remove_edges = [(p[0], node) for p in sorted_parents[1:]] #old realtion kept
            graph.remove_edges_from(remove_edges)
        else:
            continue
    return list(graph.edges)

def convert_directed_graph_to_tree(graph):
    for node in tqdm(graph.nodes):
        #print(type(node))
        parents = list(graph.predecessors(node))
        if len(parents) > 1:
            parents_with_date = [(parent, graph.nodes[parent].get('year',0)) for parent in parents ]
            sorted_parents = sorted(parents_with_date, key=lambda x: x[1])
            remove_edges = [(p[0], node) for p in sorted_parents[:-1]]
            graph.remove_edges_from(remove_edges)
        else:
            continue
    return graph


# ## MGP date completion functions


def fill_year(mgpedges, mgpid):
    '''
    In this function, we using advisee information
    of the researcher for completion of the year.
    i.e first_stud_year - 5 as researcher year, 
    if year is missing
    '''
    year = np.nan
    students = mgpedges[mgpedges['advisor']==mgpid].copy()
    if not students.empty:
        advisee_years = students['advisee_year']
        if not(advisee_years.isnull().sum() == advisee_years.shape[0]):
            min_advisee_year = min(advisee_years)
            return (min_advisee_year-5)
        else:
            return np.nan
    else:
        return np.nan



def fill_year1(mgpedges, iid):
    '''
    In this function, we are using own brother information 
    of the researcher for completion of the year. 
    i.e average value of all the brothers years 
    available missing
    '''
    advisor = mgpedges[mgpedges['advisee']==iid]['advisor'].copy()
    brothers = mgpedges[mgpedges['advisor'].isin(advisor.values)]
    if not brothers.empty:
        year = brothers['advisee_year']
        if not(year.isnull().sum() == year.shape[0]):
            avg_value = int(year.mean())
            return avg_value
        else:
            return np.nan
    else:
        return np.nan


def fill_year2(mgpedges, iid):
    '''
    In this function, we are using advisor information
    of the researcher for completion of the year.
    i.e advisor completion year + 5
    '''
    advisor=mgpedges[mgpedges['advisee']==iid]['advisor'].copy()
    advisor_as_advisee=mgpedges[mgpedges['advisee'].isin(advisor.values)].drop_duplicates(subset="advisee",
                                                                                          keep='first')
    if not advisor_as_advisee.empty:
        advisor_year=advisor_as_advisee['advisee_year']
        if not(advisor_year.isnull().sum() == advisor_year.shape[0]):
            #advisor_year=advisor_year.fillna(0)
            max_val=max(advisor_year)
            return max_val + 5
        else:
            return np.nan
    else:
        return np.nan


def fill_year3(mgpedges, mgpnodes, iid):
    '''
    In this function, we are using advisors information
    of the researcher for completion of the year.
    i.e advisor completion year + 5
    similar to function "fill_year2"
    '''
    advisor = mgpedges[mgpedges['advisee']==iid]['advisor'].copy()
    advisor_year = mgpnodes[mgpnodes['Id'].isin(advisor.values)]['Year'].copy()
    if not (advisor_year.empty) and not(advisor_year.isnull().sum() == advisor_year.shape[0]):
        #advisor_year=advisor_year.fillna(0)
        max_val = max(advisor_year)
        return max_val + 5
    else:
        return np.nan


def obtain_embedding(nodes):
    '''
    Input : Dataframe having edgelist, help in constructing directed family graph,
    Output : Dictionary with nodes as key and embeddings as value
    '''
    print("Process Started family graph embedding using node2vec")
    start_time = time.time()
    final_embed = {}
    if 'input_connected' in nodes.columns :
        graphs = nodes[nodes['input_connected']==True][['nodeid','input_edgelist']].values
        for family in graphs:
            nodeid = family[0]
            fam_g = eval(family[1])
            G1 = nx.DiGraph()
            G1.add_edges_from(fam_g)
            node_corpus = Node2Vec(G1, dimensions=100, walk_length=10, num_walks=5, workers=2, quiet=True)#seed
            model = node_corpus.fit(window=5, min_count=1) #gensim word2vec
            embed = {key: model.wv[key] for key in model.wv.index_to_key}
            final_embed[nodeid] = embed
    else:
        print("Connected graph info is not available")
    print(f"Process Completed in hours :{(time.time() - start_time)/3600}")
    return final_embed



def save_obj(obj, name):
    '''
    obj : Dictionary
    name : Filename to save
    '''
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    print("saved")


def load_obj(name ):
    '''
    name : Filename
    '''
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def pad_sequence(walks, num_seq=10, seq_length=10):
    '''
    Pad input walk to max walk length/sequence length and sequences to max number of sequences
    '''
    _= [walk.append("0") for walk in walks for _ in range(seq_length-len(walk))]
    walks_append = np.array([int(node) for walk in walks for node in walk])
    walks_append = walks_append.reshape(-1, seq_length)
    tmp = np.array([0]*seq_length*(num_seq-len(walks_append))).reshape(-1,seq_length)
    final_walks = np.append(walks_append, tmp, axis = 0)
    #sparse_walks = sparse.csr_matrix(final_walks)
    return final_walks


def get_paths(edge_list, root):
    '''
    return list of paths from root node to leaf nodes/list of list(path)
    '''
    G1 = nx.DiGraph()
    G1.add_edges_from(edge_list)
    leaf_node = [node for node in G1 if G1.out_degree(node)==0]
    paths = [nx.shortest_path(G1, root, node) for node in leaf_node]
    return paths


def find_walks(edge_list, walk_length=5, num_walks=5):
    '''
    input  : directed edge_list
    output : return list of walks(list)
    '''
    G1 = nx.DiGraph()
    G1.add_edges_from(edge_list)
    node_corpus = Node2Vec(G1, dimensions=100, walk_length=walk_length, num_walks=num_walks, workers=2, quiet=True, seed=1079)
    walks = node_corpus.walks
    walks.sort()
    new_walks = list(walk for walk,_ in itertools.groupby(walks))
    return new_walks

def mod_get_paths(edge_list, root):
    '''
    return list of paths from root node to every nodes (except root)/list of list(path)
    '''
    G1 = nx.DiGraph()
    G1.add_edges_from(edge_list)
    all_node = [node for node in G1 if node != root]
    paths = [nx.shortest_path(G1, root, node) for node in all_node]
    return paths


# def random_walk(start_node, walk_length):
#     walk = [start_node]  # starting node
    
#     for i in range(walk_length):
#         all_neighbours = [n for n in G.neighbors(start_node)]  # get all neighbours of the node
#         next_node = np.random.choice(all_neighbours, 1)[0]  # randomly pick 1 neighbour
#         walk.append(next_node)  # append this node to the walk
#         start_node = next_node  # this random node is now your current state
    
#     return walk

