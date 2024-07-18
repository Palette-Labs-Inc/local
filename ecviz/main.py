import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="Eigenvector Centrality Visualization",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

rng = np.random.default_rng()
def get_weight():
    if weight_selection == "U[0,1)":
        return rng.uniform()
    elif weight_selection == "1":
        return 1

# graph initialization
def create_initial_clusters(n1, p1, n2, p2):
    seed = np.random.randint(1, 10000)
    G1 = nx.erdos_renyi_graph(n1, p1, seed=seed+1)
    G2 = nx.erdos_renyi_graph(n2, p2, seed=seed+2)

    # Assign random weights to edges
    for (u, v) in G1.edges():
        G1[u][v]['weight'] = get_weight()
    for (u, v) in G2.edges():
        G2[u][v]['weight'] = get_weight()

    G = nx.disjoint_union_all([G1, G2])
    return G, G1, G2

def update_weight(G, G1, G2, graph_id, u, v, new_weight):
    if graph_id == 1:
        if G1.has_edge(u, v):
            G1[u][v]['weight'] += new_weight
        else:
            G1.add_edge(u, v, weight=new_weight)
        # Update or create the edge in G
        if G.has_edge(u, v):
            G[u][v]['weight'] += new_weight
        else:
            G.add_edge(u, v, weight=new_weight)
    elif graph_id == 2:
        if G2.has_edge(u, v):
            G2[u][v]['weight'] += new_weight
        else:
            G2.add_edge(u, v, weight=new_weight)
        # Calculate the offset for nodes in G2
        offset = max(G1.nodes) + 1  # Assuming G1 nodes start from 0
        G_node_u = u + offset
        G_node_v = v + offset
        # Update or create the edge in G
        if G.has_edge(G_node_u, G_node_v):
            G[G_node_u][G_node_v]['weight'] += new_weight
        else:
            G.add_edge(G_node_u, G_node_v, weight=new_weight)
    elif graph_id == 3:
        # in this case, there is no need to update G1 or G2 b/c those are the separate
        # clusters, so no internal weights are being updated within those clusters
        if G.has_edge(u, v):
            G[u][v]['weight'] += new_weight
        else:
            G.add_edge(u, v, weight=new_weight)

def calculate_eigenvector_centrality(G):
    ec_kwargs = {
        'max_iter': 1000,
        'tol': 1e-2,
    }
    if use_weight_compute_ec:
        ec_kwargs['weight'] = 'weight'
    centrality = nx.eigenvector_centrality(
        G, 
        **ec_kwargs
    )
    return centrality

def calculate_node_weight_sums(G):
    weight_sums = {node: 0 for node in G.nodes()}
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 0.0)
        weight_sums[u] += weight
        weight_sums[v] += weight
    return weight_sums

def calculate_graph_value(G):
    ec = calculate_eigenvector_centrality(G)
    w = calculate_node_weight_sums(G)
    gv = {node: alpha_ec * ec[node] + alpha_w * w[node] for node in G.nodes()}
    return gv

def plot_graph_and_heatmaps(G):
    gv_1 = st.session_state.graph_value_cluster1
    gv_2 = st.session_state.graph_value_cluster2
    gv_sums = st.session_state.graph_value_sums
    centrality_data_1 = st.session_state.centrality_data_cluster1
    centrality_data_2 = st.session_state.centrality_data_cluster2
    centrality_sums = st.session_state.centrality_sums
    weight_sum_G1 = st.session_state.weight_sum_G1
    weight_sum_G2 = st.session_state.weight_sum_G2
    weight_sums = st.session_state.weight_sums
    step = st.session_state.step
    # pos = nx.spring_layout(G)  # dynamic position
    pos = st.session_state.pos
    
    fig, ax1 = plt.subplots(figsize=(12,2))

    # Plot the graph
    centrality = calculate_eigenvector_centrality(G)
    node_color = [centrality[node] for node in G.nodes()]
    nx.draw(G, pos, ax=ax1, with_labels=True, node_size=200, node_color=node_color, cmap=plt.cm.Blues, font_size=10, font_color='black', font_weight='bold', edge_color='gray')
    ax1.set_title(f"Graph at Step {step}")    
    st.pyplot(fig, use_container_width=True)

    # plot GV
    gv_df_1 = pd.DataFrame(gv_1).reset_index().melt(id_vars='index').rename(columns={'index': 'Step', 'variable': 'Node', 'value': 'GraphValue'})
    gv_df_2 = pd.DataFrame(gv_2).reset_index().melt(id_vars='index').rename(columns={'index': 'Step', 'variable': 'Node', 'value': 'GraphValue'})
    vmin = min(gv_df_1['GraphValue'].min(), gv_df_2['GraphValue'].min())
    vmax = max(gv_df_1['GraphValue'].max(), gv_df_2['GraphValue'].max())
    heatmap_gv_g1 = alt.Chart(gv_df_1).mark_rect().encode(
        x='Step:O',
        y='Node:O',
        color=alt.Color('GraphValue:Q', scale=alt.Scale(scheme='blues', domain=[vmin, vmax]), legend=alt.Legend(title="GraphValue")),
        tooltip=['Node', 'Step', 'GraphValue']
    ).properties(
        title='Cluster 1 GraphValue Heatmap'
    )
    heatmap_gv_g2 = alt.Chart(gv_df_2).mark_rect().encode(
        x='Step:O',
        y='Node:O',
        color=alt.Color('GraphValue:Q', scale=alt.Scale(scheme='blues', domain=[vmin, vmax]), legend=alt.Legend(title="GraphValue")),
        tooltip=['Node', 'Step', 'GraphValue']
    ).properties(
        title='Cluster 2 GraphValue Heatmap'
    )
    gv_sums_df = pd.DataFrame(gv_sums, columns=['Step', 'Cluster 1', 'Cluster 2', 'Total'])
    gv_sums_df = gv_sums_df.melt(id_vars='Step', var_name='Cluster', value_name='GraphValue Sum')
    line_plot_gv = alt.Chart(gv_sums_df).mark_line(point=True).encode(
        x='Step:O',
        y='GraphValue Sum:Q',
        color='Cluster:N',
        tooltip=['Step', 'Cluster', 'GraphValue Sum']
    ).properties(
        title='Sum of GraphValue'
    )

    # plot EC    
    centrality_df_1 = pd.DataFrame(centrality_data_1).reset_index().melt(id_vars='index').rename(columns={'index': 'Step', 'variable': 'Node', 'value': 'Centrality'})
    centrality_df_2 = pd.DataFrame(centrality_data_2).reset_index().melt(id_vars='index').rename(columns={'index': 'Step', 'variable': 'Node', 'value': 'Centrality'})
    vmin = min(centrality_df_1['Centrality'].min(), centrality_df_2['Centrality'].min())
    vmax = max(centrality_df_1['Centrality'].max(), centrality_df_2['Centrality'].max())
    heatmap_ec_g1 = alt.Chart(centrality_df_1).mark_rect().encode(
        x='Step:O',
        y='Node:O',
        color=alt.Color('Centrality:Q', scale=alt.Scale(scheme='blues', domain=[vmin, vmax]), legend=alt.Legend(title="Centrality")),
        tooltip=['Node', 'Step', 'Centrality']
    ).properties(
        title='Cluster 1 Eigenvector Centrality Heatmap'
    )
    heatmap_ec_g2 = alt.Chart(centrality_df_2).mark_rect().encode(
        x='Step:O',
        y='Node:O',
        color=alt.Color('Centrality:Q', scale=alt.Scale(scheme='blues', domain=[vmin, vmax]), legend=alt.Legend(title="Centrality")),
        tooltip=['Node', 'Step', 'Centrality']
    ).properties(
        title='Cluster 2 Eigenvector Centrality Heatmap'
    )
    centrality_sums_df = pd.DataFrame(centrality_sums, columns=['Step', 'Cluster 1', 'Cluster 2', 'Total'])
    centrality_sums_df = centrality_sums_df.melt(id_vars='Step', var_name='Cluster', value_name='Centrality Sum')
    line_plot_ec = alt.Chart(centrality_sums_df).mark_line(point=True).encode(
        x='Step:O',
        y='Centrality Sum:Q',
        color='Cluster:N',
        tooltip=['Step', 'Cluster', 'Centrality Sum']
    ).properties(
        title='Sum of Eigenvector Centrality'
    )

    ## plotting weights
    weight_df_1 = pd.DataFrame(weight_sum_G1).reset_index().melt(id_vars='index').rename(columns={'index': 'Step', 'variable': 'Node', 'value': 'WeightSum'})
    weight_df_2 = pd.DataFrame(weight_sum_G2).reset_index().melt(id_vars='index').rename(columns={'index': 'Step', 'variable': 'Node', 'value': 'WeightSum'})
    vmin = min(weight_df_1['WeightSum'].min(), weight_df_2['WeightSum'].min())
    vmax = max(weight_df_1['WeightSum'].max(), weight_df_2['WeightSum'].max())
    heatmap_w_g1 = alt.Chart(weight_df_1).mark_rect().encode(
        x='Step:O',
        y='Node:O',
        color=alt.Color('WeightSum:Q', scale=alt.Scale(scheme='greens', domain=[vmin, vmax]), legend=alt.Legend(title="WeightSum")),
        tooltip=['Node', 'Step', 'WeightSum']
    ).properties(
        title='Cluster 1 WeightSum Heatmap'
    )
    heatmap_w_g2 = alt.Chart(weight_df_2).mark_rect().encode(
        x='Step:O',
        y='Node:O',
        color=alt.Color('WeightSum:Q', scale=alt.Scale(scheme='greens', domain=[vmin, vmax]), legend=alt.Legend(title="WeightSum")),
        tooltip=['Node', 'Step', 'WeightSum']
    ).properties(
        title='Cluster 2 WeightSum Heatmap'
    )
    weight_sums_df = pd.DataFrame(weight_sums, columns=['Step', 'Cluster 1', 'Cluster 2', 'Total'])
    weight_sums_df = weight_sums_df.melt(id_vars='Step', var_name='Cluster', value_name='Weight Sum')
    line_plot_sum = alt.Chart(weight_sums_df).mark_line(point=True).encode(
        x='Step:O',
        y='Weight Sum:Q',
        color='Cluster:N',
        tooltip=['Step', 'Cluster', 'Weight Sum']
    ).properties(
        title='Sum of Weights'
    )

    st.altair_chart(
        alt.vconcat(
            alt.hconcat(
                heatmap_gv_g1.properties(width=200, height=100),
                heatmap_gv_g2.properties(width=200, height=100),
                line_plot_gv.properties(width=200, height=100)
            ),
            alt.hconcat(
                heatmap_ec_g1.properties(width=200, height=100),
                heatmap_ec_g2.properties(width=200, height=100),
                line_plot_ec.properties(width=200, height=100)
            ),
            alt.hconcat(
                heatmap_w_g1.properties(width=200, height=100),
                heatmap_w_g2.properties(width=200, height=100),
                line_plot_sum.properties(width=200, height=100)
            ),
        ),
        # use_container_width=True
    )

# st.sidebar.title("Graph Configuration")
with st.sidebar:
    with st.expander("Graph Config", expanded=False):
        n1 = st.slider("Number of nodes in Cluster 1", 1, 20, 15)
        p1 = st.slider("Probability of edge creation in Cluster 1", 0.1, 1.0, 0.50)
        n2 = st.slider("Number of nodes in Cluster 2", 1, 20, 3)
        p2 = st.slider("Probability of edge creation in Cluster 2", 0.1, 1.0, 0.50)
    with st.expander("Compute Settings", expanded=False):
        ec_compute_selection = st.radio("Compute Cluster Setting", ["Together", "Separate"],
                                        help='Compute metrics (EC/GV) for the entire graph together or separately for each initialized cluster.')
        use_weight_compute_ec = st.radio("Use Weight to Compute Eigenvector Centrality", ["Yes", "No"],
                                         help='Use the edge weights to compute eigenvector centrality.')

    weight_selection = st.radio("Weight Selection", ["U[0,1)", "1"], 
                                    help='The new weight of an edge between two nodes, either can be drawn from a uniform distribution U[0,1) or static value of 1.')    
    weight_scaling = st.slider("W(C2)=scale*W(C1) ", 0.1, 1.0, 0.1,
                            help='Scaling of the weight applied when creating an edge in cluster 2, relative to the weight in cluster 1.')
    which_edge_create = st.slider('P(edge) within clusters', 0.1, 1.0, 1.0, 
                                  help='Probability of edge creation within clusters.  1-P(edge) is the probability of edge creation between clusters.')
    alpha_ec = st.slider('Alpha-EC', 0.1, 1.0, 0.5, help='Alpha for EC in Graph Value.')
    alpha_w = st.slider('Alpha-W', 0.1, 1.0, 0.5, help='Alpha for Weight in Graph Value.')

    initiate = st.sidebar.button("Reset/Start")
    run_nsteps = st.sidebar.button("Run 25 Epochs")

if initiate:
    # Step 1: Create the initial clusters
    G, G1, G2 = create_initial_clusters(n1, p1, n2, p2)

    # Initial positions for the graph layout
    pos = nx.spring_layout(G)

    # Initialize the step tracker
    st.session_state['step'] = 0
    st.session_state['G'] = G
    st.session_state['G1'] = G1
    st.session_state['G2'] = G2
    st.session_state['pos'] = pos
    st.session_state['centrality_data_cluster1'] = []
    st.session_state['centrality_data_cluster2'] = []
    st.session_state['centrality_sums'] = []
    st.session_state['weight_sum_G1'] = []
    st.session_state['weight_sum_G2'] = []
    st.session_state['weight_sums'] = []
    st.session_state['graph_value_cluster1'] = []
    st.session_state['graph_value_cluster2'] = []
    st.session_state['graph_value_sums'] = []
    

    # Computation
    if ec_compute_selection == "Together":
        centrality = calculate_eigenvector_centrality(G)
        centrality_data_1 = [centrality[node] for node in range(G1.number_of_nodes())]
        centrality_data_2 = [centrality[node] for node in range(G1.number_of_nodes(), G1.number_of_nodes() + G2.number_of_nodes())]
        centrality_sum_1 = sum(centrality[node] for node in range(G1.number_of_nodes()))
        centrality_sum_2 = sum(centrality[node] for node in range(G1.number_of_nodes(), G1.number_of_nodes() + G2.number_of_nodes()))
        total_centrality_sum = sum(centrality.values())

        gv = calculate_graph_value(G)
        gv_data_1 = [gv[node] for node in range(G1.number_of_nodes())]
        gv_data_2 = [gv[node] for node in range(G1.number_of_nodes(), G1.number_of_nodes() + G2.number_of_nodes())]
        gv_sum_1 = sum(gv[node] for node in range(G1.number_of_nodes()))
        gv_sum_2 = sum(gv[node] for node in range(G1.number_of_nodes(), G1.number_of_nodes() + G2.number_of_nodes()))
        total_gv_sum = sum(gv.values())

    elif ec_compute_selection == "Separate":
        c1 = calculate_eigenvector_centrality(G1)
        c2 = calculate_eigenvector_centrality(G2)
        centrality_data_1 = [c1[node] for node in range(G1.number_of_nodes())]
        centrality_data_2 = [c2[node] for node in range(G2.number_of_nodes())]
        centrality_sum_1 = sum(c1.values())
        centrality_sum_2 = sum(c2.values())
        total_centrality_sum = centrality_sum_1 + centrality_sum_2

        gv1 = calculate_graph_value(G1)
        gv2 = calculate_graph_value(G2)
        gv_data_1 = [gv1[node] for node in range(G1.number_of_nodes())]
        gv_data_2 = [gv2[node] for node in range(G2.number_of_nodes())]
        gv_sum_1 = sum(gv1.values())
        gv_sum_2 = sum(gv2.values())
        total_gv_sum = gv_sum_1 + gv_sum_2
    
    # Weight Computation
    weights_G1 = calculate_node_weight_sums(G1)
    weights_G2 = calculate_node_weight_sums(G2)
    weight_sum_1 = sum(weights_G1.values())
    weight_sum_2 = sum(weights_G2.values())
    total_weight_sum = weight_sum_1 + weight_sum_2
    
    # the first element in this array is the "step" of the simulation
    st.session_state['centrality_sums'].append([0, centrality_sum_1, centrality_sum_2, total_centrality_sum])
    st.session_state['centrality_data_cluster1'].append(centrality_data_1)
    st.session_state['centrality_data_cluster2'].append(centrality_data_2)
    st.session_state['weight_sum_G1'].append(weights_G1)
    st.session_state['weight_sum_G2'].append(weights_G2)
    st.session_state['weight_sums'].append([0, weight_sum_1, weight_sum_2, total_weight_sum])
    st.session_state['graph_value_cluster1'].append(gv_data_1)
    st.session_state['graph_value_cluster2'].append(gv_data_2)
    st.session_state['graph_value_sums'].append([0, gv_sum_1, gv_sum_2, total_gv_sum])

    plot_graph_and_heatmaps(
        G
    )


# Check if the step tracker is initialized
# if 'step' in st.session_state:
if run_nsteps:
    G = st.session_state.G
    G1 = st.session_state.G1
    G2 = st.session_state.G2
    pos = st.session_state.pos

    # run 50 iterations
    for ii in range(25):

        # if st.sidebar.button("Next"):
        st.session_state.step += 1

        # determine whether we should create an edge within a cluster or between clusters
        if np.random.rand() < which_edge_create:
            # create an edge within each cluster
            u = np.random.randint(0, G1.number_of_nodes())
            v = np.random.randint(0, G1.number_of_nodes())
            G1_weight = get_weight()
            update_weight(G, G1, G2, 1, u, v, G1_weight)
            
            u = np.random.randint(0, G2.number_of_nodes())
            v = np.random.randint(0, G2.number_of_nodes())
            G2_weight = G1_weight * weight_scaling
            update_weight(G, G1, G2, 2, u, v, G2_weight)
        else:
            # create an edge between the two clusters
            u = np.random.randint(0, G1.number_of_nodes())
            v = np.random.randint(G1.number_of_nodes(), G.number_of_nodes())
            G_weight = get_weight()
            update_weight(G, G1, G2, 3, u, v, G_weight)


        # Calculate eigenvector centrality for the current graph
        if ec_compute_selection == "Together":
            centrality = calculate_eigenvector_centrality(G)
            centrality_data_1 = [centrality[node] for node in range(G1.number_of_nodes())]
            centrality_data_2 = [centrality[node] for node in range(G1.number_of_nodes(), G1.number_of_nodes() + G2.number_of_nodes())]
            centrality_sum_1 = sum(centrality[node] for node in range(G1.number_of_nodes()))
            centrality_sum_2 = sum(centrality[node] for node in range(G1.number_of_nodes(), G1.number_of_nodes() + G2.number_of_nodes()))
            total_centrality_sum = sum(centrality.values())

            gv = calculate_graph_value(G)
            gv_data_1 = [gv[node] for node in range(G1.number_of_nodes())]
            gv_data_2 = [gv[node] for node in range(G1.number_of_nodes(), G1.number_of_nodes() + G2.number_of_nodes())]
            gv_sum_1 = sum(gv[node] for node in range(G1.number_of_nodes()))
            gv_sum_2 = sum(gv[node] for node in range(G1.number_of_nodes(), G1.number_of_nodes() + G2.number_of_nodes()))
            total_gv_sum = sum(gv.values())
        elif ec_compute_selection == "Separate":
            c1 = calculate_eigenvector_centrality(G1)
            c2 = calculate_eigenvector_centrality(G2)
            centrality_data_1 = [c1[node] for node in range(G1.number_of_nodes())]
            centrality_data_2 = [c2[node] for node in range(G2.number_of_nodes())]
            centrality_sum_1 = sum(c1.values())
            centrality_sum_2 = sum(c2.values())
            total_centrality_sum = centrality_sum_1 + centrality_sum_2

            gv1 = calculate_graph_value(G1)
            gv2 = calculate_graph_value(G2)
            gv_data_1 = [gv1[node] for node in range(G1.number_of_nodes())]
            gv_data_2 = [gv2[node] for node in range(G2.number_of_nodes())]
            gv_sum_1 = sum(gv1.values())
            gv_sum_2 = sum(gv2.values())
            total_gv_sum = gv_sum_1 + gv_sum_2

        st.session_state['centrality_data_cluster1'].append(centrality_data_1)
        st.session_state['centrality_data_cluster2'].append(centrality_data_2)
        st.session_state['centrality_sums'].append([st.session_state.step, centrality_sum_1, centrality_sum_2, total_centrality_sum])

        st.session_state['graph_value_cluster1'].append(gv_data_1)
        st.session_state['graph_value_cluster2'].append(gv_data_2)
        st.session_state['graph_value_sums'].append([st.session_state.step, gv_sum_1, gv_sum_2, total_gv_sum])

        weight_G1 = calculate_node_weight_sums(G1)
        weight_G2 = calculate_node_weight_sums(G2)
        weight_sum_1 = sum(weight_G1.values())
        weight_sum_2 = sum(weight_G2.values())
        total_weight_sum = weight_sum_1 + weight_sum_2
        st.session_state['weight_sum_G1'].append(weight_G1)
        st.session_state['weight_sum_G2'].append(weight_G2)
        st.session_state['weight_sums'].append([st.session_state.step, weight_sum_1, weight_sum_2, total_weight_sum])

    st.session_state['G'] = G

    # Plot the graph and heatmaps
    plot_graph_and_heatmaps(
        G
    )

    
