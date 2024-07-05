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

# graph initialization
def create_initial_clusters(n1, p1, n2, p2):
    seed = np.random.randint(1, 10000)
    G1 = nx.erdos_renyi_graph(n1, p1, seed=seed+1)
    G2 = nx.erdos_renyi_graph(n2, p2, seed=seed+2)

    # Assign random weights to edges
    for (u, v) in G1.edges():
        G1[u][v]['weight'] = np.random.rand()
    for (u, v) in G2.edges():
        G2[u][v]['weight'] = np.random.rand()

    G = nx.disjoint_union_all([G1, G2])
    return G, G1, G2

def calculate_eigenvector_centrality(G):
    centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-2, weight='weight')
    return centrality

def plot_graph_and_heatmaps(G, centrality_data_1, centrality_data_2, step, pos, centrality_sums):
    fig, ax1 = plt.subplots(figsize=(8,3))

    # Plot the graph
    centrality = calculate_eigenvector_centrality(G)
    node_color = [centrality[node] for node in G.nodes()]
    nx.draw(G, pos, ax=ax1, with_labels=True, node_size=500, node_color=node_color, cmap=plt.cm.Blues, font_size=10, font_color='black', font_weight='bold', edge_color='gray')
    ax1.set_title(f"Graph at Step {step}")
    
    st.pyplot(fig, use_container_width=True)
    
    centrality_df_1 = pd.DataFrame(centrality_data_1).reset_index().melt(id_vars='index').rename(columns={'index': 'Step', 'variable': 'Node', 'value': 'Centrality'})
    centrality_df_2 = pd.DataFrame(centrality_data_2).reset_index().melt(id_vars='index').rename(columns={'index': 'Step', 'variable': 'Node', 'value': 'Centrality'})

    # graph scaling
    vmin = min(centrality_df_1['Centrality'].min(), centrality_df_2['Centrality'].min())
    vmax = max(centrality_df_1['Centrality'].max(), centrality_df_2['Centrality'].max())

    heatmap_1 = alt.Chart(centrality_df_1).mark_rect().encode(
        x='Step:O',
        y='Node:O',
        color=alt.Color('Centrality:Q', scale=alt.Scale(domain=[vmin, vmax]), legend=alt.Legend(title="Centrality")),
        tooltip=['Node', 'Step', 'Centrality']
    ).properties(
        title='Cluster 1 Eigenvector Centrality Heatmap'
    )

    heatmap_2 = alt.Chart(centrality_df_2).mark_rect().encode(
        x='Step:O',
        y='Node:O',
        color=alt.Color('Centrality:Q', scale=alt.Scale(domain=[vmin, vmax]), legend=alt.Legend(title="Centrality")),
        tooltip=['Node', 'Step', 'Centrality']
    ).properties(
        title='Cluster 2 Eigenvector Centrality Heatmap'
    )

    centrality_sums_df = pd.DataFrame(centrality_sums, columns=['Step', 'Cluster 1', 'Cluster 2', 'Total'])
    centrality_sums_df = centrality_sums_df.melt(id_vars='Step', var_name='Cluster', value_name='Centrality Sum')

    line_plot = alt.Chart(centrality_sums_df).mark_line(point=True).encode(
        x='Step:O',
        y='Centrality Sum:Q',
        color='Cluster:N',
        tooltip=['Step', 'Cluster', 'Centrality Sum']
    ).properties(
        title='Sum of Eigenvector Centrality'
    )

    st.altair_chart(
        alt.hconcat(
            heatmap_1.properties(width=200, height=200), 
            heatmap_2.properties(width=200, height=200),
            line_plot.properties(width=300, height=200)
        ),
        use_container_width=True
    )

st.sidebar.title("Graph Configuration")
n1 = st.sidebar.slider("Number of nodes in Cluster 1", 5, 20, 10)
p1 = st.sidebar.slider("Probability of edge creation in Cluster 1", 0.1, 1.0, 0.5)
n2 = st.sidebar.slider("Number of nodes in Cluster 2", 5, 20, 10)
p2 = st.sidebar.slider("Probability of edge creation in Cluster 2", 0.1, 1.0, 0.5)
initiate = st.sidebar.button("Reset/Start")

if initiate:
    # Step 1: Create the initial clusters
    G, G1, G2 = create_initial_clusters(n1, p1, n2, p2)

    # Initial graph centrality
    centrality_data_1 = []
    centrality_data_2 = []
    centrality = calculate_eigenvector_centrality(G)
    centrality_data_1.append([centrality[node] for node in range(G1.number_of_nodes())])
    centrality_data_2.append([centrality[node] for node in range(G1.number_of_nodes(), G1.number_of_nodes() + G2.number_of_nodes())])

    # Initial positions for the graph layout
    pos = nx.spring_layout(G)

    # Store the connections to be made between clusters
    connections = [(i, j) for i in range(G1.number_of_nodes()) for j in range(G1.number_of_nodes(), G.number_of_nodes())]

    # Initialize the step tracker
    st.session_state.step = 0
    st.session_state['connections'] = connections
    st.session_state['G'] = G
    st.session_state['G1'] = G1
    st.session_state['G2'] = G2
    st.session_state['pos'] = pos
    st.session_state['centrality_sums'] = []

    # Calculate initial centrality sums
    centrality_sum_1 = sum(centrality[node] for node in range(G1.number_of_nodes()))
    centrality_sum_2 = sum(centrality[node] for node in range(G1.number_of_nodes(), G1.number_of_nodes() + G2.number_of_nodes()))
    total_centrality_sum = sum(centrality.values())
    # the first element in this array is the "step" of the simulation
    st.session_state['centrality_sums'].append([0, centrality_sum_1, centrality_sum_2, total_centrality_sum])


# Check if the step tracker is initialized
if 'step' in st.session_state:
    G = st.session_state.G
    G1 = st.session_state.G1
    G2 = st.session_state.G2
    connections = st.session_state.connections
    pos = st.session_state.pos

    # Navigation buttons in sidebar
    if st.sidebar.button("Previous") and st.session_state.step > 0:
        st.session_state.step -= 1

    if st.sidebar.button("Next") and st.session_state.step < len(connections):
        st.session_state.step += 1

    # Connect the clusters up to the current step
    # G.clear_edges()
    for i in range(st.session_state.step):
        u, v = connections[i]
        G.add_edge(u, v, weight=np.random.rand())

    # Calculate eigenvector centrality for the current graph
    centrality = calculate_eigenvector_centrality(G)
    centrality_data_1 = [[centrality[node] for node in range(G1.number_of_nodes())] for _ in range(st.session_state.step + 1)]
    centrality_data_2 = [[centrality[node] for node in range(G1.number_of_nodes(), G1.number_of_nodes() + G2.number_of_nodes())] for _ in range(st.session_state.step + 1)]

    centrality_sum_1 = sum(centrality[node] for node in range(G1.number_of_nodes()))
    centrality_sum_2 = sum(centrality[node] for node in range(G1.number_of_nodes(), G1.number_of_nodes() + G2.number_of_nodes()))
    total_centrality_sum = sum(centrality.values())
    st.session_state['centrality_sums'].append([st.session_state.step, centrality_sum_1, centrality_sum_2, total_centrality_sum])

    # Plot the graph and heatmaps
    plot_graph_and_heatmaps(G, centrality_data_1, centrality_data_2, st.session_state.step, pos, st.session_state['centrality_sums'])

    st.session_state['G'] = G
    st.session_state['connections'] = connections
