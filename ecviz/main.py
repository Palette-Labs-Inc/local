import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt
from simulation import Simulation

st.set_page_config(
    page_title="Eigenvector Centrality Visualization",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

def update_alpha():
    sim = st.session_state.simulation
    sim.update_alpha_ec(alpha)
    # st.session_state.simulation = sim  # not necessary, object reference
    plot_everything(sim)

def plot_everything(sim):
    gv_1 = sim.graph_value_cluster1
    gv_2 = sim.graph_value_cluster2
    gv_sums = sim.graph_value_sums
    gv_ratios = sim.graph_value_ratios
    gv_mroc_1 = sim.graph_value_mroc_cluster1
    gv_mroc_2 = sim.graph_value_mroc_cluster2
    gv_mroc_sums = sim.graph_value_mroc_sums
    gv_mroc_ratios = sim.graph_value_mroc_ratios
    centrality_data_1 = sim.centrality_data_cluster1
    centrality_data_2 = sim.centrality_data_cluster2
    centrality_sums = sim.centrality_sums
    weight_sum_G1 = sim.weight_sum_G1
    weight_sum_G2 = sim.weight_sum_G2
    weight_sums = sim.weight_sums
    step = sim.step
    
    pos = sim.pos

    fig, ax1 = plt.subplots(figsize=(12,2))

    centrality = sim.calculate_eigenvector_centrality(sim.G)
    node_color = [centrality[node] for node in sim.G.nodes()]
    nx.draw(sim.G, pos, ax=ax1, with_labels=True, node_size=200, node_color=node_color, cmap=plt.cm.Blues, font_size=10, font_color='black', font_weight='bold', edge_color='gray')
    ax1.set_title(f"Graph at Step {step}")    
    st.pyplot(fig, use_container_width=True)

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
    gv_sums_df = pd.DataFrame(gv_sums, columns=['Step', 'NYC', 'Buffalo', 'Total'])
    gv_sums_df['NYC'] /= sim.n1
    gv_sums_df['Buffalo'] /= sim.n2
    gv_sums_df = gv_sums_df.melt(id_vars='Step', var_name='Cluster', value_name='GraphValue Sum')
    line_plot_gv = alt.Chart(gv_sums_df).mark_line(point=True).encode(
        x='Step:O',
        y='GraphValue Sum:Q',
        color='Cluster:N',
        tooltip=['Step', 'Cluster', 'GraphValue Sum']
    ).properties(
        title='Node Avg. GraphValue'
    )
    gv_ratio_df = pd.DataFrame(gv_ratios, columns=['Step', 'Ratio'])
    gv_ratio_df['Ratio'] = gv_ratio_df['Ratio'] / (sim.n1 / sim.n2)
    line_plot_gv_ratio = alt.Chart(gv_ratio_df).mark_line(point=True).encode(
        x='Step:O',
        y='Ratio:Q',
        tooltip=['Step', 'Ratio']
    ).properties(
        title='Normalized Avg. GraphValue Ratio'
    )

    gv_mroc_df_1 = pd.DataFrame(gv_mroc_1).reset_index().melt(id_vars='index').rename(columns={'index': 'Step', 'variable': 'Node', 'value': 'GraphValue'})
    gv_mroc_df_2 = pd.DataFrame(gv_mroc_2).reset_index().melt(id_vars='index').rename(columns={'index': 'Step', 'variable': 'Node', 'value': 'GraphValue'})
    vmin = min(gv_mroc_df_1['GraphValue'].min(), gv_mroc_df_2['GraphValue'].min())
    vmax = max(gv_mroc_df_1['GraphValue'].max(), gv_mroc_df_2['GraphValue'].max())
    heatmap_gv_mroc_g1 = alt.Chart(gv_mroc_df_1).mark_rect().encode(
        x='Step:O',
        y='Node:O',
        color=alt.Color('GraphValue:Q', scale=alt.Scale(scheme='blues', domain=[vmin, vmax]), legend=alt.Legend(title="GraphValue")),
        tooltip=['Node', 'Step', 'GraphValue']
    ).properties(
        title='Cluster 1 GraphValue (MROC) Heatmap'
    )
    heatmap_gv_mroc_g2 = alt.Chart(gv_mroc_df_2).mark_rect().encode(
        x='Step:O',
        y='Node:O',
        color=alt.Color('GraphValue:Q', scale=alt.Scale(scheme='blues', domain=[vmin, vmax]), legend=alt.Legend(title="GraphValue")),
        tooltip=['Node', 'Step', 'GraphValue']
    ).properties(
        title='Cluster 2 GraphValue (MROC) Heatmap'
    )
    gv_mroc_sums_df = pd.DataFrame(gv_mroc_sums, columns=['Step', 'NYC', 'Buffalo', 'Total'])
    gv_mroc_sums_df['NYC'] /= sim.n1
    gv_mroc_sums_df['Buffalo'] /= sim.n2
    gv_mroc_sums_df = gv_mroc_sums_df.melt(id_vars='Step', var_name='Cluster', value_name='GraphValue Sum')
    line_plot_mroc_gv = alt.Chart(gv_mroc_sums_df).mark_line(point=True).encode(
        x='Step:O',
        y='GraphValue Sum:Q',
        color='Cluster:N',
        tooltip=['Step', 'Cluster', 'GraphValue Sum']
    ).properties(
        title='Sum of GraphValue (MROC)'
    )
    gv_mroc_ratio_df = pd.DataFrame(gv_mroc_ratios, columns=['Step', 'Ratio'])
    gv_mroc_ratio_df['Ratio'] = gv_mroc_ratio_df['Ratio'] / (sim.n1 / sim.n2)
    line_plot_gv_mroc_ratio = alt.Chart(gv_mroc_ratio_df).mark_line(point=True).encode(
        x='Step:O',
        y='Ratio:Q',
        tooltip=['Step', 'Ratio']
    ).properties(
        title='Normalized GraphValue Ratio'
    )

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
                line_plot_gv.properties(width=200, height=100),
                line_plot_gv_ratio.properties(width=200, height=100)
            ),
            alt.hconcat(
                heatmap_gv_mroc_g1.properties(width=200, height=100),
                heatmap_gv_mroc_g2.properties(width=200, height=100),
                line_plot_mroc_gv.properties(width=200, height=100),
                line_plot_gv_mroc_ratio.properties(width=200, height=100)
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
    )

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
    alpha = st.slider('Alpha', 0.1, 1.0, 0.5, 
                         on_change=update_alpha,
                         help='Alpha for EC in Graph Value.',)

    initiate = st.sidebar.button("Reset/Start")
    run_nsteps = st.sidebar.button("Run 25 Epochs")

if initiate:
    sim = Simulation(n1, p1, n2, p2, weight_selection, weight_scaling, which_edge_create, alpha, ec_compute_selection, use_weight_compute_ec)
    sim.initialize_simulation()
    st.session_state.simulation = sim

if 'simulation' in st.session_state:
    sim = st.session_state.simulation
    if run_nsteps:
        for _ in range(25):
            sim.run_epoch()
        plot_everything(sim)
