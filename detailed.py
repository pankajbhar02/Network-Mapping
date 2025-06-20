import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit as st
from streamlit.components.v1 import html
import tempfile
import os
import math

# --- 1. Data Processing and Graph Creation Functions ---

@st.cache_data
def get_all_hcps_details(df):
    """
    Computes aggregated details for all HCPs from the dataframe.
    """
    hcp1_cols = ['NPI_1', 'HCP_1', 'No. of Connections HCP 1', 'Influence score_1', 'City1', 'State1', 'Papers', 'Panels', 'Trials']
    hcp2_cols = ['NPI_2', 'HCP_2', 'No. of Connections HCP 2', 'Influence score_2', 'City2', 'State2', 'Papers', 'Panels', 'Trials']
    common_cols = ['NPI', 'hcp_name', 'connections', 'influence', 'city', 'state', 'papers', 'panels', 'trials']

    hcp1_renamed = df[hcp1_cols].rename(columns=dict(zip(hcp1_cols, common_cols)))
    hcp2_renamed = df[hcp2_cols].rename(columns=dict(zip(hcp2_cols, common_cols)))
    
    all_hcps_details = pd.concat([hcp1_renamed, hcp2_renamed]).groupby('NPI', as_index=False).agg(
        hcp_name=('hcp_name', lambda x: x.dropna().iloc[0] if not x.dropna().empty else 'N/A'),
        connections=('connections', 'max'),
        influence=('influence', 'first'),
        city=('city', 'first'),
        state=('state', 'first'),
        papers=('papers', lambda x: x.fillna(0).max()),
        panels=('panels', lambda x: x.fillna(0).max()),
        trials=('trials', lambda x: x.fillna(0).max())
    )
    return all_hcps_details

@st.cache_data
def get_filtered_nodes(_all_hcps_details, top_n=50, selected_states=None, selected_cities=None,
                      min_connections=0, min_influence=0, min_papers=0, min_panels=0, min_trials=0,
                      sort_by='connections'):
    """
    Filters HCPs (nodes) based on all criteria and sorts by specified metric.
    """
    filtered_nodes_df = _all_hcps_details.copy()

    if selected_states:
        filtered_nodes_df = filtered_nodes_df[filtered_nodes_df['state'].isin(selected_states)]
    if selected_cities:
        filtered_nodes_df = filtered_nodes_df[filtered_nodes_df['city'].isin(selected_cities)]

    filtered_nodes_df = filtered_nodes_df[
        (filtered_nodes_df['connections'] >= min_connections) &
        (filtered_nodes_df['influence'] >= min_influence) &
        (filtered_nodes_df['papers'] >= min_papers) &
        (filtered_nodes_df['panels'] >= min_panels) &
        (filtered_nodes_df['trials'] >= min_trials)
    ]
    
    if not filtered_nodes_df.empty:
        filtered_nodes_df = filtered_nodes_df.nlargest(min(top_n, len(filtered_nodes_df)), sort_by)
    else:
        return pd.DataFrame()
    return filtered_nodes_df

@st.cache_data
def get_filtered_edges_for_display(_original_df, filtered_node_npis, min_strength, max_strength):
    """
    Filters edges based on connection strength and ensures both connected nodes are in filtered_node_npis.
    """
    if not filtered_node_npis:
        return pd.DataFrame()

    filtered_edges = _original_df[
        (_original_df['NPI_1'].isin(filtered_node_npis)) & 
        (_original_df['NPI_2'].isin(filtered_node_npis)) &
        (_original_df['Overall Connection Strength'] >= min_strength) & 
        (_original_df['Overall Connection Strength'] <= max_strength)
    ].copy()
    return filtered_edges

@st.cache_data
def create_pyvis_network(_nodes_df, _edges_df, enable_physics,
                        global_min_connections, global_max_connections,
                        global_min_edge_strength, global_max_edge_strength):
    """
    Creates an interactive Pyvis network.
    """
    net = Network(
        height="900px", width="100%", bgcolor="#0a0a0a", font_color=None,
        directed=True, notebook=False, heading=""
    )
    
    net.force_atlas_2based(
        gravity=-500, central_gravity=0.03,
        spring_length=400, spring_strength=0.002,
        damping=1.5, overlap=1
    )
    net.toggle_physics(enable_physics)

    if not _nodes_df.empty:
        for index, node_data in _nodes_df.iterrows():
            node = node_data['NPI']
            hcp_name = node_data.get('hcp_name', str(node))
            connections = node_data.get('connections', 0)
            
            norm_connections = 0.5 if (global_max_connections - global_min_connections) == 0 else \
                              max(0, min(1, (connections - global_min_connections) / (global_max_connections - global_min_connections)))
            size = 15 + norm_connections * (70 - 15)

            bg_color = 'hsl(120, 100%, 40%)'
            border_color = 'hsl(60, 100%, 30%)'

            tooltip = f"""
            NPI ID: {node}<br/>
            HCP Name: {hcp_name}<br/>
            Connections: {connections}<br/>
            Influence Score: {node_data.get('influence', 0):.2f}<br/>
            Location: {node_data.get('city', 'N/A')}, {node_data.get('state', 'N/A')}<br/>
            Papers: {node_data.get('papers', 0)}<br/>
            Panels: {node_data.get('panels', 0)}<br/>
            Trials: {node_data.get('trials', 0)}
            """

            net.add_node(
                node, label=hcp_name, size=size, shape='dot',
                color={'background': bg_color, 'border': border_color, 'highlight': {'background': '#FFD700', 'border': '#FFA500'}},
                title=tooltip, borderWidth=2, borderWidthSelected=4,
                font={'size': 35, 'color': '#FFFFFF', 'bold': True},
                mass=max(1, connections / 50),
                physics=enable_physics
            )

    if not _edges_df.empty:
        for index, edge_data in _edges_df.iterrows():
            u, v = edge_data['NPI_1'], edge_data['NPI_2']
            weight = edge_data.get('Overall Connection Strength', 0)
            
            if net.get_node(u) and net.get_node(v):
                norm_weight = 0.5 if (global_max_edge_strength - global_min_edge_strength) == 0 else \
                             max(0, min(1, (weight - global_min_edge_strength) / (global_max_edge_strength - global_min_edge_strength)))
                width = 1 + norm_weight * (10 - 1)

                infl_u = _nodes_df[_nodes_df['NPI'] == u]['influence'].iloc[0] if not _nodes_df[_nodes_df['NPI'] == u].empty else 0
                infl_v = _nodes_df[_nodes_df['NPI'] == v]['influence'].iloc[0] if not _nodes_df[_nodes_df['NPI'] == v].empty else 0
                source_node, target_node = (u, v) if infl_u >= infl_v else (v, u)

                edge_color = f'rgba(135, 206, 235, {0.4 + norm_weight * 0.5})'
                net.add_edge(
                    source_node, target_node, width=width, color={'color': edge_color, 'highlight': 'rgba(255, 255, 0, 0.9)'},
                    arrows={'to': {'enabled': True, 'scaleFactor': 0.8}},
                    arrowStrikethrough=False,
                    smooth={'type': 'curvedCW', 'roundness': 0.15},
                    physics=enable_physics
                )

    javascript_code = """
    function setupNetworkInteractivity(network) {
        network.on("click", function (properties) {
            if (properties.nodes.length === 0) {
                network.body.data.nodes.forEach(function(node) {
                    node.color = { background: node.color.background, border: node.color.border };
                    node.borderWidth = 2;
                });
                network.body.data.nodes.update(network.body.data.nodes.get());
                return;
            }

            var nodeId = properties.nodes[0];
            if (nodeId) {
                var connectedNodes = network.getConnectedNodes(nodeId);
                var allNodesToHighlight = [nodeId].concat(connectedNodes);
                network.body.data.nodes.forEach(function(node) {
                    if (allNodesToHighlight.includes(node.id)) {
                        node.color = node.color.highlight;
                        node.borderWidth = 4;
                    } else {
                        node.color = { background: 'rgba(50,50,50,0.2)', border: 'rgba(30,30,30,0.2)' };
                        node.borderWidth = 1;
                    }
                });
                network.body.data.nodes.update(network.body.data.nodes.get());
                network.focus(nodeId, { scale: 1.5, animation: { duration: 500 } });
            }
        });
    }
    """
    
    net.html = net.html.replace(
        'var network = new vis.Network(container, data, options);',
        'var network = new vis.Network(container, data, options);\nsetupNetworkInteractivity(network);'
    )
    net.html = net.html.replace(
        '</body>',
        '<script type="text/javascript">' + javascript_code + '</script>\n</body>'
    )
    
    return net

@st.cache_data
def generate_hcp_summary_data(selected_npi, _all_hcps_details_df, _original_df, default_top_n=5):
    """
    Generates HCP summary data with lightweight high-impact metrics.
    """
    if selected_npi is None or _all_hcps_details_df.empty:
        return None

    hcp_data = _all_hcps_details_df[_all_hcps_details_df['NPI'] == selected_npi]
    if hcp_data.empty:
        return None

    hcp_data = hcp_data.iloc[0]
    total_connections = hcp_data['connections']
    influence_score = hcp_data['influence']
    papers, panels, trials = int(hcp_data['papers']), int(hcp_data['panels']), int(hcp_data['trials'])
    
    # Influence Rank
    influence_rank_text = " (only HCP in the network)"
    influence_percentile = 0
    if len(_all_hcps_details_df) > 1:
        influence_percentile = ((_all_hcps_details_df['influence'] < influence_score).sum() / (len(_all_hcps_details_df) - 1) * 100) if influence_score != _all_hcps_details_df['influence'].min() else 0
        influence_rank_text = f"placing them in the top {100 - influence_percentile:.1f}% of all HCPs"

    # Direct Connections
    direct_connections_df = _original_df[
        (_original_df['NPI_1'] == selected_npi) | (_original_df['NPI_2'] == selected_npi)
    ].copy()
    avg_connection_strength = direct_connections_df['Overall Connection Strength'].mean() if not direct_connections_df.empty else 0.0

    # Network Diversity
    connected_npis_temp = set(direct_connections_df['NPI_1'].tolist() + direct_connections_df['NPI_2'].tolist())
    connected_npis_temp.discard(selected_npi)
    connected_hcps_details = _all_hcps_details_df[_all_hcps_details_df['NPI'].isin(connected_npis_temp)]
    unique_states_connected = len(connected_hcps_details['state'].dropna().unique())
    unique_cities_connected = len(connected_hcps_details['city'].dropna().unique())

    # Dominant Metrics
    all_direct_metrics = direct_connections_df['Metrics'].dropna().tolist() if 'Metrics' in direct_connections_df.columns else []
    metric_counts = pd.Series(all_direct_metrics).value_counts()
    dominant_metrics = metric_counts.nlargest(2).index.tolist() if not metric_counts.empty else []
    dominant_metrics_text = " and ".join(dominant_metrics) if dominant_metrics else "no specific dominant metric identified"

    # Collaboration Diversity Score (Entropy-based)
    if metric_counts.empty:
        collaboration_diversity_score = 0.0
    else:
        probabilities = metric_counts / metric_counts.sum()
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
        max_entropy = math.log2(len(metric_counts)) if len(metric_counts) > 0 else 1
        collaboration_diversity_score = entropy / max_entropy if max_entropy > 0 else 0.0

    # KOL Status
    kol_thresholds = {
        'influence': _all_hcps_details_df['influence'].quantile(0.9),
        'connections': _all_hcps_details_df['connections'].quantile(0.9),
        'activity': (_all_hcps_details_df['papers'] + _all_hcps_details_df['panels'] + _all_hcps_details_df['trials']).quantile(0.9)
    }
    is_kol = (
        influence_score >= kol_thresholds['influence'] and
        total_connections >= kol_thresholds['connections'] and
        (papers + panels + trials) >= kol_thresholds['activity']
    )
    kol_status = "Key Opinion Leader" if is_kol else "Standard HCP"

    # Top Connections
    top_connections_list = []
    if not direct_connections_df.empty:
        for index, row in direct_connections_df.iterrows():
            connected_npi = row['NPI_2'] if row['NPI_1'] == selected_npi else row['NPI_1']
            connected_hcp_data = _all_hcps_details_df[_all_hcps_details_df['NPI'] == connected_npi]
            
            if not connected_hcp_data.empty:
                connected_hcp_row = connected_hcp_data.iloc[0]
                strength = row['Overall Connection Strength']
                influence = connected_hcp_row['influence']
                connection_metric_type = row.get('Metrics', 'General Collaboration')
                
                impact_statement_detail = "Engaged in a key professional collaboration."
                if connection_metric_type == 'Publishers':
                    impact_statement_detail = f"Co-authored on publications."
                elif connection_metric_type == 'Affiliations':
                    impact_statement_detail = f"Shared professional affiliations."
                elif connection_metric_type == 'Promotional Events':
                    impact_statement_detail = f"Collaborated on promotional events."
                elif connection_metric_type == 'Clinical Trials':
                    impact_statement_detail = f"Participated in clinical trials."
                elif connection_metric_type == 'Panels':
                    impact_statement_detail = f"Contributed to advisory panels."
                elif connected_hcp_row['papers'] > 0:
                    impact_statement_detail = f"Collaborated on {int(connected_hcp_row['papers'])} papers."
                elif connected_hcp_row['panels'] > 0:
                    impact_statement_detail = f"Contributed to {int(connected_hcp_row['panels'])} panels."
                elif connected_hcp_row['trials'] > 0:
                    impact_statement_detail = f"Engaged in {int(connected_hcp_row['trials'])} trials."

                top_connections_list.append({
                    'name': connected_hcp_row['hcp_name'],
                    'npi': connected_npi,
                    'strength': strength,
                    'influence': influence,
                    'metric_type': connection_metric_type,
                    'impact_statement': impact_statement_detail
                })
        
        top_connections_list.sort(key=lambda x: (0.6 * x['influence'] + 0.4 * x['strength']), reverse=True)

    # Metrics Breakdown
    defined_metrics = ['Publishers', 'Affiliations', 'Promotional Events', 'Clinical Trials', 'Panels']
    connection_metrics_counts = {metric: 0 for metric in defined_metrics}
    other_metrics_count = 0

    if 'Metrics' in direct_connections_df.columns:
        for metric_value in direct_connections_df['Metrics'].dropna():
            if metric_value in connection_metrics_counts:
                connection_metrics_counts[metric_value] += 1
            else:
                other_metrics_count += 1
    
    total_metric_instances = sum(connection_metrics_counts.values()) + other_metrics_count
    metrics_breakdown = []
    metrics_interpretation = ""
    if total_metric_instances > 0:
        for metric, count in connection_metrics_counts.items():
            if count > 0:
                percentage = (count / total_metric_instances) * 100
                metrics_breakdown.append(f"- **{metric}**: **{count}** connections (**{percentage:.1f}%** of direct metrics)")
        
        if other_metrics_count > 0:
            other_percentage = (other_metrics_count / total_metric_instances) * 100
            metrics_breakdown.append(f"- **Other/Undefined**: **{other_metrics_count}** connections (**{other_percentage:.1f}%** of direct metrics)")
        
        if metric_counts.empty:
            metrics_interpretation = "This HCP's connection dynamics are not clearly categorized by the standard metrics."
        else:
            sorted_metric_items = sorted(connection_metrics_counts.items(), key=lambda x: x[1], reverse=True)
            dominant_category_val = sorted_metric_items[0][0] if sorted_metric_items[0][1] > 0 else 'diverse engagement'
            if len(sorted_metric_items) > 1 and sorted_metric_items[1][1] > 0:
                secondary_category_val = sorted_metric_items[1][0]
                metrics_interpretation = f"This highlights a strong focus on **{dominant_category_val}** connections, with notable engagement in **{secondary_category_val}**."
            else:
                metrics_interpretation = f"This HCP's network shows a strong focus on **{dominant_category_val}** connections."
    else:
        metrics_breakdown.append("- **No specific connection metrics or direct connections found for this HCP to analyze dynamics.**")

    return {
        'hcp_data': hcp_data,
        'total_connections': total_connections,
        'influence_score': influence_score,
        'influence_rank_text': influence_rank_text,
        'influence_percentile': influence_percentile,
        'avg_connection_strength': avg_connection_strength,
        'unique_states_connected': unique_states_connected,
        'unique_cities_connected': unique_cities_connected,
        'dominant_metrics_text': dominant_metrics_text,
        'top_connections_list': top_connections_list,
        'metrics_breakdown': metrics_breakdown,
        'metrics_interpretation': metrics_interpretation,
        'papers': papers,
        'panels': panels,
        'trials': trials,
        'collaboration_diversity_score': collaboration_diversity_score,
        'kol_status': kol_status
    }

def render_hcp_summary(summary_data, selected_npi, default_top_n=5):
    """
    Renders the HCP summary with enhanced visuals and navigation.
    """
    if not summary_data:
        st.error(f"No summary available for the selected HCP (NPI: {selected_npi}). Please ensure the HCP exists in the dataset.")
        return

    hcp_data = summary_data['hcp_data']

    # Table of Contents in Sidebar
    with st.sidebar:
        st.subheader("Summary Sections")
        st.markdown("""
            <style>
            .toc-button {
                background-color: #2a2a4a;
                color: #E0E0E0;
                border-radius: 8px;
                padding: 8px;
                margin-bottom: 5px;
                width: 100%;
                text-align: left;
                border: 1px solid #4a4a6a;
                cursor: pointer;
            }
            .toc-button:hover {
                background-color: #3a3a5a;
                color: #00BFFF;
            }
            </style>
        """, unsafe_allow_html=True)
        st.markdown('<button class="toc-button', unsafe_allow_html=True)
        st.markdown('<button class="toc-button" onclick="window.location.href=\'#overview\'">I. Overview & KOL Status</button>', unsafe_allow_html=True)
        st.markdown('<button class="toc-button" onclick="window.location.href=\'#connections\'">II. Top Connections</button>', unsafe_allow_html=True)
        st.markdown('<button class="toc-button" onclick="window.location.href=\'#metrics\'">III. Connection Metrics</button>', unsafe_allow_html=True)

    # Main Summary Content
    st.markdown(f"### {hcp_data['hcp_name']} (NPI: {selected_npi}) Summary")
    st.markdown("---")

    # Section I: Overview & KOL Status
    st.markdown('<a id="overview"></a>', unsafe_allow_html=True)
    with st.expander("I. Overview & KOL Status", expanded=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            # Color-coded Total Connections
            conn_color = "#00FF00" if summary_data['total_connections'] > st.session_state['all_hcps_details']['connections'].quantile(0.75) else "#FFA500"
            conn_arrow = "🟢↑" if summary_data['total_connections'] > st.session_state['all_hcps_details']['connections'].mean() else "🔴↓"
            st.markdown(f"<span style='color:{conn_color}'>Total Connections: **{summary_data['total_connections']}** {conn_arrow}</span>", unsafe_allow_html=True)
            st.markdown(f"{summary_data['influence_rank_text']}")

            # Color-coded Influence Score
            infl_color = "#00FF00" if summary_data['influence_percentile'] > 75 else "#FFA500"
            infl_arrow = "🟢↑" if summary_data['influence_score'] > st.session_state['all_hcps_details']['influence'].mean() else "🔴↓"
            st.markdown(f"<span style='color:{infl_color}'>Influence Score: **{summary_data['influence_score']:.2f}** {infl_arrow}</span>", unsafe_allow_html=True)
        
        with col2:
            # KOL Status Badge
            kol_color = "#00BFFF" if summary_data['kol_status'] == "Key Opinion Leader" else "#808080"
            st.markdown(f"<span style='background-color:{kol_color};color:white;padding:5px;border-radius:5px'>{summary_data['kol_status']}</span>", unsafe_allow_html=True)
            st.markdown(f"Network Diversity: **{summary_data['unique_states_connected']} States**, **{summary_data['unique_cities_connected']} Cities**")
            # Collaboration Diversity Progress Bar
            st.progress(summary_data['collaboration_diversity_score'])
            st.markdown(f"Collaboration Diversity Score: **{summary_data['collaboration_diversity_score']:.2f}** (Higher = More Diverse)")

    # Section II: Top Connections
    st.markdown('<a id="connections"></a>', unsafe_allow_html=True)
    with st.expander("II. Top Connections", expanded=True):
        max_connections = len(summary_data['top_connections_list']) if summary_data['top_connections_list'] else 1
        top_n_connections = st.number_input(
            "Number of top connections to display:",
            min_value=1,
            max_value=max_connections,
            value=min(default_top_n, max_connections),
            step=1,
            key=f"top_n_connections_{selected_npi}",
            help="Enter the number of top influential connections to show."
        )
        top_connections_display = summary_data['top_connections_list'][:min(top_n_connections, len(summary_data['top_connections_list']))]
        if top_connections_display:
            # Prepare data for table
            table_data = []
            for conn in top_connections_display:
                infl_color = "#00FF00" if conn['influence'] > st.session_state['all_hcps_details']['influence'].quantile(0.75) else "#FFA500"
                table_data.append({
                    'HCP Name': conn['name'],
                    'NPI': conn['npi'],
                    'Influence Score': f"<span style='color:{infl_color}'>{conn['influence']:.2f}</span>",
                    'Connection Type': conn['metric_type'],
                    'Impact': conn['impact_statement']
                })
            # Display table
            st.markdown("""
                <style>
                .stDataFrame {
                    background-color: #1a1a2e;
                    color: #E0E0E0;
                    border-radius: 8px;
                    overflow-x: auto;
                }
                .stDataFrame table {
                    width: 100%;
                    border-collapse: collapse;
                }
                .stDataFrame th, .stDataFrame td {
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #4a4a6a;
                }
                .stDataFrame th {
                    background-color: #2a2a4a;
                    color: #00BFFF;
                }
                </style>
            """, unsafe_allow_html=True)
            st.dataframe(
                pd.DataFrame(table_data),
                use_container_width=True,
                hide_index=True,
                column_config={
                    'HCP Name': st.column_config.TextColumn(width='medium'),
                    'NPI': st.column_config.TextColumn(width='small'),
                    'Influence Score': st.column_config.TextColumn(width='small'),
                    'Connection Type': st.column_config.TextColumn(width='medium'),
                    'Impact': st.column_config.TextColumn(width='large')
                }
            )
        else:
            st.info("No direct connections found for this HCP.")

    # Section III: Connection Metrics
    st.markdown('<a id="metrics"></a>', unsafe_allow_html=True)
    with st.expander("III. Connection Metrics", expanded=True):
        st.markdown("**How You Connect**:")
        for metric in summary_data['metrics_breakdown']:
            st.markdown(f"{metric}")
        st.markdown(f"**Metrics Interpretation**: {summary_data['metrics_interpretation']}")
        st.markdown(f"**Dominant Connection Drivers**: {summary_data['dominant_metrics_text']}")

def main():
    st.set_page_config(page_title="HCP Network Visualization", layout="wide", initial_sidebar_state="expanded")
    
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        html, body, [class*="st-"] {
            font-family: 'Inter', sans-serif;
            color: #E0E0E0;
        }
        .stApp {
            background: linear-gradient(to bottom, #0a0a0a, #1a1a2e);
        }
        .custom-title {
            text-align: center;
            font-size: 32px;
            font-weight: 700;
            color: #e0e0e0;
            margin-top: 0px;
            margin-bottom: 8px;
        }
        .st-emotion-cache-1cypq8t, .st-emotion-cache-ch5erd, .st-emotion-cache-10q7qj3, .st-emotion-cache-vxbktz {
            color: #E0E0E0 !important;
            background-color: #1a1a2e;
            border-right: 1px solid #333;
        }
        .stSlider > label, .stCheckbox span, .stSelectbox label, .stMultiSelect label, .stNumberInput label, .stTextInput label {
            color: #E0E0E0;
        }
        .stMarkdown h1, h2, h3, h4, h5, h6 {
            color: #00BFFF;
            font-weight: 600;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s ease, transform 0.2s ease;
        }
        .stButton>button:hover {
            background-color: #0056b3;
            transform: translateY(-2px);
        }
        .stSelectbox[data-testid="stSelectbox"] > label {
            color: #E0E0E0;
        }
        .stSelectbox[data-testid="stSelectbox"] > div[data-baseweb="select"] > div {
            background-color: #2a2a4a;
            color: #E0E0E0;
            border-radius: 8px;
            border: 1px solid #4a4a6a;
        }
        .stDataFrame {
            background-color: #1a1a2e;
            color: #E0E0E0;
            border-radius: 8px;
            overflow-x: auto;
        }
        .stSpinner > div > div {
            color: #00BFFF;
        }
        .streamlit-expanderHeader {
            background-color: #2a2a4a;
            color: #00BFFF;
            border-radius: 8px;
            padding: 10px;
            font-weight: 600;
            border: 1px solid #4a4a6a;
        }
        .streamlit-expanderContent {
            background-color: #1a1a2e;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #4a4a6a;
            margin-top: -10px;
            box-shadow: inset 0 0 8px rgba(0,0,0,0.3);
        }
        .hcp-selector-container {
            position: fixed;
            top: 10px;
            right: 10px;
            z-index: 1000;
            background-color: #2a2a4a;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #4a4a6a;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            max-width: 300px;
        }
        .hcp-selector-container .stSelectbox > div[data-baseweb="select"] > div {
            background-color: #3a3a5a;
            color: #E0E0E0;
            border-radius: 4px;
        }
        .hcp-selector-container .stButton>button {
            width: 100%;
            margin-top: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state['page'] = 'main'
    if 'selected_hcp_npi' not in st.session_state:
        st.session_state['selected_hcp_npi'] = None
    if 'summary_data' not in st.session_state:
        st.session_state['summary_data'] = None
    if 'all_hcps_details' not in st.session_state:
        st.session_state['all_hcps_details'] = None
    if 'df' not in st.session_state:
        st.session_state['df'] = None

    # Load data
    try:
        df = pd.read_csv("Main DB_1.csv", low_memory=False)
        st.session_state['df'] = df
    except FileNotFoundError:
        st.error("File 'Main DB_1.csv' not found.")
        return
    except pd.errors.EmptyDataError:
        st.error("The CSV file is empty.")
        return
    except KeyError as e:
        st.error(f"Missing required column in CSV: {e}.")
        st.info("""
        Expected columns: 'NPI_1', 'HCP_1', 'NPI_2', 'HCP_2', 'No. of Connections HCP 1', 'No. of Connections HCP 2',
        'Influence score_1', 'Influence score_2', 'City1', 'State1', 'City2', 'State2',
        'Overall Connection Strength', and optionally 'Papers', 'Panels', 'Trials', 'Metrics'.
        """)
        return
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return

    all_hcps_details = get_all_hcps_details(df)
    st.session_state['all_hcps_details'] = all_hcps_details

    if all_hcps_details.empty:
        st.error("No HCP data available after processing. Please check the input CSV file.")
        return

    global_min_connections = 0
    global_max_connections = int(all_hcps_details['connections'].max()) if not all_hcps_details.empty else 100
    global_min_influence = 0.0
    global_max_influence = float(all_hcps_details['influence'].max()) if not all_hcps_details.empty else 1.0
    global_min_edge_strength = 0.0
    global_max_edge_strength = float(df['Overall Connection Strength'].max()) if not df.empty else 1.5

    unique_states = sorted(all_hcps_details['state'].dropna().unique())
    unique_cities = sorted(all_hcps_details['city'].dropna().unique())
    hcp_names = sorted(all_hcps_details['hcp_name'].dropna().unique().tolist())
    hcp_names.insert(0, "Select an HCP for Summary")

    # Sidebar
    with st.sidebar:
        st.subheader("Navigation")
        if st.session_state['page'] == 'main':
            if st.button("Go to HCP Summary", disabled=not st.session_state['selected_hcp_npi']):
                st.session_state['page'] = 'summary'
                st.rerun()
        else:
            if st.button("Back to Main Page"):
                st.session_state['page'] = 'main'
                st.rerun()

        if st.session_state['page'] == 'main':
            st.subheader("Geographic Filters")
            selected_states = st.multiselect(
                "Filter by States", unique_states,
                help="Select states to include HCPs from."
            )
            selected_cities = st.multiselect(
                "Filter by Cities", unique_cities,
                help="Select cities to include HCPs from."
            )
            st.subheader("HCP Core Scores")
            sort_by = st.selectbox(
                "Select Top HCPs By",
                options=["No. of Connections", "Influence Score"],
                index=0,
                help="Choose whether to display top HCPs based on connections or influence score."
            )
            sort_by = 'connections' if sort_by == "No. of Connections" else 'influence'
            min_strength, max_strength = st.slider(
                "Collaboration Score Range",
                min_value=global_min_edge_strength, max_value=global_max_edge_strength,
                value=(global_min_edge_strength, global_max_edge_strength), step=0.01
            )
            min_connections, max_connections = st.slider(
                "No. of Connections", min_value=0, max_value=global_max_connections,
                value=(0, global_max_connections)
            )
            min_influence, max_influence = st.slider(
                "Influence Score", min_value=global_min_influence, max_value=global_max_influence,
                value=(global_min_influence, global_max_influence), step=0.01, format="%.2f"
            )
            st.subheader("HCP Activity Metrics")
            max_papers_val = int(all_hcps_details['papers'].max()) if not all_hcps_details.empty else 100
            min_papers, max_papers = st.slider(
                "Number of Papers", min_value=0, max_value=max_papers_val,
                value=(0, max_papers_val)
            )
            max_panels_val = int(all_hcps_details['panels'].max()) if not all_hcps_details.empty else 100
            min_panels, max_panels = st.slider(
                "Number of Panels", min_value=0, max_value=max_panels_val,
                value=(0, max_panels_val)
            )
            max_trials_val = int(all_hcps_details['trials'].max()) if not all_hcps_details.empty else 100
            min_trials, max_trials = st.slider(
                "Number of Trials", min_value=0, max_value=max_trials_val,
                value=(0, max_trials_val)
            )
            max_top_n_for_input = len(all_hcps_details) if not all_hcps_details.empty else 1
            top_n = st.number_input(
                "Number of Top HCPs",
                min_value=1, max_value=max_top_n_for_input, value=min(50, max_top_n_for_input), step=10
            )
            st.subheader("Network Behavior")
            enable_physics = st.checkbox("Enable Physics Simulation", True)
            if st.button("Reset Network View"):
                st.session_state['reset_view'] = True

    # Main content
    if st.session_state['page'] == 'main':
        st.markdown('<div class="custom-title">HCP Network Mapping</div>', unsafe_allow_html=True)
        
        # HCP Selector Container (Fixed Position)
        with st.container():
            st.markdown(
                """
                <div class="hcp-selector-container">
                    <style>
                    .hcp-selector-container .stSelectbox > div[data-baseweb="select"] > div {
                        width: 100%;
                    }
                    </style>
                """,
                unsafe_allow_html=True
            )
            selected_hcp_name_for_summary = st.selectbox(
                "Select an HCP for Summary:",
                hcp_names,
                key="hcp_summary_selector",
                label_visibility="collapsed"  # Hide label for compact look
            )
            if selected_hcp_name_for_summary != "Select an HCP for Summary":
                try:
                    st.session_state['selected_hcp_npi'] = all_hcps_details[all_hcps_details['hcp_name'] == selected_hcp_name_for_summary]['NPI'].iloc[0]
                    st.session_state['summary_data'] = generate_hcp_summary_data(
                        st.session_state['selected_hcp_npi'], all_hcps_details, df
                    )
                except IndexError:
                    st.error(f"HCP '{selected_hcp_name_for_summary}' not found in the dataset. Please select a valid HCP.")
                    st.session_state['selected_hcp_npi'] = None
                    st.session_state['summary_data'] = None
            else:
                st.session_state['selected_hcp_npi'] = None
                st.session_state['summary_data'] = None
            st.button(
                "Go to HCP Summary",
                disabled=not st.session_state['selected_hcp_npi'],
                on_click=lambda: st.session_state.update({'page': 'summary'}),
                key="go_to_summary_button"
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with st.spinner("Generating interactive network..."):
            filtered_nodes_for_display = get_filtered_nodes(
                all_hcps_details, top_n, selected_states, selected_cities,
                min_connections, min_influence, min_papers, min_panels, min_trials,
                sort_by
            )

            if filtered_nodes_for_display.empty:
                st.warning("No HCPs found based on the selected criteria.")
                net = Network(height="900px", width="100%", bgcolor="#0a0a0a", heading="")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
                    tmp_path = tmp_file.name
                    net.save_graph(tmp_path)
                with open(tmp_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=900, scrolling=False)
                os.unlink(tmp_path)
                return

            filtered_node_npis = filtered_nodes_for_display['NPI'].tolist()
            filtered_edges_for_display = get_filtered_edges_for_display(
                df, filtered_node_npis, min_strength, max_strength
            )
            
            net = create_pyvis_network(
                filtered_nodes_for_display,
                filtered_edges_for_display,
                enable_physics,
                global_min_connections, global_max_connections,
                global_min_edge_strength, global_max_edge_strength
            )

            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
                tmp_path = tmp_file.name
                net.save_graph(tmp_path)
            
            with open(tmp_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            if st.session_state.get('reset_view', False):
                html_content = html_content.replace(
                    'var network = new vis.Network(container, data, options);',
                    'var network = new vis.Network(container, data, options); network.fit();'
                )
                st.session_state['reset_view'] = False
                
            st.components.v1.html(html_content, height=900, scrolling=False)
            os.unlink(tmp_path)

        st.markdown("---")
        st.subheader("Network Summary")
        with st.expander("Expand to view key network metrics and top HCPs"):
            # Calculate new metrics
            total_connections = len(filtered_edges_for_display) if not filtered_edges_for_display.empty else 0
            
            most_common_metric = "N/A"
            if 'Metrics' in filtered_edges_for_display.columns and not filtered_edges_for_display['Metrics'].dropna().empty:
                metric_counts = filtered_edges_for_display['Metrics'].value_counts()
                if not metric_counts.empty:
                    most_common_metric = metric_counts.index[0]
                    most_common_metric_count = metric_counts.iloc[0]
                    most_common_metric = f"{most_common_metric} ({most_common_metric_count} connections)"
            
            most_common_state = "N/A"
            most_common_city = "N/A"
            if not filtered_nodes_for_display.empty:
                state_counts = filtered_nodes_for_display['state'].value_counts()
                city_counts = filtered_nodes_for_display['city'].value_counts()
                if not state_counts.empty:
                    most_common_state = f"{state_counts.index[0]} ({state_counts.iloc[0]} HCPs)"
                if not city_counts.empty:
                    most_common_city = f"{city_counts.index[0]} ({city_counts.iloc[0]} HCPs)"

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total No. of Connections", total_connections)
                st.metric("Most Common Connection Metric", most_common_metric)
            with col2:
                st.metric("Most Common State", most_common_state)
                st.metric("Most Common City", most_common_city)

            st.markdown("---")
            st.markdown("##### Most Connected HCPs")
            if not filtered_nodes_for_display.empty:
                for _, row in filtered_nodes_for_display.nlargest(5, 'connections').iterrows():
                    st.markdown(f"- **{row['hcp_name']}** (NPI: {row['NPI']}) - {row['connections']} connections")
            else:
                st.info("No connected HCPs.")
            st.markdown("##### Most Influential HCPs")
            if not filtered_nodes_for_display.empty:
                for _, row in filtered_nodes_for_display.nlargest(5, 'influence').iterrows():
                    st.markdown(f"- **{row['hcp_name']}** (NPI: {row['NPI']}) - {row['influence']:.2f} influence")
            else:
                st.info("No influential HCPs.")
            
            st.markdown("---")
            st.markdown("##### Detailed List of Displayed HCPs")
            if not filtered_nodes_for_display.empty:
                hcp_data_for_table = [{
                    'NPI': data['NPI'],
                    'HCP Name': data.get('hcp_name', 'N/A'),
                    'Connections': data.get('connections', 0),
                    'Influence Score': data.get('influence', 0),
                    'Location': f"{data.get('city', 'N/A')}, {data.get('state', 'N/A')}",
                    'Papers': data.get('papers', 0),
                    'Panels': data.get('panels', 0),
                    'Trials': data.get('trials', 0),
                } for _, data in filtered_nodes_for_display.iterrows()]
                st.dataframe(pd.DataFrame(hcp_data_for_table))
            else:
                st.info("No HCPs to display.")

    else:  # Summary page
        st.markdown('<div class="custom-title">HCP Summary</div>', unsafe_allow_html=True)
        if st.session_state['selected_hcp_npi'] is None:
            st.error("No HCP selected. Please select an HCP from the main page.")
        else:
            render_hcp_summary(st.session_state['summary_data'], st.session_state['selected_hcp_npi'])

if __name__ == "__main__":
    main()