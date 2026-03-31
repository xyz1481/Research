import numpy as np, json, pickle, pandas as pd

node_feats = np.load('phase2_outputs/graph_node_features.npy')
edge_index  = np.load('phase2_outputs/graph_edge_index.npy')
edge_attr   = np.load('phase2_outputs/graph_edge_attr.npy')
node_idx    = json.load(open('phase2_outputs/graph_node_index.json'))
with open('phase2_outputs/supply_chain_graph.pkl','rb') as f:
    G = pickle.load(f)

print('Node features shape:', node_feats.shape)
print('Edge index shape   :', edge_index.shape)
print('Edge attr shape    :', edge_attr.shape)
print('Nodes:', list(node_idx.keys()))
print('Edges:')
for u,v,d in G.edges(data=True):
    print(f'  {u} -> {v}  count={d["weight"]:.0f}  disruption_rate={d["disruption_rate"]:.3f}')

df = pd.read_parquet('phase2_outputs/df_phase2_enriched.parquet')
city_stats = df.groupby('Origin_City').agg(
    avg_delay       = ('delay',                  'mean'),
    disruption_rate = ('disruption',             'mean'),
    total_shipments = ('Order_ID',               'count'),
    avg_geo_risk    = ('Geopolitical_Risk_Index', 'mean'),
    avg_weather     = ('Weather_Severity_Index',  'mean'),
    avg_cost        = ('Shipping_Cost_USD',        'mean'),
).reset_index()
print('\nCity-level stats:')
print(city_stats.to_string())

route_stats = df.groupby('Route_Type').agg(
    avg_delay       = ('delay',                  'mean'),
    disruption_rate = ('disruption',             'mean'),
    total_shipments = ('Order_ID',               'count'),
    avg_geo_risk    = ('Geopolitical_Risk_Index', 'mean'),
    avg_weather     = ('Weather_Severity_Index',  'mean'),
).reset_index()
print('\nRoute-level stats:')
print(route_stats.to_string())
