import plotly.express as px
import pandas as pd

features = False

if features:
    merged_data = pd.read_pickle("/home/svsosafiscella/PycharmProjects/highfluencetiming/results/features.pkl")

    fig = px.scatter_3d(merged_data,
                        x='Width', y='Energy', z='Amp', template='plotly')

else:

    clustered_file = "./results/5_kmeans_clusters.pkl"
    clustered_data = pd.read_pickle(clustered_file)

    cluster_4 = clustered_data.index[clustered_data['Cluster'] == 4].tolist()

    fig = px.scatter_3d(clustered_data, x='Width', y='Energy', z='Amp',
                        color='Cluster', template='plotly')

    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.8,
        xanchor="right",
        x=0.75,
        font=dict(
#            family="Calibri",
            size=14,
            color="black"
        ),
        bgcolor="white",
        bordercolor="White",
        borderwidth=1
    ))

fig.update_traces(marker_size=4)
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
fig.update_layout(
        #    font_family="Courier New",
        font_color="black",
        font_size=12,
        #    title_font_family="Times New Roman",
        title_font_color="black",
        legend_title_font_color="black")

name = 'eye = (x:1.5, y:1.5, z:0.375)'
camera = dict(
        eye=dict(x=-1.5, y=-1.5, z=0.375))
fig.update_layout(scene_camera=camera)

if features:
    fig.write_image("/home/svsosafiscella/PycharmProjects/highfluencetiming/figures/features_3d.pdf")
else:
    fig.write_image("/home/svsosafiscella/PycharmProjects/highfluencetiming/figures/clustered_3d.pdf")

fig.show()