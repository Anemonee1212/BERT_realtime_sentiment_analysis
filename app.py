from flask import Flask, config, render_template, request
from matplotlib.pyplot import xlabel
import pandas as pd
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go

app = Flask(__name__)


@app.route('/callback', methods = ['POST', 'GET'])
def cb():
    return gm(request.args.get('data'))


@app.route('/')
def index():
    return render_template('project_demo.html', graphJSON = gm())


def gm(country = 'Argentina'):
    df = pd.read_csv("output/viz_data.csv")

    fig = px.line(
        df[df['country'] == country], x = "hour12", y = "count", color = "sentiment",
        color_discrete_sequence = ['green', 'red'], markers = True, title = country, labels = {'hour12': 'date'}
        )
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = [0, 2, 4, 6, 8, 10],
            ticktext = ['Dec 15', 'Dec 16', 'Dec 17', 'Dec 18', 'Dec 19', 'Dec 20']
        )
    )
    fig.update_layout(
        {
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        }
    )

    graphJSON = json.dumps(fig, cls = plotly.utils.PlotlyJSONEncoder)
    print(fig.data[0])

    return graphJSON
