from flask import Flask, render_template, request
from data_preprocessing import prepare_graph_data

app = Flask(__name__)

@app.route("/")
def home():
    graph = prepare_graph_data("./outputs/16_06_2023_20_49_47/graph_communities.json")
    return render_template('graph.html', graph = graph)

if __name__ == "__main__":
    app.run(debug=True)