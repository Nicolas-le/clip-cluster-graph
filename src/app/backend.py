from flask import Flask, render_template, request
from data_preprocessing import prepare_graph_data

app = Flask(__name__)

@app.route("/")
def home():
    #graph = prepare_graph_data("./outputs/17_06_2023_11_28_35/graph_communities.json")
    graph = prepare_graph_data("./outputs/21_06_2023_12_01_27/graph_communities.json")

    return render_template('graph.html', graph = graph)

if __name__ == "__main__":
    app.run(debug=True)