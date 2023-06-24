from flask import Flask, render_template, request
from data_preprocessing import prepare_graph_data, load_captions

app = Flask(__name__)

@app.route("/")
def home():
    #graph = prepare_graph_data("./outputs/21_06_2023_12_01_27/graph_communities.json") # manuel
    graph = prepare_graph_data("./outputs/23_06_2023_15_44_38/graph_communities.json")
    captions = load_captions("./src/app/static/image_captions.json")

    return render_template('graph.html', graph = graph, captions = captions)

if __name__ == "__main__":
    app.run(debug=True)