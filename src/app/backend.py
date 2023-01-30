from flask import Flask, render_template, request
from data_preprocessing import prepare_graph_data

app = Flask(__name__)

@app.route("/")
def home():
    graph = prepare_graph_data("./outputs/26_01_2023_12_56_19/graph.json")
    #print(graph)
    return render_template('graph.html', graph = graph)

if __name__ == "__main__":
    app.run(debug=True)