from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from prediction import preprocess, reverse_normalize
app = Flask(__name__)

predictionProcess = preprocess()
model = predictionProcess.modeling()
graph = tf.get_default_graph()

@app.route("/", methods=["POST", "GET"])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == "POST":
        news = request.form.get('news')
        global graph
        with graph.as_default():
            score = stockPredict(news)
        kwargs = {'score': score}
        return render_template('index.html', **kwargs)


def stockPredict(news):
    data = predictionProcess.sequenceCheck(news)
    preprocessedData = np.array(data).reshape((1, -1))
    predicted = model.predict([preprocessedData, preprocessedData])
    answer = reverse_normalize(predicted)
    return np.round(answer, 2)


if __name__ == '__main__':
    app.run()
