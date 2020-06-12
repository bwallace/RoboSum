from flask import Flask, jsonify, request
import json 


from robot_sum import RoboSummarizer
print("instantiating summarizer...")
robo_sum = RoboSummarizer()
print("ok!")

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello World!'


@app.route('/summarize', methods=['POST'])
def summarize():
    # Note: this is probably terrible?
    studies = json.loads(request.json)['articles']
    summary = robo_sum.summarize(studies)
    print(summary)
    return jsonify(summary)