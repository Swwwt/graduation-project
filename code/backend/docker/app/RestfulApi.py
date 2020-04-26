from flask import Flask, request
from flask_restful import Resource, Api
import pandas as pd
import json
import re
import urllib.request
import requests
from bs4 import BeautifulSoup
import csv
import numpy as np
import time

from Services import Services

# @app.route('/data', methods=['PUT'])
# @app.route('/data/<name>', methods=['PUT'])
# @app.route('/robots', methods=['GET', 'POST', 'DELETE'])
# @app.route('/published_robots', methods=['GET', 'POST'])
# @app.route('/query', methods=['PUT'])
# @app.route('/api/query', methods=['PUT'])
# @app.route('/users', methods=['GET'])
# @app.route('/login', methods=['POST'])
# @app.route('/register', methods=['POST'])


app = Flask(__name__)
api = Api(app)
services = Services()
print("ok")


def _read_data_from_csv(filename):
    df = pd.read_csv(filename, names=["question", "answer"])
    return df


def _generate_data(request):
    data = []
    urls = []
    for f in request.files.getlist('file'):
        # f.save(secure_filename(f.filename))
        f.save(f.filename)
        if(f.filename.startswith('url')):
            urls.append(pd.read_csv(f.filename, names=["url"]))
        else:
            data.append(_read_data_from_csv(f.filename))

    if data:
        frames = pd.concat(data, ignore_index=True)
        print("frames", frames)

        data = []
        for i, row in frames.iterrows():
            print(i, row.question)
            data.append({"question": row.question, "answer": row.answer})
        print("data", data)

    if urls:
        frames = pd.concat(urls, ignore_index=True)
        print("frames", frames)

        pattern = re.compile(r'<.*?>')

        for i, row in frames.iterrows():
            print("url",  row.url)
            webUrl = urllib.request.urlopen(row.url)
            ht = webUrl.read()
            soup = BeautifulSoup(ht, 'html.parser')
            question = soup.find_all("a", class_="sf-accordion__link")
            answer = soup.find_all("p", class_="sf-accordion__summary")
            for r in range(0, len(answer)):
                q = question[r].string.replace(
                    '\n', '').replace('\r', '').strip()
                a = pattern.sub('', answer[r].text).replace(
                    '\xa0', ' ').replace('\n', '').replace('\r', '').strip()
                if len(a) > 100:
                    a = a.partition('.')[0] + '.'
                data.append({"question": q, "answer": a})

    print("data", data)
    return data


def _merge(data1, data2):
    d = []
    for i in data1:
        d.append((i["question"], i["answer"]))
    for i in data2:
        d.append((i["question"], i["answer"]))
    d = list(set(d))
    data = []
    for i, r in d:
        data.append({"question": i, "answer": r})
    return data


@app.route('/data', methods=['PUT'])
def generate_data():
    return json.dumps(_generate_data(request))


@app.route('/data/<name>', methods=['PUT'])
def generate_data_and_clone(name):
    data1 = _generate_data(request)
    data2 = services.data_get(name)
    print("data1", data1)
    print("data2", data2)
    data = _merge(data1, data2)
    return json.dumps(data)


@app.route('/robots', methods=['GET', 'POST', 'DELETE'])
def robots():
    user = request.args.get('user')
    name = request.args.get('name')
    if request.method == 'GET':
        if user:
            return services.view_robots_by_user(user)
        else:
            return json.dumps(services.get_robot(name))
    elif request.method == 'POST':
        info = request.get_json()
        services.post_robot(info)
        print('post robot success~')
        return 'post robot success~'
    else:
        return services.delete_robot(name)


@app.route('/query', methods=['PUT'])
def query():
    _json = request.get_json()
    print('question', _json["question"])
    print('search sort success~')
    result = services.search_sort(_json["name"], _json["question"])
    print("result", result)
    return json.dumps(result)


@app.route('/published_robots', methods=['GET', 'POST'])
def publish_robot():
    if request.method == 'GET':
        return services.view_published_robots()
    else:
        info = request.get_json()
        return services.publish_robot(info)


@app.route('/api/query', methods=['POST'])
def api_query():
    _json = request.get_json()
    print('question', _json["question"])
    print('search data success~')
    result = services.search(_json["name"], _json["question"])
    print(result[0])
    return json.dumps(result[0]["answer"])


@app.route('/users', methods=['GET'])
def get_users():
    return services.get_users()


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    return json.dumps(services.login(data["username"], data["password"]))


@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    return json.dumps(services.register(data))


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
