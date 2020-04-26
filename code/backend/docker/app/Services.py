from elasticsearch import Elasticsearch
import json
import csv
import numpy as np
import time

from FaissService import FaissService
from BertService import BertService
from RobertaService import RobertaService

INF = 1000

SEP = '-'
INFO_DB = 'robot_info'


class Services():
    def __init__(self):
        self._es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
        #self._es = Elasticsearch([{'host': 'server', 'port': 9200}])
        db = ['published', 'user_db']
        for i in db:
            if not self._es.indices.exists(index=i):
                self._es.indices.create(index=i)
                self._es.indices.put_settings(index=i, body={
                    "index.blocks.read_only_allow_delete": "false"})
        print("##es init success")

        self._faiss = FaissService()
        self._bert = BertService()
        self._roberta = RobertaService()
        print("##Services init success")

    def data_post(self, data, name):
        [raw_name, version] = name.split("_")

        if self._es.indices.exists(index=name):
            self._es.indices.delete(name)
            self._es.indices.create(index=name)

        for i, row in enumerate(data):
            self._es.index(index=name, doc_type='qna', id=i, body=row)

        v = []
        exist = self._es.exists(index="robot_info", id=raw_name)
        if exist:
            res = self._es.get(index='robot_info',
                               doc_type='info', id=raw_name)
            v = res["_source"]["version"]
            if(version not in v):
                v = v + [version]
                self._es.update(index="robot_info", doc_type="info",
                                id=raw_name, body={"doc": {"version": v}}, refresh='wait_for')
        else:
            v = [version]
            info = json.dumps(
                {"name": raw_name, "version": v})
            self._es.index(index="robot_info", id=raw_name,
                           doc_type='info', body=info)

    def data_get(self, name):
        res = self._es.search(index=name, doc_type="qna", body={
            'query': {
                'match_all': {}
            }
        }, size=INF)

        data = []
        for doc in res['hits']['hits']:
            data.append(doc['_source'])
        return data

    def delete_robot(self, robot_name):
        [user, name, version] = robot_name.split("-")
        version = int(version)
        if version:
            robot_db = user + SEP + name + SEP + str(version)
            info_db = user + SEP + INFO_DB

            res = self._es.get(index=info_db, doc_type='info', id=name)
            v = res["_source"]["version"]

            v.remove(version)

            if(v):
                self._es.update(index=info_db, doc_type="info", id=name, body={
                    "doc": {"version": v}}, refresh='wait_for')
            else:
                res = self._es.delete(index=info_db,
                                      doc_type='info', id=name, refresh='wait_for')

            res = self._es.indices.delete(robot_name)
            return "delete success~"
        else:
            info_db = user + SEP + INFO_DB
            res = self._es.get(index=info_db, doc_type='info', id=name)
            version = res["_source"]["version"]
            res = self._es.delete(index=info_db,
                                  doc_type='info', id=name, refresh='wait_for')

            for v in version:
                robot_db = user + SEP + name + SEP + str(v)
                self._es.indices.delete(robot_db)
            return "delete all~"

    def _faiss_build(self, info):
        questions = [i['question'] for i in info]
        answers = [i['answer'] for i in info]
        self._faiss.build(questions, answers)

    def _faiss_search(self, question):
        return self._faiss.search(question)

    def _search(self, name, question):
        res = self._es.search(index=name, doc_type="qna", body={
            'query': {
                'match': {'question': question}
            }
        }, size=5)  # !!!
        print("*****", name, question, res)

        ans1 = []
        for doc in res['hits']['hits']:
            print(doc['_id'], doc['_score'], doc['_source'])
            ans1.append({"id": doc['_id'],
                         "question": doc['_source']['question'],
                         "answer": doc['_source']['answer'],
                         "score": doc['_score']})
        print("ans1", ans1)
        # -----------------------------------------------------
        res = self._es.search(index=name, doc_type="qna", body={
            'query': {
                'match_all': {}
            }
        }, size=INF)
        info = [i['_source'] for i in res['hits']['hits']]
        self._faiss_build(info)

        ans2 = self._faiss_search(question)
        print("ans2", ans2)

        d_ans1 = {i["id"]: i for i in ans1}
        d_ans2 = {i["id"]: i for i in ans2}

        ids = set()
        for a in d_ans1.keys():
            ids.add(a)
        for a in d_ans2.keys():
            ids.add(a)

        result = []
        for id in ids:
            result.append({"id": id,
                           "question": d_ans1[id]["question"] if id in d_ans1 else d_ans2[id]["question"] if id in d_ans2 else None,
                           "answer": d_ans1[id]["answer"] if id in d_ans1 else d_ans2[id]["answer"],
                           "score": d_ans1[id]["score"] if id in d_ans1 else None,
                           "distance": float(d_ans2[id]["distance"]) if id in d_ans2 else None})
        print(result)
        '''
        result = [i for i in result if i['score']
                  is not None and i["distance"] is not None]
        scores = np.array([i['score'] for i in result])
        distances = np.array([i['distance'] for i in result])

        answer = []
        for i in result:
            x = (float(i['score']) - scores.mean()) / scores.std()
            y = (float(i['distance']) - distances.mean()) / distances.std()
            z = x-y
            i['confidence'] = z
            answer.append(i)

        answer.sort(key=lambda x: -x["confidence"])

        print("##final answer", answer)
        self._print_answer(answer)
        '''
        return result

    def search_sort(self, name, question):
        s_time = time.time()
        result = self._search(name, question)
        answers = [{"id": i['id'], "answer": i['answer']} for i in result]
        filter_time = time.time() - s_time

        '''
        s_time = time.time()
        url = "http://localhost:5003/ml/roberta/sort"
        payload = json.dumps({"question": question, "answers": answers})
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        roberta_sorted = json.loads(response.text)
        roberta_time = time.time() - s_time
        '''

        s_time = time.time()
        info = {"question": question, "answers": answers}
        bert_sorted = self._bert.sort_answer(info)
        bert_time = time.time() - s_time

        s_time = time.time()
        info = {"question": question, "answers": bert_sorted[:3]}
        roberta_sorted = self._roberta.sort_answer(info)
        roberta_time = time.time() - s_time

        summary = {"result": result, "filter_time": filter_time,
                   "bert_sorted": bert_sorted, "bert_time": bert_time,
                   "roberta_sorted": roberta_sorted, "roberta_time": roberta_time}
        # "roberta_sorted": roberta_sorted, "roberta_time": roberta_time}
        return summary

    def _print_answer(self, answer):
        with open('./search_answer.csv', 'w', newline='') as f:
            fieldnames = list(answer[0].keys())
            print(fieldnames)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in answer:
                writer.writerow(row)

    def publish_robot(self, info):
        (user, name, version, description) = list(info.values())
        index = name + '-' + str(version)
        print("info", info)

        self._es.index(index="published", id=index,
                       doc_type='published', body=info)
        return "publish robot~"

    def view_published_robots(self):
        res = self._es.search(index="published", doc_type="published", body={
            'query': {
                'match_all': {}
            }
        }, size=INF)
        data = []
        for doc in res['hits']['hits']:
            print(doc['_source'], type(doc['_source']))
            data.append(doc['_source'])
        print(json.dumps(data))
        return json.dumps(data)

    def login(self, username, password):
        if self._es.exists(index='user_db', doc_type='user', id=username):
            res = self._es.get(index='user_db', doc_type='user', id=username)
            real_password = res["_source"]["password"]
            print(real_password)
            if password == real_password:
                return {"status": True, "message": "log in successfully"}
            else:
                return {"status": False, "message": "wrong password"}
        else:
            return {"status": False, "message": "no such user"}

    def register(self, info):
        username = info["username"]
        password = info["password"]

        if self._es.exists(index='user_db', doc_type='user', id=username):
            return {"status": False, "message": "user already exist"}
        else:
            self._es.index(index="user_db", id=username,
                           doc_type='user', body=info)
            return {"status": True, "message": "register successfully"}

    def get_users(self):
        res = self._es.search(index="user_db", doc_type="user", body={
            'query': {
                'match_all': {}
            }
        }, size=INF)
        data = []
        for doc in res['hits']['hits']:
            print(doc['_id'])
            data.append(doc['_id'])
        print(json.dumps(data))
        return (json.dumps(data))

    def post_robot(self, info):
        (user, name, version, description, data) = list(info.values())
        print("user", user)
        print("SEP", SEP)
        robot_db = user + SEP + name + SEP + str(version)
        info_db = user + SEP + INFO_DB

        print("info", info, type(info))
        print("robot_db", robot_db)

        if self._es.indices.exists(index=robot_db):
            self._es.indices.delete(robot_db)

        self._es.indices.create(index=robot_db)
        self._es.indices.put_settings(index=robot_db, body={
            "index.blocks.read_only_allow_delete": "false"})

        for i, row in enumerate(data):
            print(row)
            self._es.index(index=robot_db, doc_type='qna', id=i, body=row)

        v = []
        exist = self._es.exists(index=info_db, id=name)
        if exist:
            res = self._es.get(index=info_db, doc_type='info', id=name)
            v = res["_source"]["version"]
            if(version not in v):
                v = [version] + v
                print("!! versions", v)
                self._es.update(index=info_db, doc_type="info",
                                id=name, body={"doc": {"version": v, "description": description}}, refresh='wait_for')
        else:
            v = [version]
            self._es.index(index=info_db, id=name,
                           doc_type='info', body={"name": name, "version": v, "description": description})

    def get_robot(self, name):
        res = self._es.search(index=name, doc_type="qna", body={
            'query': {
                'match_all': {}
            }
        }, size=INF)

        data = []
        for doc in res['hits']['hits']:
           # print(doc['_id'], doc['_source'])
            data.append(doc['_source'])
        print(json.dumps(data))
        return data

    def view_robots_by_user(self, user):
        info_db = user + SEP + INFO_DB

        if not self._es.indices.exists(index=info_db):
            self._es.indices.create(index=info_db)
            self._es.indices.put_settings(index=info_db, body={
                "index.blocks.read_only_allow_delete": "false"})
            return (json.dumps([]))

        res = self._es.search(index=info_db, doc_type="info", body={
            'query': {
                'match_all': {}
            }
        }, size=INF)
        data = []
        for doc in res['hits']['hits']:
            # print(doc['_source'])
            data.append(doc['_source'])
        print(json.dumps(data))
        return (json.dumps(data))
