FROM python:3.7

ADD app /app

RUN pip3 install torch -i https://mirrors.aliyun.com/pypi/simple/

RUN pip3 install -r ./app/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
RUN pip3 install  ./app/transformers/.

WORKDIR /app

EXPOSE 5000
