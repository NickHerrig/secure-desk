FROM roboflow/roboflow-inference-server-jetson-5.1.1

WORKDIR /app

ADD . /app

RUN pip install -r requirements.txt

CMD ["python", "main.py"]