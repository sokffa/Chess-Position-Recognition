FROM python:3.7-slim-stretch

RUN apt-get update && apt-get install -y git python3-dev gcc \
    && rm -rf /var/lib/apt/lists/*  && apt-get -y install libglib2.0; apt-get clean


COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt

COPY app app/

RUN python app/server.py

EXPOSE 5000

CMD ["python", "app/server.py", "serve"]
