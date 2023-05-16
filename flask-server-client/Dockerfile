FROM python:3.10.11-slim

WORKDIR /flask-docker

# RUN apt-get update && apt-get install -y python3-opencv

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 5000

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]