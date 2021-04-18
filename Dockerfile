FROM ubuntu 
RUN apt-get update 

FROM python:3.6
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
CMD [“echo”,”Image created”] 