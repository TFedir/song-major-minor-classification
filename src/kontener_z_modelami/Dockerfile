FROM python:3.11.3

COPY flask_service.py ./
COPY models ./models
COPY ./requirements.txt ./
RUN pip install -r requirements.txt

EXPOSE 9999
CMD [ "sh", "-c", "python flask_service.py"]

