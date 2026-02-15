
FROM python:3.10

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

COPY model/ model/

EXPOSE 7860

CMD ["python", "app.py"]
