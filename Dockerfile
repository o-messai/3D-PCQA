FROM continuumio/miniconda3

RUN apt-get update && apt-get install -y curl

ENV PATH /opt/conda/bin:$PATH

RUN conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

WORKDIR /app

COPY requirements.txt .

ADD test.py .
ADD utils.py .
ADD model.py .
ADD data_load.py .
ADD config.yaml .

RUN pip install -r requirements.txt

CMD ["echo ", "Make sure everything is installed:!"]
