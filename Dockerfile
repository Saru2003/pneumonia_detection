FROM 763104351884.dkr.ecr.eu-north-1.amazonaws.com/tensorflow-inference:latest

COPY inference.py /opt/ml/model/
COPY pneumonia_detection_saved_model/ /opt/ml/model/pneumonia_detection_saved_model

ENV SAGEMAKER_PROGRAM=inference.py
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/model/
