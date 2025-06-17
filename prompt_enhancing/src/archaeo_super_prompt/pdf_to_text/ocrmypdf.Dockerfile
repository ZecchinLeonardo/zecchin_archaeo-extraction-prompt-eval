FROM docker.io/jbarlow83/ocrmypdf:latest

# add Italian
RUN apt update && \
    apt install -y tesseract-ocr-ita parallel

ENTRYPOINT ["parallel"]
