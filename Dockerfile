# Dockerfile
#
# -> Docker image untuk serving model student-performance
#      -> base: python:3.12-slim (hindari bug pip di MLflow build-docker)
#      -> install: mlflow + scikit-learn + pandas + numpy
#      -> copy model artifact dari mlruns/ (dicopy saat docker build di CI)
#      -> serve via: mlflow models serve
# -> dipanggil oleh: docker build --build-arg RUN_ID=xxx .

FROM python:3.12-slim

WORKDIR /app

# install semua dependencies model serving
RUN pip install --no-cache-dir \
    mlflow==2.19.0 \
    scikit-learn==1.4.2 \
    pandas==2.2.2 \
    numpy==1.26.4

# copy model artifact (diisi saat CI step "Prepare model artifact")
COPY model/ /app/model/

EXPOSE 8080

# serve model via mlflow models serve - no-conda karena deps sudah di-install
CMD ["mlflow", "models", "serve", \
     "-m", "/app/model", \
     "--no-conda", \
     "--host", "0.0.0.0", \
     "--port", "8080"]
