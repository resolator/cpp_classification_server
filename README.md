# cpp_classification_server
C++ server which serves an ONNX classification model.

## Installation
To install the server build the docker image using the following command:
```sh
docker build -t cpp_server ./docker/
```

To start the server run the docker container using the following command:
```sh
docker run -d --name cpp_server --gpus 0 -p 80:80 cpp_server
```

## Example usage
The server accepts POST requests with images and responses with a result of classification.
To try the server run the following command:
```sh
curl -X POST -H "Content-Type: multipart/form-data" -F "data=@data/bee_eater.jpg" 127.0.0.1/classify
```

The response will be:
```json
{"label_id": 92, "label": "n01828970 bee eater"}
```