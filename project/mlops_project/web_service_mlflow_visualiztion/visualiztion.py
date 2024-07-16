# This script creates a prometheus pipeline to track latency and accuracy of predictions of web service application
# To further visualize it in Grafana. The script uses prometheus_client library to create a custom metrics and expose it to prometheus
# The flask application is created in predict.py and deployed using docker contrainer
