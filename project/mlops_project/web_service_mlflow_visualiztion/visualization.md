# Create a docker network 
```bash 
docker network create monitoring
```
# Start prometheus on the port 9090 based on prometheus yml file on the monitoring network
```bash 
docker run -d --name=prometheus --network=monitoring -p 9090:9090 -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus
```

# Start grafana on the same docker network on the port 3000
```bash
docker run -d --name=grafana --network=monitoring -p 3000:3000 grafana/grafana
```

# Start flask app 
```bash 
docker run -d --name=flask_app --network=monitoring -p 9696:9696 flask_app
```