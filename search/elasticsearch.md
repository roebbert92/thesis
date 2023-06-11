# Elasticsearch
```bash
docker run \
    --restart unless-stopped -d \
    -p 9200:9200 \
    -e "discovery.type=single-node" \
    -e "ES_JAVA_OPTS=-Xmx7g -Xms5g -server -XX:ActiveProcessorCount=16" \
    -e "index.store.type: mmapfs" \
    -e "indices.memory.index_buffer_size: 30%" \
    -e "index.translog.flush_threshold_ops: 50000" \
    -v /home/loebbert/elasticsearch_data:/usr/share/elasticsearch/data \
    --cpus=16 \
    --memory=12g \
    elasticsearch:7.9.2
```