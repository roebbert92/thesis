# Elasticsearch
```bash
docker run \
    --restart unless-stopped -d \
    -p 9200:9200 \
    -e "discovery.type=single-node" \
    -e "ES_JAVA_OPTS=-Xmx11g -server -XX:ActiveProcessorCount=4" \
    -e "index.store.type: mmapfs" \
    -e "indices.memory.index_buffer_size: 30%" \
    -e "index.translog.flush_threshold_ops: 50000" \
    --cpus=4 \
    --cpu-shares=4096 \
    --memory=12g \
    elasticsearch:7.9.2
```