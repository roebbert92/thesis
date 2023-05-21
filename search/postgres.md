# Use postgres docker

```bash
docker run --name postgres -p 5432:5432 -e POSTGRES_PASSWORD=thesis123. -d --restart unless-stopped postgres
```
```bash
docker exec -it postgres psql -U postgres -c "CREATE DATABASE fetahugaz;"
```

## Install pgadmin4

```bash
docker run --name pgadmin -e "PGADMIN_DEFAULT_EMAIL=admin@thesis.com" -e "PGADMIN_DEFAULT_PASSWORD=admin" -p 5050:80 -d --restart unless-stopped dpage/pgadmin4 
```

## Connect both
https://dev.to/steadylearner/how-to-set-up-postgresql-and-pgadmin-with-docker-51h