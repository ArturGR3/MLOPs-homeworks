#!/bin/bash
# Run the integration tests for hw6

# Change to the directory of this script
cd "$(dirname "$0")"

# bring in the environment variables using .env file in the parent directory
source .env

# Getting localstack up
docker-compose up -d 

# Wait for a while to let the container initialize
sleep 10

container_name=$(docker-compose ps | grep localstack | awk '{print $1}')

# Getting localstack up
docker-compose up -d 
for t in {1..10};
    do echo checking service up 
    healthy=$(docker inspect -f '{{ .State.Health.Status }}' $container_name)
    echo $healthy
    if [[ $healthy == 'healthy' ]]
    then 
        echo service is up
        break
    fi 
    sleep $t 
    echo sleeping for $t seconds
done

# creating bucket    
aws --endpoint-url ${S3_ENDPOINT_URL} s3 mb s3://${BUCKET_NAME}

# Run the integration tests
python integration_test.py

RESULT=$?

if [ $RESULT -eq 0 ]; then
  echo 'Integration tests ------------- passed successfully----------'
else
  docker-compose logs
  echo 'Integration tests ------------- failed----------------------'
fi

docker-compose down