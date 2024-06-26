### Create AWS s3 bucket in localhost (localstack) 
```bash
aws --endpoint-url=http://localhost:4566 s3 mb s3://nyc-duration
```

### List this bucket 
```bash 
aws --endpoint-url=http://localhost:4566 s3 ls
``` 

### List the files in the bucket 
```bash
aws --endpoint-url=http://localhost:4566 s3 ls s3://nyc-duration/in/
```