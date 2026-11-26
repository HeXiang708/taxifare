run-api:
	uv run uvicorn api.fast:app --reload

docker_image_name:
	echo ${GAR_IMAGE_NAME}

# Build the docker in local machine
docker_build_local:
	docker build -t ${IMAGE_NAME}:dev .
# run the docker in local machine
docker_run:
	docker run -p 8001:8080 ${IMAGE_NAME}:dev

build_test_local:
	${MAKE} docker_build_local
	${MAKE} docker_run;
