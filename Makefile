run-api:
	uv run uvicorn api.fast:app --reload

docker_image_name:
	echo ${GAR_IMAGE_NAME}

#
#		Command of the local deployement
#
# Build the docker in local machine
docker_build_local:
	docker build -t ${IMAGE_NAME}:dev .
# run the docker in local machine
docker_run:
	docker run --env-file .env -p 8001:8080 ${IMAGE_NAME}:dev

build_test_local:
	${MAKE} docker_build_local
	${MAKE} docker_run;

#
#		Command of the Clould deployement
#
# Cloud Run authentification
auth_docker:
	gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev
# Create a new repository in Cloud Run
create_docker_repo:
	gcloud artifacts repositories create ${DOCKER_REPO_NAME} \
		--repository-format=docker \
		--location=${GCP_REGION} \
		--description="Repository for storing Docker images"
# Build the docker in Cloud Run
docker_build_cloud:
	docker build --platform linux/amd64 -t ${GAR_IMAGE_NAME}:prod .
# Push the docker to Cloud Run
docker_push:
	docker push ${GAR_IMAGE_NAME}:prod
# Deploy the docker
docker_deploy:
	gcloud run deploy --image ${GAR_IMAGE_NAME}:prod --region ${GCP_REGION} --env-vars-file production.env

build_deploy_cloud:
	${MAKE} docker_build_cloud
	${MAKE} docker_push
	${MAKE} docker_deploy

#
# Load the model from cloud storage
#
# 1. Create a bucket
# 2. Copy the model from local to cloud: gcloud storage cp models/mymodel.keras gs://${BUCKET_NAME}
# 3. In Dockerfile: Don't copy model, but just create a folder called models.
# 4. Update .env, production.env and predict to adapt a loading from cloud storage
create_gcp_bucket:
	gcloud storage buckets create --location ${GCP_REGION} gs://${BUCKET_NAME}
	gcloud storage ls
