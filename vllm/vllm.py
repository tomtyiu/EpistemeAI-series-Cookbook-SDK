# Install vLLM from pip:
# pip install vllm

# Load and run the model:
vllm serve "EpistemeAI2/Fireball-Alpaca-Llama-3.1-8B-Instruct-KTO-beta"

# Call the server using curl:
curl -X POST "http://localhost:8000/v1/chat/completions" \ 
	-H "Content-Type: application/json" \ 
	--data '{
		"model": "EpistemeAI2/Fireball-Alpaca-Llama-3.1-8B-Instruct-KTO-beta"
		"messages": [
			{"role": "user", "content": "Hello!"}
		]
	}'
#Docker image

# Deploy with docker on Linux:
docker run --runtime nvidia --gpus all \
	--name my_vllm_container \
	-v ~/.cache/huggingface:/root/.cache/huggingface \
 	--env "HUGGING_FACE_HUB_TOKEN=<secret>" \
	-p 8000:8000 \
	--ipc=host \
	vllm/vllm-openai:latest \
	--model EpistemeAI2/Fireball-Alpaca-Llama-3.1-8B-Instruct-KTO-beta

Copy
# Load and run the model:
docker exec -it my_vllm_container bash -c "vllm serve EpistemeAI2/Fireball-Alpaca-Llama-3.1-8B-Instruct-KTO-beta"

# Call the server using curl:
curl -X POST "http://localhost:8000/v1/chat/completions" \ 
	-H "Content-Type: application/json" \ 
	--data '{
		"model": "EpistemeAI2/Fireball-Alpaca-Llama-3.1-8B-Instruct-KTO-beta"
		"messages": [
			{"role": "user", "content": "Hello!"}
		]
	}'
