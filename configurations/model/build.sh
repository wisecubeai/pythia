docker build -t pythia-llamafile \
	--build-arg MODEL_URL=TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
	--build-arg MODEL_FILE=tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf .