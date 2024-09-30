# Use debian trixie for gcc13
FROM debian:trixie as builder

ARG MODEL_URL
ARG MODEL_FILE

# Set work directory
WORKDIR /download

# Update and install dependencies
RUN mkdir out && \
    apt-get update && \
    apt-get install -y curl git gcc make python3-pip

RUN curl -L -o ./unzip https://cosmo.zip/pub/cosmos/bin/unzip && \
    chmod 755 unzip && mv unzip /usr/local/bin


# Install Huggingface hub and download model file
RUN mkdir -p /download/model
RUN pip3 install huggingface_hub[cli] --break-system-packages
RUN huggingface-cli download $MODEL_URL $MODEL_FILE --local-dir /download/model


# Checkout llamafile codebase
RUN git clone https://github.com/Mozilla-Ocho/llamafile.git

# Build llamafile
RUN cd llamafile && make -j8 LLAMA_DISABLE_LOGS=1; exit 0
RUN cd llamafile && make install PREFIX=/download/out; exit 0


# Create container
FROM debian:stable as out


# Create a non-root user
RUN addgroup --gid 1000 user && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos "" user

# Switch to user
USER user

# Set working directory
WORKDIR /usr/local

# Copy llamafile and man pages
COPY --from=builder /download/out/bin ./bin
COPY --from=builder /download/model/*.gguf /model


# Expose 8080 port.
EXPOSE 8080

# Set entrypoint.
ENTRYPOINT ["/bin/sh", "/usr/local/bin/llamafile"]

# Set default command.
CMD ["--server", "--nobrowser", "--host", "0.0.0.0", "-m", "/model"]