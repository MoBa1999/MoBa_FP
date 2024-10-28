# syntax=docker/dockerfile:1

FROM pytorch/pytorch
RUN apt-get update && apt-get install -y git screen
WORKDIR /app
SHELL ["/bin/bash", "--login", "-c"]
    
# Create the environment
COPY env.yml . 
RUN conda env create -f env.yml
CMD ["/bin/bash"]
