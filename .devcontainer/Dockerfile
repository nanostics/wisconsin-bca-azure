FROM mcr.microsoft.com/devcontainers/anaconda:0-3

COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
# https://www.sckaiser.com/blog/2023/01/30/conda-codespaces.html
# make a new environment instead of creating one
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env create -n env -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>