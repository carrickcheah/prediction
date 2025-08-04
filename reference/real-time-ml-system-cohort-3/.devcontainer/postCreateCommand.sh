#!/usr/bin/env bash
set -e
set -o noglob

# apt-extras
# apt-get update -y && apt-get install -y --no-install-recommends \
#     openssl ca-certificates \
#     pkg-config libssl-dev \
#     xgboost libclang-dev \
#     direnv fzf \
#     gettext \
#     jq \
#     build-essential \
#     wget

# direnv
direnv allow && echo '#' >> ~/.bashrc && echo 'eval "$(direnv hook bash)"' >> ~/.bashrc

# fzf
echo '#' >> ~/.bashrc
echo 'eval "$(fzf --bash)"' >> ~/.bashrc
