# syntax=docker/dockerfile:1

FROM serversideup/github-cli:alpine-2.92.0

RUN <<EOF_RUN

set -ex

apk add coreutils bash yq

# Install the latest `nv-gha-aws` GitHub CLI plugin
gh extension install nv-gha-runners/gh-nv-gha-aws

mv ~/.local /local
chmod -R 0777 /local

EOF_RUN

COPY generate.sh /generate.sh
