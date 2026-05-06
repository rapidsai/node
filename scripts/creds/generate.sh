#!/usr/bin/env bash

set -Eeuo pipefail

ln -s /local ~/.local

declare -a args=(--hostname github.com)

gh auth status -a "${args[@]}" || :

if gh auth status -a "${args[@]}" &>/dev/null \
&& test -f /gh/config.yml \
&& test -f /gh/hosts.yml; then
    cp -ar /gh/*.yml /config/gh/
    if ! grep -q 'oauth_token:' /config/gh/hosts.yml; then
        yq -i ".*.oauth_token = \"$(gh auth token)\" | .*.users.*.oauth_token = \"$(gh auth token)\"" /config/gh/hosts.yml
    fi
    unset GH_TOKEN GITHUB_TOKEN
fi

declare -a wanted_scopes=(read:org read:enterprise)
declare -a active_scopes="($(
    gh api -i -X GET --silent rate_limit  \
    2>/dev/null                           \
  | grep -i 'x-oauth-scopes:'             \
  | cut -d' ' -f1 --complement            \
  | tr -d ','                             \
  | tr '\r' '\n'                          \
  | tr '\n' ' '                           \
  | tr -s '[:blank:]'                     \
))"

declare -a needed_scopes="($(
  comm -23                                             \
    <(IFS=$'\n'; echo "${wanted_scopes[*]}" | sort -s) \
    <(IFS=$'\n'; echo "${active_scopes[*]}" | sort -s) \
))"

wanted_scopes=("${active_scopes[@]}" "${wanted_scopes[@]}")
read -ra wanted_scopes <<< "${wanted_scopes[*]/#/--scopes }"

if ! gh auth status -a "${args[@]}" &>/dev/null; then
    # Login with additional scopes
    unset GH_TOKEN GITHUB_TOKEN
    args+=("${wanted_scopes[@]}")
    args+=(--insecure-storage)
    args+=(--skip-ssh-key)
    args+=(--git-protocol)
    args+=("$(gh config get git_protocol)")
    args[-1]="${args[-1]:-https}"
    gh auth login "${args[@]}" --web
elif test "${#needed_scopes[@]}" -gt 0; then
    # Refresh with additional scopes
    unset GH_TOKEN GITHUB_TOKEN
    args+=("${wanted_scopes[@]}")
    args+=(--insecure-storage)
    gh auth refresh "${args[@]}"
fi

unset args wanted_scopes active_scopes needed_scopes

declare -a args=(
    --duration 43200
    --profile default
    --output creds-file
    --aud "${AWS_AUDIENCE:-sts.amazonaws.com}"
    --idp-url "${AWS_IDP_URL:-https://token.rapids.nvidia.com}"
    --role-arn "${AWS_ROLE_ARN:-arn:aws:iam::279114543810:role/rapids-token-sccache-devs}"
)

# Generate temporary S3 credentials that expire after 12 hours.
# These allow RW access to the `rapids-sccache-devs` build cache S3 bucket.

gh nv-gha-aws org nvidia "${args[@]}" 2>/dev/null >/out/credentials

cat <<EOF > /out/sccache
[cache.s3]
bucket = "rapids-sccache-devs"
region = "us-east-2"
key_prefix = "node"

[cache.s3.preprocessor_cache_mode]
use_preprocessor_cache_mode = true
key_prefix = "node/preprocessor"

[dist]
scheduler_url = "https://$(uname -m | sed -e 's/x86_/amd/' -e 's/aarch/arm/').linux.sccache.rapids.nvidia.com"
fallback_to_local_compile = true
max_retries = 5

[dist.auth]
type = "token"
token = "$(gh auth token)"
EOF


unset args
