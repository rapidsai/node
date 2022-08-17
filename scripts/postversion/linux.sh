#!/usr/bin/env bash

set -Eeo pipefail

lerna version \
    --yes --no-push \
    --ignore-scripts \
    --force-publish="*" \
    --no-git-tag-version \
    ${npm_package_version:-patch}

# Replace ^ with ~
find modules -type f -name 'package.json' -exec \
    sed -i -E -e 's+(@rapidsai/.*)": "\^+\1": "~+g' {} \;

sed -i -E -e "s/$npm_old_version/$npm_package_version/g" USAGE.md
sed -i -E -e "s/$npm_old_version/$npm_package_version/g" docker-compose.devel.yml
sed -i -E -e "s/$npm_old_version/$npm_package_version/g" docker-compose.runtime.yml
sed -i -E -e "s/$npm_old_version/$npm_package_version/g" .github/workflows/release.yml
