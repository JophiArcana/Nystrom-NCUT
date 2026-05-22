#!/bin/bash

while IFS= read -r line; do
    if [[ "$line" =~ ^version\ =\ \"([0-9]+\.[0-9]+\.[0-9]+)\"$ ]]; then
        version=${BASH_REMATCH[1]}
        break
    fi
done < pyproject.toml

python -m build
twine upload "dist/nystrom_ncut-$version*"
