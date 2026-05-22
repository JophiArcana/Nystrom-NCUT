#!/bin/bash

while IFS= read -r line; do
    if [[ "$line" =~ ^version\ =\ \"([0-9]+\.[0-9]+\.[0-9]+)\"$ ]]; then
        version=${BASH_REMATCH[1]}
        break
    fi
done < pyproject.toml

python3.12 -m build
python3.12 -m twine upload "dist/nystrom_ncut-$version*"
