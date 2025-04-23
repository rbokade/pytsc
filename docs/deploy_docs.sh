#!/bin/bash

# Exit on error
set -e

# Build docs
cd docs
make html

# Move to build output
cd _build/html

# Add .nojekyll to ensure GitHub Pages loads _static etc.
touch .nojekyll

# Init new git repo inside build dir
git init
git add .
git commit -m "Deploy docs"

# Push to gh-pages branch
git push --force https://github.com/rbokade/pytsc.git main:gh-pages

# Cleanup
cd ../..
rm -rf docs/_build/html/.git
