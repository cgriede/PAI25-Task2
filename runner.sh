#!/bin/bash
docker build --tag task2 .; docker run --rm -u $(id -u):$(id -g) -v "$(pwd):/results" task2
#windows specific docker cmd


