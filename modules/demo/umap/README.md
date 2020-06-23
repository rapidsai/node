```shell
$ docker run --rm -it \
    --network=host -w /rapids/notebooks \
    -v "$PWD/umap.py:/rapids/notebooks/umap.py" \
    rapidsai/rapidsai-dev-nightly:0.15-cuda10.2-devel-ubuntu18.04-py3.7

\# python umap.py
```
