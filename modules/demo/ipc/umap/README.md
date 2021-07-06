```shell
$ docker run --rm -it \
    --network=host -w /rapids/notebooks \
    -v "$PWD/umap.py:/rapids/notebooks/umap.py" \
    rapidsai/rapidsai-dev-nightly:21.08-cuda11.2-devel-ubuntu20.04-py3.8

\# python umap.py
```
