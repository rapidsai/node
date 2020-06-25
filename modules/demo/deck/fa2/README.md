```shell
$ docker run --rm -it \
    --network=host -w /rapids/notebooks \
    -v "$PWD/fa2.py:/rapids/notebooks/fa2.py" \
    -v "$PWD/data:/rapids/notebooks/data" \
    rapidsai/rapidsai-dev-nightly:0.15-cuda10.2-devel-ubuntu18.04-py3.7

\# python fa2.py
```
