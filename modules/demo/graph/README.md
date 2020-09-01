# Rapids.js + Deck.gl | Graph Visualization

![](./graph_demo1.png)
![](./graph_demo2.png)
![](./graph_demo3.png)

# Run Steps

## OpenGL Window Setup
The visualization calls OpenGL window rendering from inside a docker container. Go to the demo root directory `rapids-js/`.

```bash
# run and access
npm run demo modules/demo/graph
```

## Run the Docker Container

You can setup and run the visualization with the docker commands below. Once the app is started, it will render the graph demo with openGL window. 


```bash
# setup directory
cd modules/demo/graph

# pull docker image
docker run --rm -it \
    --network=host -w "$PWD" -v "$PWD:$PWD" \
    rapidsai/rapidsai-nightly:cuda10.2-runtime-ubuntu18.04 \
    bash -l

# dask
pip install "git+https://github.com/dask/distributed.git" --upgrade --no-deps
pip install "git+https://github.com/dask/dask.git" --upgrade --no-deps

# run and access
python fa2.py
```


## Data 
There are 10 CSV files:

- Friday-02-03-2018_TrafficForML_CICFlowMeter.csv
- Friday-16-02-2018_TrafficForML_CICFlowMeter.csv
- Friday-23-02-2018_TrafficForML_CICFlowMeter.csv
- Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv
- Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv
- Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv
- Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv
- Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv
- Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv
- Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv

The dataset comes from CSE-CIC-IDS2018 on AWS: A collaborative project between the Communications Security Establishment (CSE) & the Canadian Institute for Cybersecurity (CIC).
For more information on how the data was extracted, refer to the CSE-CIC-IDS2018, [https://www.unb.ca/cic/datasets/ids-2018.html](https://www.unb.ca/cic/datasets/ids-2018.html).



## FAQ and Known Issues
*What hardware do I need to run this locally?*
To run you need an NVIDIA GPU with at least 24GB of memory, and a Linux OS as defined in the [RAPIDS requirements](https://rapids.ai/start.html#req).


*How did you get source and destination nodes?*
There’s no defined source and destination nodes in this dataset. the easiest way to do that is to run some sort of table -> graph transformation with a hypergraph.
The graph demo is an example of using cuDF/cuGraph in Python, and sharing the CUDA buffers with OpenGL in node.js via node-cuda bindings.



## Acknowledgments and Data Sources

- CSE-CIC-IDS2018 on AWS: A collaborative project between the Communications Security Establishment (CSE) & the Canadian Institute for Cybersecurity (CIC), [https://www.unb.ca/cic/datasets/ids-2018.html](https://www.unb.ca/cic/datasets/ids-2018.html) 
- GPU accelerated with [RAPIDS](https://rapids.ai/) [cudf](https://github.com/rapidsai/cudf) and [cugraph](https://github.com/rapidsai/cugraph) libraries

