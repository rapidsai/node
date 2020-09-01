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
There is 1 main dataset:

- 2010 Census for Population Density (~2.9 GB) | download on first run

For more information on how the Census and ACS data was prepared to show individual points, refer to the `/data_prep` folder.



## FAQ and Known Issues
*What hardware do I need to run this locally?*
To run you need an NVIDIA GPU with at least 24GB of memory, and a Linux OS as defined in the [RAPIDS requirements](https://rapids.ai/start.html#req).

*How are the population and case counts filtered?*
Zooming in or out of a region on the map filters the data to that only displayed. 

*Why is the population data from 2010?*
Only census data is recorded on a block level. For more details on census boundaries refer to the [TIGERweb app](https://tigerweb.geo.census.gov/tigerwebmain/TIGERweb_apps.html). 

*How did you get individual point locations?*
The population density points are randomly placed within a census block and associated to match distribution counts at a census block level. As such, they are not actual individuals, only a statistical representation of one.


## Acknowledgments and Data Sources

- 2010 Population Census and 2018 ACS data used with permission from IPUMS NHGIS, University of Minnesota, [www.nhgis.org](www.nhgis.org) ( not for redistribution )
- Base map layer provided by [mapbox](https://www.mapbox.com/)
- Dashboard developed with Plot.ly [Dash](https://dash.plotly.com/)
- Geospatial point rendering developed with [Datashader](https://datashader.org/)
- GPU accelerated with [RAPIDS](https://rapids.ai/) [cudf](https://github.com/rapidsai/cudf) and [cupy](https://cupy.chainer.org/) libraries
- For source code visit our [GitHub](https://github.com/rapidsai/plotly-dash-rapids-census-demo)
