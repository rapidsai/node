# Rapids.js + Deck.gl | Graph Visualization

![](./graph_demo1.png)
![](./graph_demo2.png)
![](./graph_demo3.png)

# Installation and Run Steps [Work In Progress]

## Base Layer Setup
The visualization uses a Mapbox base layer that requires an access token. Create one for free [here](https://www.mapbox.com/help/define-access-token/). Go to the demo root directory `plotly_demo/` and create a token file named `.mapbox_token`. Copy your token contents into the file.

## Running the Visualization App

You can setup and run the visualization with the conda or docker commands below. Once the app is started, it will look for the datasets locally and if not found will download them.

## Data 
There is 1 main dataset:

- 2010 Census for Population Density (~2.9 GB) | download on first run

For more information on how the Census and ACS data was prepared to show individual points, refer to the `/data_prep` folder.

### Conda Env

```bash
# setup directory
cd plotly_demo

# setup conda environment 
conda env create --name plotly_env --file environment.yml
source activate plotly_env

# run and access
python app.py
```


### Docker

Verify the following arguments in the Dockerfile match your system:

1. CUDA_VERSION: Supported versions are `10.0, 10.1, 10.2`
2. LINUX_VERSION: Supported OS values are `ubuntu16.04, ubuntu18.04, centos7`

The most up to date OS and CUDA versions supported can be found here: [RAPIDS requirements](https://rapids.ai/start.html#req)

```bash
# build
docker build -t plotly_demo .

# run and access via: http://localhost:8050 / http://ip_address:8050 / http://0.0.0.0:8050
docker run --gpus all -d -p 8050:8050 plotly_demo
```

## Dependencies

- plotly=4.5
- cudf
- dash=1.8
- pandas=0.25.3
- cupy=7.1
- datashader=0.10
- dask-cuda=0.12.0
- dash-daq=0.3.2
- dash_html_components
- gunicorn=20.0
- requests=2.22.0+
- pyproj


## FAQ and Known Issues
*What hardware do I need to run this locally?*
To run you need an NVIDIA GPU with at least 24GB of memory, and a Linux OS as defined in the [RAPIDS requirements](https://rapids.ai/start.html#req).

*How are the population and case counts filtered?*
Zooming in or out of a region on the map filters the data to that only displayed. 

*Why is the population data from 2010?*
Only census data is recorded on a block level. For more details on census boundaries refer to the [TIGERweb app](https://tigerweb.geo.census.gov/tigerwebmain/TIGERweb_apps.html). 

*How did you get individual point locations?*
The population density points are randomly placed within a census block and associated to match distribution counts at a census block level. As such, they are not actual individuals, only a statistical representation of one.

*The dashboard stop responding or the chart data disappeared!*
Try using the 'clear all selections' button. If that does no work, use the 'reset GPU' button and then refresh the page. This usually resolves any issue. 

*How do I request a feature or report a bug?*
Create an [Issue](https://github.com/rapidsai/plotly-dash-rapids-census-demo/issues) and we will get to it asap. 


## Acknowledgments and Data Sources

- 2010 Population Census and 2018 ACS data used with permission from IPUMS NHGIS, University of Minnesota, [www.nhgis.org](www.nhgis.org) ( not for redistribution )
- Base map layer provided by [mapbox](https://www.mapbox.com/)
- Dashboard developed with Plot.ly [Dash](https://dash.plotly.com/)
- Geospatial point rendering developed with [Datashader](https://datashader.org/)
- GPU accelerated with [RAPIDS](https://rapids.ai/) [cudf](https://github.com/rapidsai/cudf) and [cupy](https://cupy.chainer.org/) libraries
- For source code visit our [GitHub](https://github.com/rapidsai/plotly-dash-rapids-census-demo)
