# Rapids.js + Deck.gl | Census Visualization

![](./census_demo0.png)
![](./census_demo1.png)

# Run Steps 

## Rendering in OpenGL window
The visualization calls OpenGL window rendering. Go to the root directory `rapids-js/`.

```bash
# run and access
./modules/demo/start.sh deck/census
```

## Rendering with mapbox on web browser

The visualization calls OpenGL window rendering. Go to the demo folder `rapids-js//modules/demo/deck/census/`.

```bash
# run and access
npm start
```

## Data 
There is 1 main dataset:

- 2010 Census for Population Density (~3.7 GB) | download on first run



## FAQ and Known Issues
*What hardware do I need to run this locally?*
To run you need an NVIDIA GPU with at least 24GB of memory, and a Linux OS as defined in the [RAPIDS requirements](https://rapids.ai/start.html#req).

*How are the population and case counts filtered?*
Zooming in or out of a region on the map filters the data to that only displayed. 

*Why is the population data from 2010?*
Only census data is recorded on a block level. For more details on census boundaries refer to the [TIGERweb app](https://tigerweb.geo.census.gov/tigerwebmain/TIGERweb_apps.html). 

*Why is the dataset an arrow?*
JS can't allocate a buffer large enough to fix the whole thing in memory, so the arrow allows us to stream the table in smaller chunks.
One benefit of “arrow” is that data arrives in chunks, so rather than 1 giant table of 320M rows, we have the table split up into 1000 chunks of 320,000 rows each.



## Acknowledgments and Data Sources

- 2010 Population Census and 2018 ACS data used with permission from IPUMS NHGIS, University of Minnesota, [www.nhgis.org](www.nhgis.org) ( not for redistribution )
- Base map layer provided by [mapbox](https://www.mapbox.com/)
- GPU accelerated with [RAPIDS](https://rapids.ai/) [cudf](https://github.com/rapidsai/cudf) 
