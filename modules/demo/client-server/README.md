# Client-Server Demo

This project demonstrates ways of serving @rapidsai/cudf over a node-express api using 2 dashboards:
1. Uber Movement Dataset
2. Fannie Mae Mortgage Dataset

## Main Dependencies

1. @rapidsai/cudf
2. express
3. deckgl
4. @apache/arrow

## Data

1. Uber Dataset

The data needs to be downloaded from uber movements page [here](https://movement.uber.com/explore/san_francisco/travel-times). It's not a direct download link, and the following sequence of actions should facilitate the data download:

`Click 'Download data' > Click 'All data' > Slect '2020 Quarter' > Download 'Travel Times By Date By Hour Buckets (All Days).csv' (1.7gb)`

Save the file as `san_fran_uber.csv` in the folder `node-rapids/modules/demo/client-server/public/data`

2. Mortgage Dataset

This dataset can be downloaded [here](https://drive.google.com/file/d/1KZBzbw9z-BkyuxfN4HB0u_vKbpaEjDTm/view?usp=sharing).

Save the file as `mortgage.csv` in the folder `node-rapids/modules/demo/client-server/public/data`

## Installation

To install dependencies, run the following from the root directory for `node-rapids`

```bash
yarn
```

To run the demo
```bash
yarn demo #and select the client-server demo from the list of demos

cd modules/demo/client-server
yarn start #yarn dev for dev environment
```

# Uber Movement Dashboard
![Screenshot](./_static/uber.png)


# Mortgage Dashboard
![Screenshot](./_static/mortgage.png)
