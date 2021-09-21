# BlazingCluster Server Demo

This demo demonstrates the BlazingCluster module which allows for multi-GPU SQL queries using BlazingSQL on node.

## Main Dependencies

1. @rapidsai/blazingsql
2. fastify-nextjs
3. react-awesome-query-builder

## Installation

To install dependencies, run the following from the root directory for `node-rapids`

```bash
yarn
```

To run the demo
```bash
yarn demo # and select the blazingcluster-server demo from the list of demos

cd modules/demo/blazingsql/blazingcluster-server
yarn start
```

## Dataset

The dataset used for this demo is the entire collection of 2021 english Wikipedia pages. This includes the following for each page...

1. Page ID
2. Revision ID
3. Page URL
4. Page Title
5. Page text

This ends up totaling to about ~17GB (uncompressed) worth of data.

### Dataset Extraction

There are quite a lot of outdated tutorials on how to extract Wikipedia data that no longer work. The method I'll be showcasing was the only one that was successful.

1. Visit https://dumps.wikimedia.org/enwiki/latest/ and download `enwiki-latest-pages-articles.xml.bz2`. There are various other locations available as well to download the latest wikipedia pages-article dump.
2. Extract the wikipedia pages-article dump (this should be a `.xml` file)
3. We will be using this tool to extract the wikipedia pages-article `.xml` file https://github.com/attardi/wikiextractor
4. Clone the `wikiextractor` repo and follow the insallation instructions to be able to run the script locally.
5. Move the extracted wikipedia page-article `.xml` file inside of your `wikiextractor` cloned directory
6. You can follow the `README.md` on the `wikiextractor` page for additional arguments to run the script
7. Use your own specific command args or use the following...

`python -m wikiextractor.WikiExtractor --json enwiki-20210901-pages-articles-multistream.xml`

The running of this command should create a `text` folder which will contain multiple folders inside. These folders contain wikipedia page data in `.json` form.

From here you can simply use a python script to parse the data in the form you need it in.
