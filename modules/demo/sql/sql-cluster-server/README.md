# SQLCluster Server Demo

This demo demonstrates the SQLCluster module which allows for multi-GPU SQL queries using our SQL engine.

## Main Dependencies

1. @rapidsai/sql
2. fastify-nextjs
3. react-awesome-query-builder

## Installation

To install dependencies, run the following from the root directory for `node-rapids`

```bash
yarn
```

To run the demo
```bash
yarn demo # and select the sql-cluster-server demo from the list of demos

cd modules/demo/sql/sql-cluster-server
yarn start
```

## Interesting Queries

Using the query builder you can seamlessly build queries and execute them against our dataset. Here are some interesting queries if you need some inspiration...

- Who appears in more Wikipedia pages, Plato or Newton?
  - Select `text`, using the `like` operator, type in `Plato`/`Newton` and check out how many results are returned.
- Which programming language is referenced the most across all Wikipedia pages?
  - Select `text`, using the `like` operator, type in your favorite programming language and see how popular it is.
- Is there any Wikipedia page that avoids using the most common english word `the`?
  - Select `text`, using the `Is not empty` operator
  - Click `ADD GROUP`, select `text`, using the `not like` operator, type in `the`.
- How many Wikipedia pages have your first name in the `title`?
  - Select `title`, using the `like` operator, type in your first name.
- How many Wikipedia pages are redirects to other pages?
  - Select `text`, using the `Is empty` operator.

## Dataset

The dataset used for this demo is the entire collection of 2021 english Wikipedia pages. This includes the following for each page...

1. Page ID
2. Revision ID
3. Page URL
4. Page Title
5. Page Text

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
