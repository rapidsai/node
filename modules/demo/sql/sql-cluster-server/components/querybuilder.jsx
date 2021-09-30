// Copyright (c) 2021, NVIDIA CORPORATION.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import 'react-awesome-query-builder/lib/css/styles.css';

import React from 'react';
import { Builder, Query, Utils as QbUtils } from 'react-awesome-query-builder';
import MaterialConfig from 'react-awesome-query-builder/lib/config/material';

const config = {
  ...MaterialConfig,
  fields: {
    id: {
      label: 'id',
      type: 'number',
      valueSources: ['value'],
      preferWidgets: ['number'],
    },
    revid: {
      label: 'revid',
      type: 'number',
      valueSources: ['value'],
      preferWidgets: ['number'],
    },
    url: {
      label: 'url',
      type: 'text',
      valueSources: ['value'],
      excludeOperators: ["proximity", "equal", "not_equal"],
      mainWidgetProps: {
        valueLabel: "url",
        valuePlaceholder: "Enter url",
      },
    },
    title: {
      label: 'title',
      type: 'text',
      valueSources: ['value'],
      excludeOperators: ["proximity", "equal", "not_equal"],
      mainWidgetProps: {
        valueLabel: "title",
        valuePlaceholder: "Enter title",
      },
    },
    text: {
      label: 'text',
      type: 'text',
      valueSources: ['value'],
      excludeOperators: ["proximity", "equal", "not_equal"],
      mainWidgetProps: {
        valueLabel: "text",
        valuePlaceholder: "Enter text",
      },
    }
  }
};

const queryValue = {
  'id': '9a99988a-0123-4456-b89a-b1607f326fd8',
  'type': 'group',
  'children1': {
    'a98ab9b9-cdef-4012-b456-71607f326fd9': {
      'type': 'rule',
      'properties': {
        field: null,
        operator: null,
        value: [],
        valueSrc: [],
        'type': 'rule',
      }
    }
  }
};

export class QueryBuilder extends React.Component {
  constructor() {
    super();
    this.state = {
      tree: QbUtils.checkTree(QbUtils.loadTree(queryValue), config),
      config: config,
    };
  }

  render = () => (<div><Query{...config} value={this.state.tree} onChange={
    this.onChange} renderBuilder={this.renderBuilder} />
  </div>)

  renderBuilder = (props) => (
    <div className='query-builder-container'>
      <div className='query-builder qb-lite'>
        <Builder {
          ...props} />
      </div>
    </div>
  )

  onChange = (immutableTree, config) => {
    this.setState({ tree: immutableTree, config: config });
    this.props.onQueryChange(this._parseQuery(JSON.stringify(QbUtils.sqlFormat(immutableTree, config))));
  }

  _parseQuery(query) {
    if (query === undefined || query.length == 0) return '';
    // Hacky, but the sql builder uses 'NOT EMPTY' and 'EMPTY' when constructing the query.
    // Let's just replace any instances with 'NOT NULL' and 'NULL'.
    query = query.replace('NOT EMPTY', 'NOT NULL');
    query = query.replace('EMPTY', 'NULL');
    return `SELECT id, revid, url, title, text FROM test_table WHERE ${JSON.parse(query)}`;
  }
}
