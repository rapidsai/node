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

import { StaticMap } from 'react-map-gl';
import { useRef, useState, Fragment } from 'react';
import AutoSizer from 'react-virtualized-auto-sizer';

import DeckWithMapboxMaps from './deck-with-mapbox-maps';

import isBrowser from '../is-browser';
import WebRTCFrame from './webrtcframe';
import styles from '../styles/Playground.module.css';

const AceEditor = (() => {
  if (isBrowser) {
    const AceEditor = require('react-ace').default;
    require('brace/mode/json');
    require('brace/theme/github');
    return AceEditor;
  }
  return () => <div />;
})();

const { JSONConverter, JSONConfiguration } = require('@deck.gl/json');
const JSON_CONVERTER_CONFIGURATION = require('@rapidsai/demo-deck-playground/src/configuration').default;
const JSON_TEMPLATES = require('@rapidsai/demo-deck-playground/json-examples').default;
const INITIAL_TEMPLATE = Object.keys(JSON_TEMPLATES)[0];

type Props = {
  rtcId: string;
};

const Playground = (props: Props) => {

  const configuration = useRef(new JSONConfiguration(JSON_CONVERTER_CONFIGURATION));
  const converter = useRef(new JSONConverter({ configuration: configuration.current }));

  const [template, updateTemplate] = useState(INITIAL_TEMPLATE);
  const [json, updateJSON] = useState(JSON_TEMPLATES[template]);
  const [jsonProps, updateJSONProps] = useState(() => converter.current.convert(json));
  const [editorContent, updateEditorContent] = useState(JSON.stringify(json, null, 2));

  const onEditorChanged = (text: string) => {
    // Parse JSON, while capturing and ignoring exceptions
    try {
      const json = text && JSON.parse(text);
      updateJSON(json);
      updateJSONProps(converter.current.convert(json));
      updateEditorContent(JSON.stringify(json, null, 2));
    } catch (error) {
      // ignore error, user is editing and not yet correct JSON
    }
  };

  return (
    <Fragment>
      <div className={styles['left-pane']}>
        <select
          name="JSON templates"
          defaultValue={template}
          onChange={(event) => updateTemplate(event.target.value)}>
          {Object.entries(JSON_TEMPLATES).map(([key]) => (
            <option key={key} value={key}>
              {key}
            </option>
          ))}
        </select>
        <div className={styles['editor']}>
          <AutoSizer>
            {({ width, height }) => (
              <AceEditor
                width={`${width}px`}
                height={`${height}px`}
                mode="json"
                theme="github"
                onChange={onEditorChanged}
                name="AceEditorDiv"
                editorProps={{ $blockScrolling: true }}
                value={editorContent}
              />
            )}
          </AutoSizer>
        </div>
      </div>
      <div className={styles['right-pane']}>
        <AutoSizer defaultWidth={800} defaultHeight={600}>
          {({ width, height }) => (
            <WebRTCFrame
              width={width}
              height={height}
              rtcId={props.rtcId}
            >
              <DeckWithMapboxMaps
                id="json-deck"
                {...jsonProps}
                Map={StaticMap}
              />
            </WebRTCFrame>
          )}
        </AutoSizer>
      </div>
    </Fragment>
  )
};

export default Playground;
