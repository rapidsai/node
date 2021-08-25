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

import { useRef, useState, useCallback } from 'react';
import type { NextPage, GetServerSidePropsContext } from 'next';
import Head from 'next/head';

import isBrowser from '../../is-browser';
import styles from '../../styles/Home.module.css';
import WebRTCFrame from '../../components/webrtcframe';

import AutoSizer from 'react-virtualized-auto-sizer';


const AceEditor = (() => {
  if (isBrowser) {
    const AceEditor = require('react-ace');
    require('brace/mode/json');
    require('brace/theme/github');
    return AceEditor;
  }
  return () => { };
})();

const DeckGL = require('@deck.gl/react').default;
const { JSONConverter, JSONConfiguration } = require('@deck.gl/json');

const JSON_CONVERTER_CONFIGURATION = require('@rapidsai/demo-deck-playground/src/configuration');
const JSON_TEMPLATES = require('@rapidsai/demo-deck-playground/json-examples');
const INITIAL_TEMPLATE = Object.keys(JSON_TEMPLATES)[0];

type Props = {
  rtcId: string;
  render: (props: any, ...children: React.ReactNode[]) => React.ReactElement;
};

import renderRemoteOrVideo from '../../render/render';

export async function getServerSideProps({ params = { rtcId: '' } }: GetServerSidePropsContext<{ rtcId: string }>) {
  const { getPeer } = require('../../render/broker');
  if (getPeer(params.rtcId)) {
    return {
      props: {
        rtcId: params.rtcId
      }
    };
  }
  return {
    redirect: {
      permanent: false,
      destination: `/`,
    }
  };
}

const Playground = (({ rtcId }) => {

  const configuration = useRef(new JSONConfiguration(JSON_CONVERTER_CONFIGURATION));
  const converter = useRef(new JSONConverter({ configuration: configuration.current }));

  const [json, updateJSON] = useState(JSON_TEMPLATES[INITIAL_TEMPLATE]);
  const [editorText, updateEditorText] = useState(JSON.stringify(json, null, 2));
  const jsonProps = useCallback((json: any) => converter.current.convert(json), [json]);

  const onEditorChanged = (text: string) => {
    // Parse JSON, while capturing and ignoring exceptions
    try {
      const json = text && JSON.parse(text);
      updateJSON(json);
      updateEditorText(JSON.stringify(json, null, 2));
    } catch (error) {
      // ignore error, user is editing and not yet correct JSON
    }
  };

  return (
    <div className={styles.container}>
      <Head>
        <title>rtc session id: {rtcId}</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className={styles.main}>
        {/* Left Pane: Ace Editor and Template Selector */}
        <div id="left-pane">
          <select name="JSON templates">
            {Object.entries(JSON_TEMPLATES).map(([key]) => (
              <option key={key} value={key}>
                {key}
              </option>
            ))}
          </select>
          <div id="editor">
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
                  value={editorText}
                />
              )}
            </AutoSizer>
          </div>
        </div>

        <WebRTCFrame rtcId={rtcId} render={renderRemoteOrVideo}>
          <DeckGL {...jsonProps} />
        </WebRTCFrame>
      </main>
    </div>
  );
}) as NextPage<Props>;

export default Playground;
