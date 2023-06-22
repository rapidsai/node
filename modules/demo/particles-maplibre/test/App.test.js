import React from 'react';
import { render } from '@testing-library/react';
import App from '../src/App.jsx';
import '@testing-library/jest-dom';

/*
const { JSDOM } = require('jsdom');
const jsdom = new JSDOM('<!doctype html><html><body></body></html>', {
  url: 'http://localhost'
});
global.window = jsdom.window;
global.document = jsdom.window.document;

window.URL.createObjectURL = jest.fn(() => 'mock-url');
*/
jest.mock('../src/App.css', () => ({}));
jest.mock('maplibre-gl', () => ({
  Map: jest.fn(() => ({
    on: jest.fn(() => ({})),
    remove: jest.fn(() => ({}))
  }))
}));
jest.mock('apache-arrow', () => ({}));

describe('App', () => {
  test('renders without crashing', () => {
    render(<App />);
  });
});

