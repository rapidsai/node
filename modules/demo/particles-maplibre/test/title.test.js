import React from 'react';
import { render } from '@testing-library/react';
import Title from '../src/Title';
import '@testing-library/jest-dom';

describe('Title', () => {
  test('renders the component', () => {
    const { getByText } = render(<Title />);
    const text = getByText('Particles');
    expect(text).toBeInTheDocument();
  });
});
