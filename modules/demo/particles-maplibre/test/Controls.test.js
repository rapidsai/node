// Copyright (c) 2023, NVIDIA CORPORATION.
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

import '@testing-library/jest-dom';

import {fireEvent, render} from '@testing-library/react';
import React from 'react';

import Controls from '../src/Controls/Controls';

describe('Controls', () => {
  test('renders point budget input with default value', () => {
    const {getByLabelText} = render(<Controls props = {
      {}
    } />);
    const pointBudgetInput = getByLabelText('Point Budget');
    expect(pointBudgetInput).toBeInTheDocument();
    expect(pointBudgetInput).toHaveValue(500000);
  });

  test('updates point budget input and slider value when input changes', () => {
    const { getByLabelText, getByRole, getByText } = render(<Controls props={{}} />);
    const pointBudgetInput = getByLabelText('Point Budget');
    const slider           = getByRole('slider');
    fireEvent.change(pointBudgetInput, {target: {value: '250000'}});
    expect(pointBudgetInput).toHaveValue(250000);
  });

  test('renders visible points and total points labels', () => {
    const { getByText } = render(<Controls props={
      { pointOffset: 10, totalCount: 100 }} />);
    expect(getByText('Visible Points: 10')).toBeInTheDocument();
    expect(getByText('Total Points: 100')).toBeInTheDocument();
  });
});
