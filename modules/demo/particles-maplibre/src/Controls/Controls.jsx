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

import React, { useState } from 'react';

import IntegerSlider from './IntegerSlider.jsx';

function Controls({ props }) {
  const [pointBudget, setPointBudget] = useState(500000);
  const [budgetMin, setBudgetMin] = useState(1000);
  const [budgetMax, setBudgetMax] = useState(1000000);

  const handleValueChange = (pointBudget) => { setPointBudget(pointBudget); };
  const handleInputChange = (pointBudget) => {
    setPointBudget(pointBudget.target.value);
    setBudgetMin(pointBudget.target.value / 3);
    setBudgetMax(pointBudget.target.value * 5);
  };

  return (
    <div className="controls">
      <label>Point Budget
        <input type="number" id="point-budget-input" value={pointBudget} onChange={handleInputChange} />
      </label>
      <p>
        <IntegerSlider
          value={pointBudget}
          min={budgetMin}
          max={budgetMax}
          step={pointBudget / 100}
          onChange={handleValueChange}
        />
      </p>
      <p>Visible Points: {props.pointOffset}</p>
      <p>Total Points: {props.totalCount}</p>
    </div>
  );
}

export default Controls;
