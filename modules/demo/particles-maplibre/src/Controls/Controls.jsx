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
      <p><label>Visible Points: {props.pointOffset}</label></p>
      <p><label>Total Points: {props.totalCount}</label></p>
    </div>
  );
}

export default Controls;
