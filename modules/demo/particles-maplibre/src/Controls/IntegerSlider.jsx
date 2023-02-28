import React, { useState } from 'react';

function IntegerSlider({ value, min, max, step, onChange }) {
  const [sliderValue, setSliderValue] = useState(value);

  const handleSliderChange = (event) => {
    const newValue = parseInt(event.target.value, 10);
    setSliderValue(newValue);
    onChange(newValue);
  };

  return (
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={sliderValue}
      onChange={handleSliderChange}
      style={{ width: '100%' }}
    />
  );
}

export default IntegerSlider
