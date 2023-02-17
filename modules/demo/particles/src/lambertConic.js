/**
 * Copyright (c) 2023 NVIDIA Corporation
 */

const lambertConic = (function() {
  const R       = 6738.137;
  const lambda0 = -87.623177 * Math.PI / 180;
  const phi0    = 41.881832 * Math.PI / 180;
  const phi1    = 33 * Math.PI / 180;
  const phi2    = 45 * Math.PI / 180;
  const n       = (Math.log(Math.cos(phi1) * (1 / Math.cos(phi2)))) /
            Math.log(Math.tan(Math.PI / 4 + phi1 / 2) - (1 / Math.tan(Math.PI / 4 + phi1 / 2)));
  const F = (Math.cos(phi1) * Math.pow(Math.tan(Math.PI / 4 + phi1 / 2), n)) / n;
  return {
    forward: function(phi, lambda) {
      phi *= Math.PI / 180;
      lambda *= Math.PI / 180;
      const rho  = R * F * Math.pow((1 / Math.tan(Math.PI / 4 + phi / 2)), n);
      const rho0 = R * F * Math.pow((1 / Math.tan(Math.PI / 4 + phi0 / 2)), n);
      const E    = rho * Math.sin(n * (lambda - lambda0));
      const N    = rho0 - rho * Math.cos(n * (lambda - lambda0));
      return [E, N];
    },
    inverse: function(E, N) {
      const rho0   = R * F * Math.pow((1 / Math.tan(Math.PI / 4 + phi0 / 2)), n);
      const rho    = Math.sqrt(E * E + (rho0 - N) * (rho0 - N));
      const phi    = 2 * Math.atan(Math.pow(rho / (R * F), 1 / n)) - Math.PI / 2;
      const lambda = lambda0 + Math.atan(E / (rho0 - N)) / n;
      return [phi * 180 / Math.PI, lambda * 180 / Math.PI];
    }
  };
})();
const chicago      = lambertConic.forward(41.881832, -87.623177);
console.log(chicago);
console.log(lambertConic.inverse(chicago[0], chicago[1]));
const denver = lambertConic.forward(39.739235, -104.990251);
console.log(denver);
console.log(lambertConic.inverse(denver[0], denver[1]));

const inverseConic = (function() {
  const R       = 6738.137;
  const lambda0 = -87.623177 * Math.PI / 180;
  const phi0    = 41.881832 * Math.PI / 180;
  const phi1    = 33 * Math.PI / 180;
  const phi2    = 45 * Math.PI / 180;
  const n       = (Math.log(Math.cos(phi1) * (1 / Math.cos(phi2)))) /
            Math.log(Math.tan(Math.PI / 4 + phi1 / 2) - (1 / Math.tan(Math.PI / 4 + phi1 / 2)));
  const F = (Math.cos(phi1) * Math.pow(Math.tan(Math.PI / 4 + phi1 / 2), n)) / n;
  return function(E, N) {
    const rho0   = R * F * Math.pow((1 / Math.tan(Math.PI / 4 + phi0 / 2)), n);
    const rho    = Math.sqrt(E * E + (rho0 - N) * (rho0 - N));
    const phi    = 2 * Math.atan(Math.pow(rho / (R * F), 1 / n)) - Math.PI / 2;
    const lambda = lambda0 + Math.atan(E / (rho0 - N)) / n;
    return [phi * 180 / Math.PI, lambda * 180 / Math.PI];
  }
})();
