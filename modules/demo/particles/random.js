// clang-format off
const { Series, Int32, scope } = require("@rapidsai/cudf");


const getRandom = (n) => {
  const a = Series.sequence({size: n, init: 1});
  const b = a.reverse();
  const s = Array.from({length:10}).reduce(
      (s) => scope(() => s
            .mul((Math.random() > 0.5 ? a : b).mul(Math.random())),
            [a, b]
          ),
      Series.sequence({size: n, init: 1, step: 1})
  ).mod(n);
  return s;
}

// host permutation
const permute = (sequence) => {
  for(let i = 0 ; i < sequence.length ; ++i){
    let j = parseInt(Math.random() * (i+1));
    [sequence[i], sequence[j]] = [sequence[j], sequence[i]]
  }
  return sequence
}
// host range
const range = (n) => {
  return [...Array(n + 1).keys()];
}
console.log('host permutation:');
console.log(permute(range(10)));

const n = 10;


const original = Series.sequence({size: n, init: 0});
const copy = original.copy();
const s = getRandom(n);
const t = getRandom(n);

const positions = Series.sequence({size: n, init: 0, step: 1});

const sIndices = s.cast(new Int32);
console.log("The original random mask.");
console.log(sIndices.toArray());
const mask = Series.sequence({size: n, init: 0, step: 0});
const filling = mask.scatter(sIndices, sIndices).gather(positions);
console.log("The values that exist in the original random mask.");
console.log(filling.toArray());
const fillingMask = filling.ne(0);
const fillingFilter = filling.filter(fillingMask);
console.log("The gathered present values:");
console.log(fillingFilter.toArray());
const missing = Series.sequence({size: n, init: 0, step: 1});
const unfilled = missing.sub(filling);
console.log("The values that are missing from the original random mask.");
console.log(unfilled.toArray());
