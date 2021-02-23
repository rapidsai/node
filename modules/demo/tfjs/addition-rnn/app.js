import chalk from 'chalk';

// Populate DOM with expected demo elements
document.body.append(
    createElement('p', 'trainStatus'),
    createElement('div', 'lossChart'),
    createElement('div', 'accuracyChart'),
    createElement('p', 'testExamples'),
    createElement('button', 'trainModel'),
    createElement('input', 'digits', 2),
    createElement('input', 'trainingSize', 5000),
    createElement('select', 'rnnType'),
    createElement('input', 'rnnLayers', 1),
    createElement('input', 'rnnLayerSize', 128),
    createElement('input', 'batchSize', 128),
    createElement('input', 'trainIterations', 100),
    createElement('input', 'numTestExamples', 20),
);

document.getElementById('rnnType').append(
    createElement('option', null, 'SimpleRNN'),
    createElement('option', null, 'GRU'),
    createElement('option', null, 'LSTM')
);

document.getElementById('rnnType').selectedIndex = 0;

// Listen for changes to `trainStatus` content and pipe to the console
new MutationObserver((mutations, observer) => {
    console.clear();
    console.log(mutations.map(({ addedNodes }) =>
        Array.from(addedNodes).map((n) => n.textContent).join('\n')
    ).join('\n'));
}).observe(document.getElementById('trainStatus'), {childList: true});

// Listen for changes to `testExamples` content and pipe to the console
new MutationObserver((mutations, observer) => {
    console.log(mutations.map(({ addedNodes }) =>
        Array.from(addedNodes)
            .filter(({ nodeName, classList }) =>
                (nodeName === 'DIV') && (classList = Array.from(classList)) &&
                (classList.includes('answer-wrong') || classList.includes('answer-correct')))
            .map(({ textContent, classList }) =>
                Array.from(classList).includes('answer-correct')
                ? chalk.green(textContent) : chalk.red(textContent))
            .join('\n')
    ).join('\n'));
}).observe(document.getElementById('testExamples'), {childList: true});

// Import and run the TFJS demo
require('./demo');

// Start the TFJS demo training
setTimeout(() => document
    .getElementById('trainModel')
    .dispatchEvent(new KeyboardEvent('click')),
    100
);

function createElement(type, id, value) {
    const elt = document.createElement(type);
    if (arguments.length > 1 && id) elt.setAttribute('id', id);
    if (arguments.length > 2) elt.setAttribute('value', value);
    return elt;
}
