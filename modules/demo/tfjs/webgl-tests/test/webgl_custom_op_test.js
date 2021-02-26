/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as tf from '@tensorflow/tfjs-core';
import { engine, test_util } from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import { describeWithFlags } from '@tensorflow/tfjs-core/dist/jasmine_util';
import { WEBGL_ENVS } from '@tensorflow/tfjs-backend-webgl/dist/backend_webgl_test_registry';
describeWithFlags('custom-op webgl', WEBGL_ENVS, () => {
    class SquareAndAddKernel {
        constructor(inputShape) {
            this.variableNames = ['X'];
            this.outputShape = inputShape.slice();
            this.userCode = `
          void main() {
            float x = getXAtOutCoords();
            float value = x * x + x;
            setOutput(value);
          }
        `;
        }
    }
    class SquareAndAddBackpropKernel {
        constructor(inputShape) {
            this.variableNames = ['X'];
            this.outputShape = inputShape.slice();
            this.userCode = `
          void main() {
            float x = getXAtOutCoords();
            float value = 2.0 * x + 1.0;
            setOutput(value);
          }
        `;
        }
    }
    function squareAndAdd(x) {
        const fn = tf.customGrad((x, save) => {
            save([x]);
            const webglBackend = tf.backend();
            const program = new SquareAndAddKernel(x.shape);
            const backpropProgram = new SquareAndAddBackpropKernel(x.shape);
            const outInfo = webglBackend.compileAndRun(program, [x]);
            const value = engine().makeTensorFromDataId(outInfo.dataId, outInfo.shape, outInfo.dtype);
            const gradFunc = (dy, saved) => {
                const [x] = saved;
                const backInfo = webglBackend.compileAndRun(backpropProgram, [x]);
                const back = engine().makeTensorFromDataId(backInfo.dataId, backInfo.shape, backInfo.dtype);
                return back.mul(dy);
            };
            return { value, gradFunc };
        });
        return fn(x);
    }
    it('lets users use custom operations', async () => {
        const inputArr = [1, 2, 3, 4];
        const input = tf.tensor(inputArr);
        const output = squareAndAdd(input);
        test_util.expectArraysClose(await output.data(), inputArr.map(x => x * x + x));
    });
    it('lets users define gradients for operations', async () => {
        const inputArr = [1, 2, 3, 4];
        const input = tf.tensor(inputArr);
        const grads = tf.valueAndGrad(x => squareAndAdd(x));
        const { value, grad } = grads(input);
        test_util.expectArraysClose(await value.data(), inputArr.map(x => x * x + x));
        test_util.expectArraysClose(await grad.data(), inputArr.map(x => 2 * x + 1));
    });
    it('multiplies by dy parameter when it is passed', async () => {
        const inputArr = [1, 2, 3, 4];
        const input = tf.tensor(inputArr);
        const grads = tf.valueAndGrad(x => squareAndAdd(x));
        const { value, grad } = grads(input, tf.zerosLike(input));
        test_util.expectArraysClose(await value.data(), inputArr.map(x => x * x + x));
        test_util.expectArraysClose(await grad.data(), inputArr.map(() => 0.0));
    });
});
//# sourceMappingURL=webgl_custom_op_test.js.map
