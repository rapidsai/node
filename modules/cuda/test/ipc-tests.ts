import { Readable, Writable } from 'stream';
import { spawn, ChildProcessByStdio } from 'child_process';

test(`ipc works between subprocesses`, async () => {
    let src: ChildProcessByStdio<Writable, Readable, null> | undefined;
    let dst: ChildProcessByStdio<Writable, Readable, null> | undefined;
    try {
        src = spawnIPCSourceSubprocess(7, 8);
        const hndl = await readChildProcessOutput(src);
        if (hndl) {
            dst = spawnIPCTargetSubprocess(hndl);
            const data = await readChildProcessOutput(dst);
            if (data) {
                expect(data).toStrictEqual('[7,7,7,7,8,8,8,8]');
            } else {
                throw new Error(`Invalid data from target child process: ${JSON.stringify(data)}`);
            }
        } else {
            throw new Error(`Invalid IPC handle from source child process: ${JSON.stringify(hndl)}`);
        }
    } finally {
        dst && !dst.killed && dst.kill();
        src && !src.killed && src.kill();
    }
});

async function readChildProcessOutput(proc: ChildProcessByStdio<Writable, Readable, null>) {
    const { stdout } = proc;
    return (async () => {
        for await (const chunk of stdout) {
            if (chunk) { return '' + chunk; }
        }
        return '';
    })();
}

function spawnIPCSourceSubprocess(first: number, second: number) {
    return spawn('node', [`-e`, `
const { CUDAMemory, CUDA } = require(".");
const dmem = CUDAMemory.alloc(8);
dmem.fill(${first}, 0, 4).fill(${second}, 4, 8);
const hndl = CUDA.ipc.getMemHandle(dmem.buffer);
process.stdout.write(JSON.stringify([...hndl]));
process.on("exit", () => {
    CUDA.ipc.closeMemHandle(hndl);
    CUDA.mem.free(dmem.buffer);
});
setInterval(() => { }, 60 * 1000);
`], { stdio: ['pipe', 'pipe', 'inherit'] });
}

function spawnIPCTargetSubprocess(hndl: string) {
    return spawn('node', ['-e', `
const { CUDAMemory, CUDA } = require(".");
const hmem = new Buffer(8);
const hndl = Buffer.from(JSON.parse("${hndl}"));
const dmem = new CUDAMemory(CUDA.ipc.openMemHandle(hndl));
dmem.copyInto(hmem);
process.stdout.write(JSON.stringify([...hmem]));
CUDA.ipc.closeMemHandle(dmem.buffer);
`], { stdio: ['pipe', 'pipe', 'inherit'] });
}
