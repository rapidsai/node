import { Readable, Writable } from 'stream';
import { spawn, ChildProcessByStdio } from 'child_process';

test(`ipc works between subprocesses`, async () => {
    let src: ChildProcessByStdio<Writable, Readable, null> | undefined;
    let dst: ChildProcessByStdio<Writable, Readable, null> | undefined;
    try {
        src = spawnIPCSourceSubprocess(7, 8);
        const hndl = await readChildProcessOutput(src);
        if (hndl) {
            dst = spawnIPCTargetSubprocess(JSON.parse(hndl));
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
const { Uint8Buffer } = require(".");
const dmem = new Uint8Buffer(8);
const hndl = dmem.getIpcHandle();
dmem.fill(${first}, 0, 4).fill(${second}, 4, 8);
process.stdout.write(JSON.stringify(hndl));
process.on("exit", () => hndl.close());
setInterval(() => { }, 60 * 1000);
`], { stdio: ['pipe', 'pipe', 'inherit'] });
}

function spawnIPCTargetSubprocess({ handle }: { handle: Array<number> }) {
    return spawn('node', ['-e', `
const { Uint8Buffer, IpcMemory } = require(".");
const hmem = new Uint8Array(8);
const dmem = new IpcMemory([${handle.toString()}]);
new Uint8Buffer(dmem).copyInto(hmem).buffer.close();
process.stdout.write(JSON.stringify([...hmem]));
`], { stdio: ['pipe', 'pipe', 'inherit'] });
}
