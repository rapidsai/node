// based on https://github.com/cheton/is-electron
// https://github.com/electron/electron/issues/2288
/* global window, process, navigator */
/* eslint-disable complexity */
export function isElectron(mockUserAgent?: string) {
    // Renderer process
    if (
        typeof window !== 'undefined' &&
        typeof window.process === 'object' &&
        (<any> window.process).type === 'renderer'
    ) {
        if ((<any> window.process).glfwWindow) {
            return false;
        }
        console.log('isElectron=true');
        return true;
    }
    // Main process
    if (
        typeof process !== 'undefined' &&
        typeof process.versions === 'object' &&
        Boolean((<any> process.versions).electron)
    ) {
        console.log('isElectron=true');
        return true;
    }
    // Detect the user agent when the `nodeIntegration` option is set to true
    const realUserAgent =
        typeof navigator === 'object' && typeof navigator.userAgent === 'string' && navigator.userAgent;
    const userAgent = mockUserAgent || realUserAgent;
    if (userAgent && userAgent.indexOf('Electron') >= 0) {
        console.log('isElectron=true');
        return true;
    }
    return false;
}

export default isElectron;
