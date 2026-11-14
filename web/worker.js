// Web Worker: owns the Pyodide (Python + NumPy) runtime so the page stays responsive.
// Terminating this worker cancels a calculation.
import { loadPyodide } from 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.mjs';

const PACKAGE_FILES = ['__init__', 'model', 'solver', 'imaging', 'experiment'];
let py;

const ready = (async () => {
  py = await loadPyodide();
  await py.loadPackage('numpy');
  // Copy the simulation package (identical to the repository's microseismic/) into
  // Pyodide's virtual file system so `import microseismic` works.
  py.FS.mkdirTree('microseismic');
  for (const name of PACKAGE_FILES) {
    const response = await fetch(`./microseismic/${name}.py`);
    if (!response.ok) throw new Error(`Cannot load microseismic/${name}.py (${response.status})`);
    py.FS.writeFile(`microseismic/${name}.py`, await response.text());
  }
  const bridge = await fetch('./bridge.py');
  if (!bridge.ok) throw new Error(`Cannot load bridge.py (${bridge.status})`);
  py.FS.writeFile('bridge.py', await bridge.text());
  py.globals.set('report_progress', (percent, phase) => self.postMessage({ type: 'progress', percent, phase }));
  await py.runPythonAsync('import sys\nif "." not in sys.path: sys.path.insert(0, ".")\nfrom bridge import run');
  self.postMessage({ type: 'ready' });
})();
ready.catch(error => self.postMessage({ type: 'error', message: String(error) }));

self.onmessage = async ({ data }) => {
  try {
    await ready;
    py.globals.set('request_json', JSON.stringify(data));
    const result = await py.runPythonAsync('run(request_json, report_progress)');
    try {
      const converted = result.toJs({ dict_converter: Object.fromEntries });
      self.postMessage({ type: 'result', id: data.id, result: converted });
    } finally { result.destroy(); }
  } catch (error) {
    self.postMessage({ type: 'error', id: data.id, message: String(error).trim().split('\n').at(-1) });
  }
};
