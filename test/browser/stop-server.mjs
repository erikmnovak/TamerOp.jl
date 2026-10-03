import { writeFile, rm } from 'node:fs/promises';
import { setTimeout } from 'node:timers/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// Let Julia close Bonito sessions normally before Playwright's signal fallback.
// Sending SIGINT while Bonito's cleanup tasks are waiting can interrupt teardown.
export default async function stopServer() {
  if (process.env.TAMEROP_BROWSER_REUSE_SERVER === '1') return;
  const port = process.env.TAMEROP_BROWSER_INTERVAL_PORT || 8848;
  let response;
  try {
    response = await fetch(`http://127.0.0.1:${port}/health`, { signal: AbortSignal.timeout(2000) });
  } catch (error) {
    if (error.cause?.code === 'ECONNREFUSED') return;
    throw error;
  }
  const { pid } = await response.json();
  if (!response.ok || !Number.isSafeInteger(pid) || pid <= 0) throw new Error('Invalid fixture process identity');
  const directory = path.dirname(fileURLToPath(import.meta.url));
  const stopFile = process.env.TAMEROP_BROWSER_STOP_FILE || path.join(directory, '.stop-server');
  await writeFile(stopFile, '');
  const deadline = Date.now() + 15_000;
  while (Date.now() < deadline) {
    try {
      process.kill(pid, 0); // Read-only existence check; do not signal the process.
    } catch (error) {
      if (error.code !== 'ESRCH') throw error;
      await rm(stopFile, { force: true });
      return;
    }
    await setTimeout(100);
  }
  throw new Error('The fixture did not exit after its stop request');
}
