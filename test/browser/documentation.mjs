// Review the built documentation without starting a Julia inspector or notebook.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const directory = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(directory, '../../docs/build');
const output = path.join(directory, 'test-results/documentation');
// A project Pages site lives below the repository name, not at the host root.
const prefix = '/TamerOp.jl/';
await fs.access(path.join(root, 'catalog.json'));
await fs.mkdir(output, { recursive: true });
const mime = { '.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript',
  '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png',
  '.woff2': 'font/woff2' };
const server = http.createServer(async (req, res) => {
  try {
    let pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    if (!pathname.startsWith(prefix)) throw new Error('Outside site prefix');
    pathname = '/' + pathname.slice(prefix.length);
    if (pathname.endsWith('/')) pathname += 'index.html';
    const filename = path.resolve(root, '.' + pathname);
    if (!filename.startsWith(root + path.sep)) throw new Error('Outside build');
    const bytes = await fs.readFile(filename);
    res.writeHead(200, { 'Content-Type': mime[path.extname(filename)] ?? 'application/octet-stream' });
    res.end(bytes);
  } catch {
    res.writeHead(404); res.end('Not found');
  }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const base = `http://127.0.0.1:${server.address().port}${prefix}`;
let browser;
const checked = [];
try {
  browser = await chromium.launch({ headless: true, chromiumSandbox: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1050 } });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {
    if (message.type() === 'error' && /KaTeX|MathJax|ParseError/i.test(message.text()))
      errors.push(message.text());
  });
  page.on('response', response => {
    if (response.url().startsWith(new URL(base).origin) && response.status() >= 400)
      errors.push(`${response.status()} ${response.url()}`);
  });
  const routes = ['index.html', 'reading_map.html', 'topic_map.html',
    'topics/encodings.html', 'collections/mathematics.html', 'collections/using.html',
    'collections/recipes.html', 'start/install.html', 'guides/optional_integrations.html',
    'collections/api.html', 'guides/inputs_to_objects.html', 'guides/spaces_and_maps.html',
    'implementation/qq_coordinates.html', 'benchmarks/phat.html',
    'contributing/index.html', 'tutorials/ring.html'];
  async function checkPageRendering(label) {
    await page.evaluate(() => document.fonts.ready);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, label);
    const brokenImages = await page.locator('article img').evaluateAll(async images => {
      await Promise.all(images.map(image => image.decode().catch(() => {})));
      return images.filter(image => !image.complete || !image.naturalWidth).map(image => image.src);
    });
    assert.deepEqual(brokenImages, [], label);
    assert.equal(await page.locator('.katex-error, mjx-merror').count(), 0, `${label}: mathematics renders`);
  }
  for (const route of routes) {
    await page.goto(base + route);
    await page.locator('#site-navigation').waitFor();
    await checkPageRendering(route);
    assert.equal(await page.locator('.site-entry-links a').count(), 4, route);
    assert.deepEqual(await page.locator('.site-collection').evaluateAll(elements =>
      elements.map(element => element.dataset.collection)),
      ['mathematics', 'using', 'recipes', 'api', 'implementation', 'benchmarks'], route);
    assert.equal(await page.locator('#documenter-sidebar-button').isVisible(), false, route);
    assert.equal(await page.locator('.site-nav-bottom a').filter({ hasText: 'Contributors' }).count(), 1);
    await page.screenshot({ path: path.join(output, route.replaceAll('/', '-') + '.png') });
    checked.push(route);
  }

  // The directory entrance and every nested Introduction link share one page.
  await page.goto(base);
  const introduction = await page.locator('#documenter-page').innerText();
  await page.goto(base + 'guides/spaces_and_maps.html');
  await page.locator('.site-entry-links a').filter({ hasText: /^Introduction$/ }).click();
  assert.equal(page.url(), base + 'index.html');
  assert.equal(await page.locator('#documenter-page').innerText(), introduction);
  assert.equal(await page.locator('.home-opening').count(), 1);

  // Downloaded notebooks must work from the same nested deployment location.
  for (const lesson of ['ring', 'inspect_encoding', 'inputs_to_objects']) {
    const response = await context.request.get(base + `downloads/${lesson}.ipynb`);
    assert.equal(response.status(), 200);
    const notebook = await response.json();
    assert.equal(notebook.nbformat, 4);
    assert.ok(notebook.cells.some(cell => cell.cell_type === 'code' && cell.outputs?.length));
  }

  // Direct entry into an article retains collection context and separates its outline.
  await page.goto(base + 'guides/spaces_and_maps.html');
  assert.equal(await page.locator('.site-collection[open]').count(), 1);
  assert.equal(await page.locator('.site-collection a[aria-current="page"]').count(), 1);
  assert.ok(await page.locator('.site-outline a').count() > 3);
  assert.equal(await page.locator('.docs-footer-nextpage').count(), 0);
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  for (const selector of ['.site-entry-links', '.site-nav-bottom']) {
    const box = await page.locator(selector).boundingBox();
    assert.ok(box.y >= 0 && box.y + box.height <= 1050, `${selector} remains visible`);
  }
  await page.evaluate(() => window.scrollTo(0, 0));
  const firstHeading = page.locator('.site-outline a').first();
  const fragment = await firstHeading.getAttribute('href');
  await firstHeading.click();
  assert.equal(new URL(page.url()).hash, fragment);

  // Search the live Documenter index, including labels added to dynamic results.
  await page.locator('#documenter-search-query').click();
  await page.locator('.documenter-search-input').fill('spaces');
  await page.locator('.search-result-link .site-search-meta').first().waitFor();
  const labels = await page.locator('.site-search-meta').allTextContents();
  assert.ok(labels.some(label => /guide|lesson/i.test(label)), labels.join('\n'));
  await page.keyboard.press('Escape');

  // Native disclosures remain keyboard-operable; narrow screens expose the same menu.
  for (const width of [768, 390, 320]) {
    await page.setViewportSize({ width, height: 844 });
    await page.goto(base + 'guides/spaces_and_maps.html');
    const toggle = page.locator('#documenter-sidebar-button');
    await toggle.focus(); await page.keyboard.press('Enter');
    assert.equal(await toggle.getAttribute('aria-expanded'), 'true');
    await page.waitForFunction(() =>
      document.getElementById('site-navigation').getBoundingClientRect().left === 0);
    await page.locator('.site-nav-bottom a').scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(output, `navigation-${width}.png`) });
    await page.keyboard.press('Escape');
    assert.equal(await toggle.getAttribute('aria-expanded'), 'false');
    assert.equal(await toggle.evaluate(el => el === document.activeElement), true);
    await page.waitForFunction(() => {
      const menu = document.getElementById('site-navigation');
      return menu.getBoundingClientRect().right <= 0 || getComputedStyle(menu).visibility === 'hidden';
    });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
    const outline = page.locator('.site-outline-disclosure');
    assert.equal(await outline.getAttribute('open'), null);
    await outline.locator('summary').click();
    assert.notEqual(await outline.getAttribute('open'), null);
    await page.screenshot({ path: path.join(output, `article-${width}.png`) });
  }
  await page.locator('#documenter-sidebar-button').click();
  await page.locator('#documenter-search-query').click();
  await page.locator('.documenter-search-input').fill('maps');
  await page.locator('.site-search-meta').first().waitFor();
  await page.keyboard.press('Escape');
  await page.waitForFunction(() => document.activeElement?.id === 'documenter-sidebar-button');
  await page.setViewportSize({ width: 1440, height: 1050 });
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.goto(base + 'topic_map.html');
  assert.equal(await page.locator('#documenter-sidebar-button').isVisible(), false);
  assert.equal(Math.round((await page.locator('#site-navigation').boundingBox()).width), 280);
  await page.screenshot({ path: path.join(output, 'topic-map-dark.png') });
  // Changing Documenter's theme must preserve the navigation's own UI state.
  for (const theme of ['catppuccin-latte', 'catppuccin-frappe', 'catppuccin-macchiato', 'catppuccin-mocha']) {
    await page.locator('#documenter-settings-button').click();
    await page.locator('#documenter-themepicker').selectOption(theme);
    await page.locator('#documenter-settings button.delete').click();
    assert.equal(await page.locator('#documenter-sidebar-button').isVisible(), false, theme);
    assert.equal(Math.round((await page.locator('#site-navigation').boundingBox()).width), 280, theme);
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator('#documenter-settings-button').click();
  await page.locator('#documenter-themepicker').selectOption('documenter-light');
  await page.locator('#documenter-settings button.delete').click();
  await page.waitForFunction(() => !document.documentElement.classList.contains('theme--documenter-dark'));
  await page.locator('#documenter-sidebar-button').click();
  assert.equal(await page.locator('#documenter-sidebar-button').getAttribute('aria-expanded'), 'true');
  await page.locator('.site-nav-close').waitFor({ state: 'visible' });
  assert.equal(await page.locator('.site-nav-close').isVisible(), true);
  await page.keyboard.press('Escape');
  assert.equal(await page.locator('#documenter-sidebar-button').isVisible(), true);

  // Review both construction routes where readers enter them, including saved figures.
  // Viewport screenshots keep the long guide readable at its actual display size.
  for (const theme of ['documenter-light', 'documenter-dark']) {
    const themeName = theme.replace('documenter-', '');
    await page.locator('#documenter-settings-button').click();
    await page.locator('#documenter-themepicker').selectOption(theme);
    await page.locator('#documenter-settings button.delete').click();
    for (const width of [1440, 390, 320]) {
      await page.setViewportSize({ width, height: width === 1440 ? 1050 : 844 });
      await page.goto(base + 'index.html');
      await page.locator('.katex').first().waitFor();
      await checkPageRendering(`homepage ${themeName} ${width}`);
      await page.screenshot({ path: path.join(output, `homepage-${themeName}-${width}.png`) });
      const entrances = page.locator('.home-start');
      assert.equal(await entrances.count(), 2);
      // At narrow widths both learning choices remain side by side.
      const entranceBoxes = await Promise.all([0, 1].map(i => entrances.nth(i).boundingBox()));
      assert.ok(entranceBoxes.every(box => box && box.x >= 0 && box.x + box.width <= width));
      assert.ok(Math.abs(entranceBoxes[0].y - entranceBoxes[1].y) < 2 || width === 1440);
      const viewportHeight = page.viewportSize().height;
      assert.ok(entranceBoxes.every(box => box.y >= 0 && box.y + box.height <= viewportHeight),
        `Both learning entrances fit the initial ${width}×${viewportHeight} viewport`);
      await page.locator('.home-workflow').scrollIntoViewIfNeeded();
      await page.screenshot({ path: path.join(output, `homepage-workflow-${themeName}-${width}.png`) });
      await page.locator('.home-perspectives').scrollIntoViewIfNeeded();
      await page.screenshot({ path: path.join(output, `homepage-capabilities-${themeName}-${width}.png`) });

      // Follow a real homepage entrance rather than opening the guide only by URL.
      await page.locator('.home-returning a').click();
      assert.equal(page.url(), base + 'guides/inputs_to_objects.html');
      await page.locator('.katex').first().waitFor();
      await checkPageRendering(`inputs guide ${themeName} ${width}`);
      assert.equal(await page.locator('.site-collection[open]').getAttribute('data-collection'), 'using');
      await page.screenshot({ path: path.join(output, `inputs-${themeName}-${width}.png`) });
      const figures = page.locator('article img');
      assert.ok(await figures.count() >= 2, 'Both input routes have a saved visual result');
      const selectedFigures = [...new Set([0, (await figures.count()) - 1])];
      for (const i of selectedFigures) {
        assert.ok((await figures.nth(i).getAttribute('alt'))?.trim(), 'Teaching figures have descriptions');
        await figures.nth(i).scrollIntoViewIfNeeded();
        await page.screenshot({ path: path.join(output, `inputs-figure-${i + 1}-${themeName}-${width}.png`) });
      }
      // The computed matrices should be readable as matrices, not a long tuple.
      for (const [name, shape] of [['addition', /2×4 Matrix/], ['inclusion', /4×2 Matrix/]]) {
        const matrixResult = page.locator('article pre').filter({ hasText: shape }).first();
        await matrixResult.scrollIntoViewIfNeeded();
        await page.screenshot({ path: path.join(output, `inputs-${name}-${themeName}-${width}.png`) });
        assert.equal(await matrixResult.evaluate(element => element.scrollHeight > element.clientHeight + 1), false,
          `The ${name} matrix rows remain visible at ${width}px`);
      }
      const algebraDiagram = page.locator('article .katex-display').last();
      await algebraDiagram.scrollIntoViewIfNeeded();
      assert.ok((await algebraDiagram.innerText()).trim(), 'The algebra continuation has a rendered diagram');
      assert.equal(await algebraDiagram.locator('.katex-error').count(), 0);
      await checkPageRendering(`inputs algebra diagram ${themeName} ${width}`);
      await page.screenshot({ path: path.join(output, `inputs-algebra-${themeName}-${width}.png`) });
      assert.equal(await algebraDiagram.evaluate(element => element.scrollWidth > element.clientWidth + 1), false,
        `The algebra diagram fits its displayed width at ${width}px`);
    }
  }
  assert.deepEqual(errors, [], 'Browser JavaScript errors');
  await context.close();

  // Navigation and topic destinations are actual HTML links, including without JS.
  const plain = await browser.newContext({ javaScriptEnabled: false, viewport: { width: 390, height: 844 } });
  const fallback = await plain.newPage();
  await fallback.goto(base + 'topic_map.html');
  const topics = fallback.locator('article a[href="topics/encodings.html"]').first();
  await topics.click();
  assert.match(fallback.url(), /topics\/encodings\.html$/);
  assert.ok(await fallback.locator('article a[href*="finite_encodings.html"]').count());
  await fallback.screenshot({ path: path.join(output, 'no-javascript.png') });
  await plain.close();
  await browser.close(); browser = null;

  // Use Chrome's tab zoom API: reducing viewport width alone is not browser zoom.
  const extension = path.join(directory, 'zoom-extension');
  const zoomContext = await chromium.launchPersistentContext('', {
    channel: 'chromium', headless: true, chromiumSandbox: true,
    viewport: { width: 1440, height: 1050 },
    args: [`--disable-extensions-except=${extension}`, `--load-extension=${extension}`],
  });
  try {
    const worker = zoomContext.serviceWorkers()[0] || await zoomContext.waitForEvent('serviceworker');
    const zoomPage = await zoomContext.newPage();
    await zoomPage.goto(base + 'guides/spaces_and_maps.html');
    const dpr = await zoomPage.evaluate(() => devicePixelRatio);
    for (const factor of [1.5, 2]) {
      const actual = await worker.evaluate(async ({ url, factor }) => {
        const tabs = await chrome.tabs.query({ url });
        await chrome.tabs.setZoom(tabs[0].id, factor);
        return chrome.tabs.getZoom(tabs[0].id);
      }, { url: zoomPage.url(), factor });
      assert.equal(actual, factor);
      await zoomPage.waitForFunction(({ dpr, factor }) =>
        Math.abs(devicePixelRatio - dpr * factor) < .01, { dpr, factor });
      assert.equal(await zoomPage.evaluate(() => visualViewport.scale), 1);
      assert.equal(await zoomPage.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
      const toggle = zoomPage.locator('#documenter-sidebar-button');
      await toggle.click();
      await zoomPage.locator('.site-nav-bottom a').scrollIntoViewIfNeeded();
      await zoomPage.keyboard.press('Escape');
      assert.equal(await toggle.evaluate(el => el === document.activeElement), true);
      await zoomPage.waitForFunction(() => {
        const menu = document.getElementById('site-navigation');
        return menu.getBoundingClientRect().right <= 0 || getComputedStyle(menu).visibility === 'hidden';
      });
      const cdp = await zoomContext.newCDPSession(zoomPage);
      const shot = await cdp.send('Page.captureScreenshot', { captureBeyondViewport: false });
      await fs.writeFile(path.join(output, `zoom-${factor * 100}.png`), Buffer.from(shot.data, 'base64'));
      await cdp.detach();
    }
  } finally { await zoomContext.close(); }
  console.log(JSON.stringify({ pages: checked, checks: ['desktop', 'mobile', 'keyboard',
    'article outline', 'search labels', 'dark theme', 'no JavaScript', '150% and 200% browser zoom'],
    screenshots: output }, null, 2));
} finally {
  if (browser) await browser.close();
  await new Promise(resolve => server.close(resolve));
}
