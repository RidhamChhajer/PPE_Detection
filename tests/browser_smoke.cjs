// Optional browser check: Playwright is a development tool, not an app dependency.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs/promises');

(async () => {
  const browser = await chromium.launch({headless: true,
    executablePath: process.env.PPE_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe'});
  try {
    const page = await browser.newPage({viewport: {width: 1280, height: 900}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    let active = false;
    let disconnected = false;
    let assignment = {face: 0, feet: 1};
    const status = () => ({state: active ? 'running' : 'stopped', active, generation: 1,
      feeds: Object.fromEntries(Object.entries(assignment).filter(([role, camera]) => camera !== null &&
        !(role === 'feet' && assignment.face === camera)).map(([role, camera]) => [role, {
        roles: assignment.face !== null && assignment.face === assignment.feet ? ['face', 'feet'] : [role],
        camera, state: !active ? 'stopped' : disconnected && role === 'face' ? 'failed' : 'running',
        error: disconnected && role === 'face' ? 'Camera disconnected.' : null,
        fps: 10, latency_ms: 100, frame_age_ms: 110}]))});
    await page.route('**/api/cameras', route => route.fulfill({json: [
      {index: 0, name: 'Camera 0'}, {index: 1, name: 'Camera 1'}]}));
    await page.route('**/api/live/**', async route => {
      const url = route.request().url();
      if (url.includes('/start')) {
        assignment = route.request().postDataJSON();
        active = true;
      }
      if (url.includes('/stop')) active = false;
      if (url.includes('/stream')) return route.fulfill({contentType: 'image/png',
        body: Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aV3sAAAAASUVORK5CYII=', 'base64')});
      return route.fulfill({json: status()});
    });
    await page.goto(process.argv[2] || 'http://127.0.0.1:8766');
    await page.locator('#upload-tab').focus();
    await page.keyboard.press('ArrowRight');
    await page.waitForFunction(() => document.querySelector('#face-camera').options.length === 3);
    await page.locator('#face-camera').selectOption('0');
    await page.locator('#feet-camera').selectOption('0');
    assert(await page.locator('#start-camera').isEnabled());
    await page.locator('#start-camera').click();
    await page.locator('#face-preview').waitFor({state: 'visible'});
    assert(await page.locator('#feet-card').isHidden());
    assert.match(await page.locator('#face-title').innerText(), /Shared/);
    assert.equal(await page.locator('#face-card .tag:visible').count(), 2);
    await page.locator('#stop-camera').click();
    for (const [face, feet, visible, hidden] of [['0', '', 'face', 'feet'], ['', '1', 'feet', 'face']]) {
      await page.locator('#face-camera').selectOption(face);
      await page.locator('#feet-camera').selectOption(feet);
      await page.locator('#start-camera').click();
      await page.locator(`#${visible}-preview`).waitFor({state: 'visible'});
      assert(await page.locator(`#${hidden}-card`).isHidden());
      assert.equal(assignment[hidden], null);
      await page.locator('#stop-camera').click();
    }
    await page.locator('#face-camera').selectOption('0');
    await page.locator('#feet-camera').selectOption('1');
    await page.locator('#start-camera').click();
    await page.locator('#face-preview').waitFor({state: 'visible'});
    await page.locator('#feet-preview').waitFor({state: 'visible'});
    assert(await page.locator('#video-file').isDisabled());
    const out = path.resolve('artifacts/browser');
    await fs.mkdir(out, {recursive: true});
    await page.screenshot({path: path.join(out, 'two-cameras-desktop.png'), fullPage: true});
    disconnected = true;
    await page.waitForFunction(() => document.querySelector('#face-badge').textContent === 'FAILED');
    assert(await page.locator('#face-preview').isHidden());
    assert(await page.locator('#feet-preview').isVisible());
    await page.locator('#stop-camera').click();
    await page.setViewportSize({width: 390, height: 844});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({path: path.join(out, 'two-cameras-mobile.png'), fullPage: true});
    await page.locator('#upload-tab').click();
    await page.locator('#video-file').setInputFiles({name: 'invalid.txt', mimeType: 'text/plain', buffer: Buffer.from('x')});
    assert(await page.locator('#process-button').isDisabled());
    assert(await page.locator('#error').isVisible());
    if (process.argv[3]) {
      await page.locator('#video-file').setInputFiles(path.resolve(process.argv[3]));
      await page.locator('#process-button').click();
      await page.waitForFunction(() => ['COMPLETE', 'FAILED'].includes(document.querySelector('#result-badge').textContent), null, {timeout: 240000});
      assert.equal(await page.locator('#result-badge').innerText(), 'COMPLETE');
      assert(await page.locator('#output-video').isVisible());
      const download = await Promise.all([page.waitForEvent('download'), page.locator('#download').click()]);
      assert.equal(await download[0].failure(), null);
    }
    assert.deepEqual(errors, []);
    console.log('PASS: tabs, all four camera arrangements, two previews, disconnect, stop, narrow layout, invalid upload' +
      (process.argv[3] ? ', upload/playback/download.' : '.'));
  } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode = 1;});
