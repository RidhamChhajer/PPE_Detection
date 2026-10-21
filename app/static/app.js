'use strict';
const form = document.querySelector('#upload-form');
const input = document.querySelector('#video-file');
const button = document.querySelector('#process-button');
const dropzone = document.querySelector('#dropzone');
const error = document.querySelector('#error');
const token = document.querySelector('meta[name="csrf-token"]').content;
let running = false;
function showError(message) { error.textContent = message; error.hidden = false; }
function fileError(file) {
  if (!file || !file.size) return 'Choose a non-empty video.';
  if (!/\.(mp4|mov|mkv|avi|webm|m4v)$/i.test(file.name)) return 'Choose an MP4, MOV, MKV, AVI, WebM, or M4V video.';
  return file.size > 512*1024*1024 ? 'Choose a video smaller than 512 MB.' : '';
}
function selected() {
  const file = input.files[0];
  error.hidden = true;
  button.disabled = !file || running;
  document.querySelector('#file-label').textContent = file ? file.name : 'Choose a video';
  document.querySelector('#file-detail').textContent = file ? `${(file.size/1024/1024).toFixed(1)} MB · Ready to process` : 'or drag and drop it here';
  const message = file ? fileError(file) : '';
  input.setAttribute('aria-invalid', String(!!message));
  if (message) showError(message);
  syncControls();
}
input.addEventListener('change', selected);
for (const name of ['dragenter', 'dragover']) dropzone.addEventListener(name, event => { event.preventDefault(); if (!running) dropzone.classList.add('drag'); });
for (const name of ['dragleave', 'drop']) dropzone.addEventListener(name, event => { event.preventDefault(); dropzone.classList.remove('drag'); });
dropzone.addEventListener('drop', event => {
  if (running || liveActive || liveBusy) return;
  if (event.dataTransfer.files.length !== 1) { showError('Select one video at a time.'); return; }
  input.files = event.dataTransfer.files; selected();
});
function setProgress(job) {
  document.querySelector('#progress-region').hidden = false;
  document.querySelector('#stage').textContent = job.stage;
  document.querySelector('#percentage').textContent = `${job.progress}%`;
  document.querySelector('#progress').value = job.progress;
  document.querySelector('#progress-detail').textContent = job.frames_total ? `${job.frames_done} / ${job.frames_total} frames` : 'Keep this page open to see the result.';
}
async function responseData(response) {
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Request failed. Please try again.');
  return data;
}
async function follow(id) {
  running = true; syncControls();
  let failures = 0;
  try {
    while (true) {
      let job;
      try { job = await responseData(await fetch(`/api/jobs/${id}`, {cache:'no-store'})); failures = 0; }
      catch (failure) { if (++failures >= 4) throw failure; await new Promise(resolve => setTimeout(resolve, 1500)); continue; }
      setProgress(job);
      if (job.state === 'failed') throw new Error(job.error);
      if (job.state === 'complete') {
        const player = document.querySelector('#output-video');
        player.src = `/api/jobs/${id}/video`;
        player.hidden = false;
        document.querySelector('#empty-output').hidden = true;
        document.querySelector('#result-footer').hidden = false;
        document.querySelector('#result-badge').textContent = 'COMPLETE';
        document.querySelector('#download').href = `/api/jobs/${id}/download`;
        document.querySelector('#result-summary').textContent = `${job.frames} frames · ${job.width} × ${job.height} · ${job.fps.toFixed(2)} FPS · ${job.duration.toFixed(2)} s`;
        document.querySelector('#progress-detail').textContent = 'Complete video verified. Ready to play or download.';
        break;
      }
      await new Promise(resolve => setTimeout(resolve, 750));
    }
  } catch (failure) { showError(failure.message); document.querySelector('#result-badge').textContent = 'FAILED'; }
  finally { running = false; syncControls(); }
}
form.addEventListener('submit', async event => {
  event.preventDefault();
  if (running || liveActive || liveBusy || !input.files.length) return;
  if (fileError(input.files[0])) { showError(fileError(input.files[0])); input.focus(); return; }
  running = true; syncControls(); error.hidden = true;
  document.querySelector('#result-badge').textContent = 'PROCESSING';
  const player = document.querySelector('#output-video');
  player.pause(); player.removeAttribute('src'); player.load(); player.hidden = true;
  document.querySelector('#result-footer').hidden = true;
  document.querySelector('#empty-output').hidden = false;
  setProgress({stage:'Uploading video', progress:0});
  const body = new FormData(); body.append('video', input.files[0]);
  try {
    const result = await responseData(await fetch('/api/jobs', {method:'POST',headers:{'X-CSRF-Token':token},body}));
    sessionStorage.setItem('ppe-job', result.id);
    await follow(result.id);
  } catch (failure) { showError(failure.message); document.querySelector('#result-badge').textContent = 'FAILED'; running = false; syncControls(); }
});
const uploadTab = document.querySelector('#upload-tab');
const liveTab = document.querySelector('#live-tab');
const uploadWorkspace = document.querySelector('#upload-workspace');
const liveWorkspace = document.querySelector('#live-workspace');
const selects = Object.fromEntries(['face', 'feet'].map(role => [role, document.querySelector(`#${role}-camera`)]));
const refreshCameras = document.querySelector('#refresh-cameras');
const startCamera = document.querySelector('#start-camera');
const stopCamera = document.querySelector('#stop-camera');
const cameraStatus = document.querySelector('#camera-status');
const cameraError = document.querySelector('#camera-error');
let liveActive = false;
let liveBusy = false;
let searching = false;
let camerasLoaded = false;

function syncControls() {
  input.disabled = running || liveActive || liveBusy;
  button.disabled = input.disabled || !input.files.length || !!fileError(input.files[0]);
  for (const select of Object.values(selects)) select.disabled = liveActive || liveBusy || searching || running;
  refreshCameras.disabled = liveActive || liveBusy || searching || running;
  startCamera.hidden = liveActive;
  stopCamera.hidden = !liveActive;
  startCamera.disabled = liveBusy || searching || running || (!selects.face.value && !selects.feet.value);
  stopCamera.disabled = liveBusy;
}
function selectMode(mode) {
  const live = mode === 'live';
  for (const [tab, selected] of [[uploadTab, !live], [liveTab, live]]) {
    tab.setAttribute('aria-selected', String(selected));
    tab.tabIndex = selected ? 0 : -1;
  }
  uploadWorkspace.hidden = live;
  liveWorkspace.hidden = !live;
  if (live && !camerasLoaded && !liveActive && !liveBusy && !running) loadCameras();
}
for (const [tab, mode] of [[uploadTab, 'upload'], [liveTab, 'live']]) {
  tab.addEventListener('click', () => selectMode(mode));
  tab.addEventListener('keydown', event => {
    if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
      event.preventDefault();
      const next = event.key === 'Home' ? uploadTab : event.key === 'End' ? liveTab : tab === uploadTab ? liveTab : uploadTab;
      selectMode(next === liveTab ? 'live' : 'upload');
      next.focus();
    }
  });
}
liveTab.tabIndex = -1;
function showCameraError(message) { cameraError.textContent = message; cameraError.hidden = false; }
function updateLive(status) {
  liveActive = status.active;
  document.querySelector('#live-badge').textContent = status.state.toUpperCase();
  cameraStatus.textContent = status.active ? 'Live detection is active. Each preview shows its own status.' :
    'Select either view, use the same camera for both, or assign two cameras. Choose Not used to disable a view.';
  if (status.active) {
    for (const select of Object.values(selects)) select.value = '';
    for (const feed of Object.values(status.feeds)) {
      for (const role of feed.roles || []) {
        if (![...selects[role].options].some(option => option.value === String(feed.camera))) {
          selects[role].add(new Option(`Camera ${feed.camera}`, String(feed.camera)));
        }
        selects[role].value = String(feed.camera);
      }
    }
  }
  document.querySelector('.live-feeds').classList.toggle('single-feed', status.active && Object.keys(status.feeds).length === 1);
  for (const role of ['face', 'feet']) {
    const feed = status.feeds[role];
    const preview = document.querySelector(`#${role}-preview`);
    const empty = document.querySelector(`#${role}-empty`);
    const metrics = document.querySelector(`#${role}-metrics`);
    const roles = feed?.roles || [role];
    const card = document.querySelector(`#${role}-card`);
    card.hidden = status.active && !feed;
    document.querySelector(`#${role}-title`).textContent = roles.length === 2 ? 'Shared camera · face and feet' : `${role === 'face' ? 'Face' : 'Feet'} camera`;
    for (const tag of card.querySelectorAll('[data-role]')) tag.hidden = !roles.includes(tag.dataset.role);
    const ready = status.active && feed?.state === 'running';
    document.querySelector(`#${role}-badge`).textContent = (feed?.state || 'stopped').toUpperCase();
    if (ready) {
      if (preview.hidden || preview.dataset.generation !== String(status.generation)) {
        preview.src = `/api/live/${role}/stream?generation=${status.generation}`;
        preview.dataset.generation = String(status.generation);
      }
      preview.hidden = false;
      empty.hidden = true;
      metrics.textContent = `${feed.fps.toFixed(1)} processed FPS · ${feed.latency_ms.toFixed(0)} ms capture-to-preview encoding · ${feed.frame_age_ms.toFixed(0)} ms frame age`;
    } else {
      preview.removeAttribute('src');
      preview.hidden = true;
      empty.hidden = false;
      empty.textContent = feed?.error || (feed?.state === 'stale' ? 'No fresh frame. Check the camera connection.' :
        status.active ? 'Waiting for a camera frame...' : 'Camera is stopped.');
      metrics.textContent = feed?.state === 'failed' ? 'Stop cameras, reconnect, and start again.' : 'Waiting for camera.';
    }
  }
  if (status.error) showCameraError(status.error);
  syncControls();
}
async function loadCameras() {
  if (searching || liveActive || liveBusy || running) return;
  searching = true;
  cameraError.hidden = true;
  cameraStatus.textContent = 'Searching for connected cameras...';
  syncControls();
  try {
    const cameras = await responseData(await fetch('/api/cameras', {cache:'no-store', signal:AbortSignal.timeout(30000)}));
    for (const select of Object.values(selects)) {
      const value = select.value;
      select.replaceChildren(new Option(cameras.length ? 'Not used' : 'No cameras found', ''));
      for (const camera of cameras) select.add(new Option(camera.name, String(camera.index)));
      select.value = cameras.some(camera => String(camera.index) === value) ? value : '';
    }
    camerasLoaded = true;
    cameraStatus.textContent = cameras.length ? 'Select either view, share one camera, or assign two different cameras.' :
      'No cameras found. Connect a camera, check Windows camera permissions, then refresh.';
  } catch (failure) { showCameraError(failure.message); }
  finally { searching = false; syncControls(); }
}
refreshCameras.addEventListener('click', loadCameras);
for (const select of Object.values(selects)) select.addEventListener('change', () => {
  cameraError.hidden = true;
  syncControls();
});
async function pollCamera() {
  try {
    const status = await responseData(await fetch('/api/live/status', {cache:'no-store', signal:AbortSignal.timeout(10000)}));
    if (!liveBusy && !searching) updateLive(status);
  } catch (failure) {
    showCameraError('Connection to the local server was lost. Retrying...');
    for (const role of ['face', 'feet']) {
      document.querySelector(`#${role}-preview`).hidden = true;
      document.querySelector(`#${role}-empty`).hidden = false;
      document.querySelector(`#${role}-empty`).textContent = 'Connection lost. Waiting for the server...';
    }
  } finally { setTimeout(pollCamera, 1000); }
}
async function cameraAction(action) {
  if (liveBusy || searching || running) return;
  if (action === 'start' && startCamera.disabled) return;
  liveBusy = true;
  cameraError.hidden = true;
  cameraStatus.textContent = action === 'start' ? 'Opening selected cameras...' : 'Stopping cameras...';
  syncControls();
  try {
    const options = {method:'POST', headers:{'X-CSRF-Token':token}, signal:AbortSignal.timeout(30000)};
    if (action === 'start') {
      options.headers['Content-Type'] = 'application/json';
      options.body = JSON.stringify(Object.fromEntries(Object.entries(selects).map(([role, select]) =>
        [role, select.value === '' ? null : Number(select.value)])));
    }
    updateLive(await responseData(await fetch(`/api/live/${action}`, options)));
  } catch (failure) { showCameraError(failure.message); }
  finally { liveBusy = false; syncControls(); }
}
startCamera.addEventListener('click', () => cameraAction('start'));
stopCamera.addEventListener('click', () => cameraAction('stop'));
for (const role of ['face', 'feet']) document.querySelector(`#${role}-preview`).addEventListener('error', event => {
  event.target.hidden = true;
  document.querySelector(`#${role}-empty`).hidden = false;
  document.querySelector(`#${role}-empty`).textContent = 'Preview interrupted. Retrying...';
});
syncControls();
pollCamera();
const previous = sessionStorage.getItem('ppe-job');
if (previous) follow(previous);
