'use strict';
const form = document.querySelector('#upload-form');
const input = document.querySelector('#video-file');
const button = document.querySelector('#process-button');
const dropzone = document.querySelector('#dropzone');
const error = document.querySelector('#error');
const token = document.querySelector('meta[name="csrf-token"]').content;
let running = false;
function showError(message) { error.textContent = message; error.hidden = false; }
function selected() {
  const file = input.files[0];
  error.hidden = true;
  button.disabled = !file || running;
  document.querySelector('#file-label').textContent = file ? file.name : 'Choose a video';
  document.querySelector('#file-detail').textContent = file ? `${(file.size/1024/1024).toFixed(1)} MB · Ready to process` : 'or drag and drop it here';
  if (file && file.size > 512*1024*1024) { showError('Choose a video smaller than 512 MB.'); button.disabled = true; }
}
input.addEventListener('change', selected);
for (const name of ['dragenter', 'dragover']) dropzone.addEventListener(name, event => { event.preventDefault(); if (!running) dropzone.classList.add('drag'); });
for (const name of ['dragleave', 'drop']) dropzone.addEventListener(name, event => { event.preventDefault(); dropzone.classList.remove('drag'); });
dropzone.addEventListener('drop', event => {
  if (running) return;
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
  running = true; button.disabled = true; input.disabled = true;
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
  finally { running = false; input.disabled = false; button.disabled = !input.files.length; }
}
form.addEventListener('submit', async event => {
  event.preventDefault();
  if (running || !input.files.length) return;
  running = true; button.disabled = true; input.disabled = true; error.hidden = true;
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
  } catch (failure) { showError(failure.message); document.querySelector('#result-badge').textContent = 'FAILED'; running = false; input.disabled = false; button.disabled = false; }
});
const previous = sessionStorage.getItem('ppe-job');
if (previous) follow(previous);
