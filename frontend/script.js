const API_BASE = "http://127.0.0.1:5000";

const fileInput   = document.getElementById('fileInput');
const preview     = document.getElementById('preview');
const emptyState  = document.getElementById('emptyState');
const progress    = document.getElementById('progress');
const progressBar = document.getElementById('progressBar');
const predictBtn  = document.getElementById('predictBtn');
const genState    = document.getElementById('genState');
const downloadBtn = document.getElementById('downloadBtn');
const dropZone    = document.getElementById('dropZone');
const browseLink  = document.getElementById('browseLink');

let lastPdfBlobUrl = null;

function resetUI() {
  if (preview) { preview.style.display = 'none'; preview.src = ''; }
  if (emptyState) emptyState.style.display = 'block';
  if (predictBtn) predictBtn.disabled = true;
  if (genState) genState.style.display = 'none';
  if (downloadBtn) { downloadBtn.style.display = 'none'; downloadBtn.removeAttribute('data-url'); }
  if (progress) progress.style.display = 'none';
  if (progressBar) progressBar.style.width = '0%';
  if (lastPdfBlobUrl) { URL.revokeObjectURL(lastPdfBlobUrl); lastPdfBlobUrl = null; }
}
resetUI();

function isJPG(file) {
  if (!file) return false;
  return file.type === 'image/jpeg' || /\.jpe?g$/i.test(file.name || '');
}

function handleFile(file) {
  if (!file) return;
  if (!isJPG(file)) { alert('Please select a JPG/JPEG file.'); resetUI(); return; }

  progress.style.display = 'block'; progressBar.style.width = '0%';
  const reader = new FileReader();
  reader.onprogress = e => {
    if (e.lengthComputable) progressBar.style.width = Math.min(100, Math.round((e.loaded/e.total)*100)) + '%';
  };
  reader.onload = e => {
    preview.src = e.target.result;
    preview.onload = () => {
      emptyState.style.display = 'none';
      preview.style.display = 'block';
      progressBar.style.width = '100%';
      setTimeout(() => progress.style.display = 'none', 200);
      predictBtn.disabled = false;
      genState.style.display = 'none';
      downloadBtn.style.display = 'none';
    };
  };
  reader.onerror = () => { alert('Failed to load the image.'); resetUI(); };
  reader.readAsDataURL(file);
}

fileInput?.addEventListener('change', () => {
  const file = fileInput.files && fileInput.files[0];
  handleFile(file);
});

browseLink?.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); fileInput?.click(); });

if (dropZone) {
  dropZone.addEventListener('click', (e) => { if (!e.target.closest('.upload-btn')) fileInput?.click(); });
  ['dragenter','dragover'].forEach(evt => dropZone.addEventListener(evt, (e)=>{ e.preventDefault(); e.stopPropagation(); dropZone.classList.add('dragover'); }));
  ['dragleave','drop'].forEach(evt => dropZone.addEventListener(evt, (e)=>{ e.preventDefault(); e.stopPropagation(); dropZone.classList.remove('dragover'); }));
  dropZone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) {
      const dt = new DataTransfer(); dt.items.add(file); fileInput.files = dt.files;
      handleFile(file);
    }
  });
}

predictBtn?.addEventListener('click', async () => {
  const file = fileInput?.files && fileInput.files[0];
  if (!file) { alert('Please select an image first.'); return; }

  predictBtn.disabled = true; const oldText = predictBtn.textContent; predictBtn.textContent = 'Predicting…';
  try {
    const form = new FormData();
    form.append('image', file);
    form.append('name', document.getElementById('pname')?.value || '');
    form.append('age', document.getElementById('page')?.value || '');
    form.append('gender', document.getElementById('pgender')?.value || '');

    const resp = await fetch(`${API_BASE}/predict`, { method: 'POST', body: form });
    if (!resp.ok) {
      const maybeJson = await resp.json().catch(()=>null);
      const msg = (maybeJson && maybeJson.error) ? maybeJson.error : `Server error (${resp.status})`;
      throw new Error(msg);
    }

    const blob = await resp.blob();
    if (lastPdfBlobUrl) URL.revokeObjectURL(lastPdfBlobUrl);
    lastPdfBlobUrl = URL.createObjectURL(blob);

    genState.style.display = 'flex';
    downloadBtn.style.display = 'inline-block';
    downloadBtn.setAttribute('data-url', lastPdfBlobUrl);
  } catch (err) {
    alert(`Prediction failed: ${err.message}`);
  } finally {
    predictBtn.textContent = oldText;
    predictBtn.disabled = false;
  }
});

downloadBtn?.addEventListener('click', () => {
  const url = downloadBtn.getAttribute('data-url'); if (!url) return;
  const a = document.createElement('a'); a.href = url; a.download = 'Brain_Tumor_Report.pdf'; document.body.appendChild(a); a.click(); a.remove();
});
