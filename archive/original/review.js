// Embedded file decisions are durable defaults; browser edits are explicit overrides.
const reviewData = JSON.parse(document.querySelector('#review-data').textContent);
const runHash = reviewData.run_sha256;
const key = 'pencil-baseline-review-' + runHash.slice(0, 16);
const cards = [...document.querySelectorAll('article')];
const defaults = Object.fromEntries(reviewData.decisions.map(row => [row.id, row]));
let saved = {};
try { saved = JSON.parse(localStorage.getItem(key) || '{}'); } catch (_) {}
if (!saved || typeof saved !== 'object' || Array.isArray(saved)) saved = {};
function current(c) {
  const base = defaults[c.dataset.id];
  const row = {...base, verdict: c.querySelector('.verdict').value,
    preference: c.querySelector('.preference').value, notes: c.querySelector('.review-note').value};
  row.winner = {reference_better: 'ChatGPT', baseline_better: reviewData.model.id, similar: 'tie'}[row.preference] || null;
  if (saved[row.id]?.fields?.verdict) row.verdict_basis = 'User-selected in the review page.';
  if (saved[row.id]?.fields?.preference) row.preference_basis = 'User-selected in the review page.';
  if (saved[row.id]?.fields?.notes) row.source = 'browser_edit_of_saved_feedback';
  return row;
}
function progress() {
  const reviewed = cards.filter(c => c.querySelector('.verdict').value !== 'pending').length;
  const winners = cards.filter(c => c.querySelector('.preference').value !== 'not_rated').length;
  document.querySelector('#review-progress').textContent = reviewed + ' / ' + cards.length + ' verdicts · ' + winners + ' comparisons';
}
function persist() {
  try { localStorage.setItem(key, JSON.stringify(saved)); }
  catch (_) { document.querySelector('#status-note').textContent = 'Browser storage unavailable. Export your changes.'; }
  progress();
}
function filter() {
  const kind = document.querySelector('#kind').value, status = document.querySelector('#status').value;
  for (const c of cards) c.hidden = (kind !== 'all' && c.dataset.kind !== kind) ||
    (status === 'pending' && c.querySelector('.verdict').value !== 'pending') ||
    (status === 'reviewed' && c.querySelector('.verdict').value === 'pending');
}
for (const c of cards) {
  const id = c.dataset.id, previous = saved[id] || {}, base = defaults[id];
  for (const field of ['verdict', 'preference', 'notes']) {
    const control = c.querySelector(field === 'notes' ? '.review-note' : '.' + field);
    // Empty legacy records must not hide newly embedded dictated feedback.
    const meaningful = previous.fields ? previous.fields[field] :
      field === 'notes' ? Boolean(previous.notes) : previous[field] && !['pending', 'not_rated'].includes(previous[field]);
    control.value = meaningful ? previous[field] : base[field];
    control.addEventListener(field === 'notes' ? 'input' : 'change', () => {
      saved[id] = {...(saved[id] || {}), [field]: control.value,
        fields: {...(saved[id]?.fields || {}), [field]: true}};
      if (field === 'verdict') c.querySelector('.verdict-basis').textContent = 'User-selected in the review page.';
      persist(); filter();
    });
  }
  if (previous.fields?.verdict) c.querySelector('.verdict-basis').textContent = 'User-selected in the review page.';
}
document.querySelector('#kind').onchange = filter;
document.querySelector('#status').onchange = filter;
document.querySelector('#export').onclick = () => {
  const data = {...reviewData, decisions: cards.map(current)};
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'}));
  const a = document.createElement('a'); a.href = url; a.download = 'pencil-baseline-review-decisions.json'; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
document.querySelector('#file').onchange = async event => {
  try {
    const data = JSON.parse(await event.target.files[0].text());
    if (data.run_sha256 !== runHash || !Array.isArray(data.decisions)) throw Error('This review belongs to another run');
    // Validate the entire import before applying any row.
    const seen = new Set();
    for (const row of data.decisions) {
      if (!defaults[row.id] || seen.has(row.id) ||
          !['pending','good_baseline','needs_improvement','reject'].includes(row.verdict) ||
          !['not_rated','reference_better','similar','baseline_better'].includes(row.preference) ||
          typeof row.notes !== 'string') throw Error('Invalid review record: ' + row.id);
      seen.add(row.id);
    }
    for (const row of data.decisions) {
      const c = cards.find(c => c.dataset.id === row.id);
      for (const field of ['verdict','preference','notes'])
        c.querySelector(field === 'notes' ? '.review-note' : '.' + field).value = row[field];
      saved[row.id] = {...row, fields: {verdict:true,preference:true,notes:true}};
      c.querySelector('.verdict-basis').textContent = 'User-selected in an imported review.';
    }
    persist(); filter();
  } catch (error) { alert(error.message); }
};
progress();
