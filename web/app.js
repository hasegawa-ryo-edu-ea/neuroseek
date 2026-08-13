const strings = {
  en: {subtitle:'LEARNED KNOWLEDGE GRAPH SEARCH',isolated:'TRAINER ISOLATED',navSearch:'01 / SEARCH',navPolicy:'02 / POLICY',navAbout:'03 / METHOD',evidence:'LOCAL EVIDENCE',awaiting:'AWAITING QUERY',awaitingDetail:'Enter a word, Q-ID, or place to inspect this Jetson\'s local graph.',facts:'OBSERVED FACTS',factsEmpty:'The selected entity\'s real outgoing graph edges will appear here.',boundary:'EVIDENCE BOUNDARY',boundaryDetail:'Names are resolved through Wikidata. Only edges shown here are stored in the local read-only snapshot.',theater:'GRAPH NAVIGATION THEATER',startTitle:'ASK THE GRAPH',selectedPath:'selected local edge',alternatives:'alternative local edge',graphHint:'DRAG TO PAN · WHEEL TO ZOOM · CLICK A NODE',queryLabel:'QUERY INPUT — natural language or Wikidata Q-ID',searchButton:'EXPLORE',relationLabel:'RELATION FILTER',filterButton:'FILTER',story:'THE STORY',asked:'ASKED',askedDetail:'Your term becomes an entity candidate.',found:'FOUND',foundDetail:'Choose a candidate available in the local snapshot.',explored:'EXPLORED',exploredDetail:'Follow real outgoing graph edges.',verified:'BOUNDARY',verifiedDetail:'The graph view never claims online context as local evidence.',policyEyebrow:'LEARNED POLICY DEMONSTRATION',policyTitle:'A policy chooses the graph program.',policyDetail:'This runs the immutable presentation checkpoint against a held-out graph task. The reference answer is not passed to the policy.',runPolicy:'RUN HELD-OUT TASK',policyReady:'Ready. This action is CPU-only and does not interfere with training.',methodEyebrow:'WHY NEUROSEEK',methodTitle:'Learn → Explore → Prove',learn:'LEARN',learnDetail:'A learned policy selects graph operations.',explore:'EXPLORE',exploreDetail:'The program traverses the memory-mapped local graph.',prove:'PROVE',proveDetail:'A separate validator accepts reconstructable evidence only.',footer:'STATUS: READY'},
  ja: {subtitle:'学習済み知識グラフ探索',isolated:'学習器から分離',navSearch:'01 / 検索',navPolicy:'02 / 方策',navAbout:'03 / 手法',evidence:'ローカル証拠',awaiting:'検索待機中',awaitingDetail:'単語・Q-ID・場所を入力して、このJetson内のグラフを調べます。',facts:'観測した事実',factsEmpty:'選んだ対象から出る実際のローカルグラフエッジをここに表示します。',boundary:'証拠の境界',boundaryDetail:'名称はWikidataで解決します。ここに表示するエッジだけがローカル読み取り専用スナップショットの証拠です。',theater:'グラフ探索シアター',startTitle:'グラフに問いかける',selectedPath:'選択中のローカルエッジ',alternatives:'他のローカルエッジ',graphHint:'ドラッグで移動 · ホイールで拡大 · ノードを選択',queryLabel:'クエリ入力 — 自然言語または Wikidata Q-ID',searchButton:'探索',relationLabel:'関係フィルタ',filterButton:'絞り込み',story:'ストーリー',asked:'質問',askedDetail:'入力語を候補エンティティへ解決します。',found:'発見',foundDetail:'ローカルスナップショットで使える候補を選びます。',explored:'探索',exploredDetail:'実際の出力グラフエッジをたどります。',verified:'境界',verifiedDetail:'オンラインの説明をローカル証拠として扱いません。',policyEyebrow:'学習済み方策デモ',policyTitle:'方策がグラフプログラムを選ぶ。',policyDetail:'不変の発表用チェックポイントを未使用グラフ課題で実行します。正解は方策に渡しません。',runPolicy:'未使用課題を実行',policyReady:'準備完了。この操作はCPU専用で、学習を妨害しません。',methodEyebrow:'NEUROSEEKの特徴',methodTitle:'学習 → 探索 → 証明',learn:'学習',learnDetail:'学習済み方策がグラフ演算を選びます。',explore:'探索',exploreDetail:'プログラムがメモリマップされたローカルグラフをたどります。',prove:'証明',proveDetail:'別系統の検証器が再構築可能な証拠だけを受理します。',footer:'状態: 準備完了'}
};

let lang = 'ja';
let selected = null;
let graphData = null;
let policyData = null;
let graphMode = 'local';
const $ = selector => document.querySelector(selector);
const scene = { x: 0, y: 0, zoom: 1, drag: null, selected: 0, reveal: 0, animation: 0, nodes: [], size: { w: 0, h: 0 } };
const t = key => strings[lang][key] || key;
const notice = value => { $('#notice').textContent = value; };
const stage = value => { $('#story-state').textContent = value; };

function applyLanguage() {
  document.documentElement.lang = lang;
  document.querySelectorAll('[data-i18n]').forEach(node => { node.textContent = t(node.dataset.i18n); });
  $('#language').textContent = lang === 'ja' ? 'EN' : '日本語';
  if (graphData) renderGraph(graphData, false);
}

async function api(path) {
  const response = await fetch(path);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Request failed');
  return data;
}

function summary(data) {
  const relationCount = new Set(data.edges.map(edge => edge.relation.id)).size;
  $('#graph-summary').innerHTML = [
    `<span>NODE <b>${data.root.id}</b></span>`,
    `<span>LOCAL EDGES <b>${data.edges.length}</b></span>`,
    `<span>RELATIONS <b>${relationCount}</b></span>`,
    `<span>SECOND HOP <b>${data.second ? data.second.edges.length : '…'}</b></span>`,
    `<span>SNAPSHOT <b>${data.graph.entities.toLocaleString()}</b></span>`,
    `<span id="view-stat">VIEW <b>${scene.zoom.toFixed(2)}×</b></span>`,
  ].join('');
}

function updateSelection() {
  if (!graphData || !graphData.edges[scene.selected]) return;
  const edge = graphData.edges[scene.selected];
  $('#selected-edge').textContent = `${String(scene.selected + 1).padStart(2, '0')}  ${edge.relation.label} → ${edge.target.label}`;
  [...$('#facts').children].forEach((node, index) => node.classList.toggle('selected-fact', index === scene.selected));
}

async function expandSecond(index) {
  if (!graphData || !graphData.edges[index]) return;
  const source = graphData;
  const entity = source.edges[index].target.id;
  try {
    const expanded = await api(`/api/graph?entity=${encodeURIComponent(entity)}&lang=${lang}`);
    if (graphData !== source) return;
    source.second = { from: index, root: expanded.root, edges: expanded.edges.slice(0, 8) };
    scene.revealSecond = performance.now();
    summary(source);
    notice(lang === 'ja' ? `二段目: ${expanded.root.label} から ${source.second.edges.length} 本の実エッジを展開しました。` : `Second hop: expanded ${source.second.edges.length} real edges from ${expanded.root.label}.`);
  } catch (error) { notice(error.message); }
}

function viewStat() {
  const stat = $('#view-stat');
  if (stat) stat.innerHTML = `VIEW <b>${scene.zoom.toFixed(2)}×</b>`;
}

async function search() {
  const query = $('#query').value.trim();
  if (!query) return;
  notice(lang === 'ja' ? '候補をローカルグラフに照合しています…' : 'Resolving candidates against the local graph…');
  stage('RESOLVING');
  try {
    const data = await api(`/api/search?q=${encodeURIComponent(query)}&lang=${lang}`);
    const row = $('#candidate-row');
    row.innerHTML = '';
    data.candidates.forEach((candidate, index) => {
      const button = document.createElement('button');
      button.className = `candidate ${candidate.local ? 'local' : 'external'} reveal-item`;
      button.style.setProperty('--delay', `${90 + index * 65}ms`);
      button.textContent = `${candidate.local ? '●' : '○'} ${candidate.label} [${candidate.identifier}]`;
      button.title = candidate.description || '';
      button.onclick = () => candidate.local ? select(candidate) : notice(lang === 'ja' ? 'この候補はWikidataの説明にはありますが、ローカル証拠にはありません。' : 'This candidate is Wikidata context, not local evidence.');
      row.appendChild(button);
    });
    const first = data.candidates.find(candidate => candidate.local);
    if (first) await select(first);
    else { notice(lang === 'ja' ? 'ローカルにある候補はありません。外部候補は証拠としては扱いません。' : 'No candidate is in the local snapshot. External context is not presented as evidence.'); stage('BOUNDARY'); }
  } catch (error) { notice(error.message); stage('ERROR'); }
}

async function select(candidate) { selected = candidate; $('#relation').value = ''; await loadGraph(); }

async function loadGraph() {
  if (!selected) return;
  stage('EXPLORING');
  notice(lang === 'ja' ? 'ローカルグラフの実エッジを展開しています…' : 'Unfolding real local graph edges…');
  try {
    graphMode = 'local';
    policyData = null;
    graphData = await api(`/api/graph?entity=${encodeURIComponent(selected.identifier)}&relation=${encodeURIComponent($('#relation').value)}&lang=${lang}`);
    renderGraph(graphData, true);
    stage('EVIDENCE READY');
    notice(lang === 'ja' ? `${graphData.edges.length} 本のローカルエッジを段階的に可視化しました。` : `Unfolding ${graphData.edges.length} local graph edges.`);
  } catch (error) { notice(error.message); stage('BOUNDARY'); }
}

function renderGraph(data, animate = true) {
  $('#graph-title').textContent = `${data.root.label}  [${data.root.id}]`;
  $('#entity-card').className = 'entity-card result-arrive';
  $('#entity-card').innerHTML = `<strong>${data.root.label}</strong><p>${data.root.id} · ${t('evidence')}</p>${data.relation_filter ? `<p>${t('relationLabel')}: ${data.relation_filter.label} [${data.relation_filter.identifier}]</p>` : ''}`;
  $('#facts').innerHTML = data.edges.length
    ? data.edges.map((edge, index) => `<li class="reveal-item" style="--delay:${180 + index * 58}ms"><b>${String(index + 1).padStart(2, '0')}</b> <em>${edge.relation.label}</em><br>→ ${edge.target.label} <span>[${edge.target.id}]</span></li>`).join('')
    : `<li>${lang === 'ja' ? '該当するローカルエッジはありません。' : 'No matching local edges.'}</li>`;
  scene.selected = 0;
  scene.reveal = animate ? performance.now() : 0;
  scene.revealSecond = 0;
  if (animate) data.second = null;
  resetView();
  summary(data);
  updateSelection();
  draw(data);
  if (animate) window.setTimeout(() => { if (graphData === data) expandSecond(0); }, 850);
}

function resetView() { scene.x = 0; scene.y = 0; scene.zoom = 1; viewStat(); }

function nodesFor(data, width, height) {
  const root = { x: width * .5, y: height * .47, label: data.root.label, id: data.root.id };
  const count = Math.min(18, data.edges.length);
  const first = data.edges.slice(0, count).map((edge, index) => {
    const angle = -Math.PI / 2 + (Math.PI * 2 * index / Math.max(1, count));
    const ring = index % 2 ? .33 : .39;
    return {
      x: width * .5 + Math.cos(angle) * width * ring,
      y: height * .47 + Math.sin(angle) * height * .35,
      label: edge.target.label,
      id: edge.target.id,
      relation: edge.relation.label, level: 1,
    };
  });
  const second = ((data.second && data.second.edges) || []).map((edge, index, rows) => {
    const origin = first[data.second.from] || root;
    const base = Math.atan2(origin.y - root.y, origin.x - root.x);
    const spread = (index - (rows.length - 1) / 2) * .28;
    const distance = Math.min(width, height) * .27;
    return { x: origin.x + Math.cos(base + spread) * distance, y: origin.y + Math.sin(base + spread) * distance, label: edge.target.label, id: edge.target.id, relation: edge.relation.label, level: 2, parent: data.second.from };
  });
  return { root, first, second, all: [root, ...first, ...second] };
}

function draw(data) {
  cancelAnimationFrame(scene.animation);
  const canvas = $('#graph-canvas');
  const ctx = canvas.getContext('2d');
  const dpr = devicePixelRatio || 1;
  function paint(now) {
    const rect = canvas.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) { canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr); }
    scene.size = { w: width, h: height };
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const nodes = nodesFor(data, width, height);
    scene.nodes = nodes.all;
    const elapsed = scene.reveal ? now - scene.reveal : 10000;
    ctx.save();
    ctx.translate(scene.x, scene.y);
    ctx.scale(scene.zoom, scene.zoom);
    ctx.textAlign = 'center';
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    data.edges.slice(0, 18).forEach((edge, index) => {
      const node = nodes.first[index];
      const visible = Math.max(0, Math.min(1, (elapsed - index * 95) / 350));
      if (!visible) return;
      const active = index === scene.selected;
      ctx.globalAlpha = visible * (active ? 1 : .38);
      ctx.strokeStyle = active ? '#2fe6ff' : '#77818c';
      ctx.lineWidth = active ? 2.2 : 1;
      ctx.setLineDash(active ? [9, 8] : []);
      ctx.lineDashOffset = active ? -(now / 18) : 0;
      ctx.beginPath();
      ctx.moveTo(nodes.root.x, nodes.root.y);
      ctx.quadraticCurveTo((nodes.root.x + node.x) / 2, (nodes.root.y + node.y) / 2 - 18, node.x, node.y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = visible * (active ? 1 : .7);
      ctx.fillStyle = active ? '#2fe6ff' : '#8b949e';
      ctx.fillText(edge.relation.label.slice(0, 18), (nodes.root.x + node.x) / 2, (nodes.root.y + node.y) / 2 - 12);
    });
    const secondElapsed = scene.revealSecond ? now - scene.revealSecond : 0;
    ((data.second && data.second.edges) || []).forEach((edge, index) => {
      const node = nodes.second[index]; const origin = nodes.first[data.second.from];
      const visible = Math.max(0, Math.min(1, (secondElapsed - index * 125) / 420));
      if (!visible) return;
      ctx.globalAlpha = visible * .9; ctx.strokeStyle = '#00d992'; ctx.lineWidth = 1.4; ctx.setLineDash([4, 6]); ctx.lineDashOffset = -(now / 22);
      ctx.beginPath(); ctx.moveTo(origin.x, origin.y); ctx.quadraticCurveTo((origin.x + node.x) / 2, (origin.y + node.y) / 2 + 15, node.x, node.y); ctx.stroke(); ctx.setLineDash([]);
      ctx.globalAlpha = visible; ctx.fillStyle = '#00d992'; ctx.fillText(edge.relation.label.slice(0, 16), (origin.x + node.x) / 2, (origin.y + node.y) / 2 + 18);
    });
    nodes.all.forEach((node, index) => {
      const visible = node.level === 2 ? Math.max(0, Math.min(1, (secondElapsed - (index - nodes.first.length - 1) * 125) / 420)) : (index === 0 ? Math.max(0, Math.min(1, elapsed / 300)) : Math.max(0, Math.min(1, (elapsed - (index - 1) * 95) / 350)));
      if (!visible) return;
      const hot = index === 0 || node.level === 1 && index - 1 === scene.selected;
      const radius = (hot ? 14 : 7) * (.65 + .35 * visible);
      ctx.globalAlpha = visible;
      ctx.fillStyle = '#101010';
      ctx.strokeStyle = hot ? '#2fe6ff' : node.level === 2 ? '#00d992' : '#8b949e';
      ctx.lineWidth = hot ? 2.5 : 1.5;
      ctx.beginPath(); ctx.arc(node.x, node.y, radius, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      if (hot) { ctx.globalAlpha = visible * (.16 + .13 * Math.sin(now / 180)); ctx.beginPath(); ctx.arc(node.x, node.y, radius + 9 + 2 * Math.sin(now / 220), 0, Math.PI * 2); ctx.stroke(); }
      ctx.globalAlpha = visible;
      ctx.fillStyle = hot ? '#f2f2f2' : node.level === 2 ? '#d9fff1' : '#bdbdbd';
      ctx.fillText(node.label.slice(0, 19), node.x, node.y + 26);
      ctx.fillStyle = hot ? '#2fe6ff' : node.level === 2 ? '#00d992' : '#8b949e';
      ctx.fillText(node.id, node.x, node.y + 39);
    });
    ctx.restore();
    ctx.globalAlpha = 1;
    scene.animation = requestAnimationFrame(paint);
  }
  scene.animation = requestAnimationFrame(paint);
}

function policyCandidates(data) {
  const proofIds = new Set(data.outcome.proof_path.map(node => node.identifier));
  const unique = new Map();
  data.steps.forEach(step => step.frontier_sample.forEach(node => { if (!proofIds.has(node.identifier) && !unique.has(node.identifier)) unique.set(node.identifier, node); }));
  return [...unique.values()].slice(0, 14);
}

function renderPolicyPath(data) {
  graphMode = 'policy'; graphData = null; policyData = data; scene.reveal = performance.now(); resetView();
  const valid = data.outcome.valid_proof;
  $('#graph-title').textContent = `${valid ? 'VALID POLICY PROOF' : 'POLICY PATH'}  [${data.query.task_id}]`;
  $('#entity-card').className = 'entity-card result-arrive';
  $('#entity-card').innerHTML = `<strong>${valid ? 'VALID PROOF' : 'UNVERIFIED PATH'}</strong><p>${data.query.family} · ${data.steps.length} operators · ${data.outcome.credits} credits</p><p>${lang === 'ja' ? '方策が出した経路と、途中で見えた候補を同時表示しています。' : 'Showing the emitted policy path together with frontier candidates encountered during search.'}</p>`;
  const candidates = policyCandidates(data);
  $('#facts').innerHTML = candidates.map((node, index) => `<li class="reveal-item" style="--delay:${220 + index * 55}ms"><b>ALT</b> <em>${node.label}</em><br>→ ${node.identifier}</li>`).join('') || `<li>${lang === 'ja' ? '探索フロンティア候補はありません。' : 'No frontier alternatives were emitted.'}</li>`;
  $('#graph-summary').innerHTML = [`<span>PROGRAM <b>${data.steps.length} OPS</b></span>`,`<span>PROOF <b>${data.outcome.proof_path.length} NODES</b></span>`,`<span>FRONTIER <b>${candidates.length} WORDS</b></span>`,`<span>VALID <b>${valid ? 'YES' : 'NO'}</b></span>`,`<span id="view-stat">VIEW <b>1.00×</b></span>`].join('');
  $('#selected-edge').textContent = `${valid ? 'VERIFIED' : 'UNVERIFIED'} · ${data.outcome.answer ? data.outcome.answer.label : 'NO ANSWER'}`;
  drawPolicy(data, candidates);
  stage(valid ? 'POLICY PROOF' : 'POLICY PATH');
  notice(lang === 'ja' ? '学習済み方策の実行経路を表示中です。灰色ノードは探索中に現れた別候補で、証明経路ではありません。' : 'Showing the learned policy path. Gray nodes are frontier alternatives, not proof-path nodes.');
}

function drawPolicy(data, candidates) {
  cancelAnimationFrame(scene.animation);
  const canvas = $('#graph-canvas'), ctx = canvas.getContext('2d'), dpr = devicePixelRatio || 1;
  function paint(now) {
    const rect = canvas.getBoundingClientRect(), width = rect.width, height = rect.height;
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) { canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr); }
    scene.size = { w: width, h: height }; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
    const path = data.outcome.proof_path.map((node, index, rows) => ({ ...node, level: 'proof', index, x: width * (.18 + .64 * index / Math.max(1, rows.length - 1)), y: height * .50 }));
    const alternative = candidates.map((node, index) => ({ ...node, level: 'candidate', x: width * (.18 + .64 * ((index % 7) / 6)), y: height * (index < 7 ? .17 : .83) }));
    scene.nodes = [...path, ...alternative]; const elapsed = now - scene.reveal;
    ctx.save(); ctx.translate(scene.x, scene.y); ctx.scale(scene.zoom, scene.zoom); ctx.textAlign = 'center'; ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    path.slice(1).forEach((node, index) => { const visible = Math.max(0, Math.min(1, (elapsed - 280 - index * 420) / 360)); if (!visible) return; const previous = path[index]; const relation = data.query.relations[index] || data.query.relations[data.query.relations.length - 1]; ctx.globalAlpha = visible; ctx.strokeStyle = '#2fe6ff'; ctx.lineWidth = 2.4; ctx.setLineDash([10, 7]); ctx.lineDashOffset = -(now / 18); ctx.beginPath(); ctx.moveTo(previous.x + 16, previous.y); ctx.lineTo(node.x - 16, node.y); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle = '#2fe6ff'; ctx.fillText(relation ? relation.label : 'NEXT', (previous.x + node.x) / 2, previous.y - 18); });
    alternative.forEach((node, index) => { const visible = Math.max(0, Math.min(1, (elapsed - 1050 - index * 85) / 300)); if (!visible || path.length < 2) return; const origin = path[Math.min(1, path.length - 1)]; ctx.globalAlpha = visible * .46; ctx.strokeStyle = '#8b949e'; ctx.lineWidth = 1; ctx.setLineDash([3, 7]); ctx.beginPath(); ctx.moveTo(origin.x, origin.y); ctx.lineTo(node.x, node.y); ctx.stroke(); ctx.setLineDash([]); });
    scene.nodes.forEach(node => { const index = node.level === 'proof' ? node.index : alternative.indexOf(node); const delay = node.level === 'proof' ? index * 420 : 1050 + index * 85; const visible = Math.max(0, Math.min(1, (elapsed - delay) / 330)); if (!visible) return; const proof = node.level === 'proof'; const final = proof && node.index === path.length - 1; const radius = proof ? (final ? 16 : 12) : 6; ctx.globalAlpha = visible; ctx.fillStyle = '#101010'; ctx.strokeStyle = final ? '#00d992' : proof ? '#2fe6ff' : '#8b949e'; ctx.lineWidth = proof ? 2.5 : 1.2; ctx.beginPath(); ctx.arc(node.x, node.y, radius, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); if (final) { ctx.globalAlpha = visible * (.18 + .12 * Math.sin(now / 180)); ctx.beginPath(); ctx.arc(node.x, node.y, radius + 10, 0, Math.PI * 2); ctx.stroke(); } ctx.globalAlpha = visible; ctx.fillStyle = proof ? '#f2f2f2' : '#bdbdbd'; ctx.fillText(node.label.slice(0, 18), node.x, node.y + 29); ctx.fillStyle = final ? '#00d992' : proof ? '#2fe6ff' : '#8b949e'; ctx.fillText(node.identifier, node.x, node.y + 43); });
    ctx.restore(); ctx.globalAlpha = 1; scene.animation = requestAnimationFrame(paint);
  }
  scene.animation = requestAnimationFrame(paint);
}

async function showPolicyPath() {
  $('#run-policy-path').disabled = true;
  stage('RUNNING POLICY'); notice(lang === 'ja' ? '固定チェックポイントで学習済み方策をCPU実行中…' : 'Running the immutable policy checkpoint on CPU…');
  try { renderPolicyPath(await api(`/api/policy?task=0&lang=${lang}`)); } catch (error) { notice(error.message); stage('ERROR'); } finally { $('#run-policy-path').disabled = false; }
}

function point(event) { const rect = $('#graph-canvas').getBoundingClientRect(); return { x: event.clientX - rect.left, y: event.clientY - rect.top }; }
function clampView() { const { w, h } = scene.size; scene.x = Math.max(-w * .65, Math.min(w * .65, scene.x)); scene.y = Math.max(-h * .65, Math.min(h * .65, scene.y)); }
function graphSetup() {
  const canvas = $('#graph-canvas');
  canvas.addEventListener('pointerdown', event => {
    const p = point(event);
    canvas.setPointerCapture(event.pointerId);
    scene.drag = { pointer: event.pointerId, px: p.x, py: p.y, x: scene.x, y: scene.y };
    canvas.classList.add('dragging');
  });
  canvas.addEventListener('pointermove', event => {
    if (!scene.drag || scene.drag.pointer !== event.pointerId) return;
    const p = point(event);
    scene.x = scene.drag.x + p.x - scene.drag.px;
    scene.y = scene.drag.y + p.y - scene.drag.py;
    clampView();
  });
  const finish = event => {
    if (!scene.drag || scene.drag.pointer !== event.pointerId) return;
    const p = point(event);
    const moved = Math.hypot(p.x - scene.drag.px, p.y - scene.drag.py);
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    canvas.classList.remove('dragging');
    scene.drag = null;
    if (moved >= 6) return;
    const world = { x: (p.x - scene.x) / scene.zoom, y: (p.y - scene.y) / scene.zoom };
    if (graphMode === 'policy') {
      const found = scene.nodes.find(node => Math.hypot(world.x - node.x, world.y - node.y) < 24);
      if (found) notice(found.level === 'proof' ? `${found.label} [${found.identifier}] — ${lang === 'ja' ? '証明経路のノード' : 'proof-path node'}` : `${found.label} [${found.identifier}] — ${lang === 'ja' ? '探索中に現れたフロンティア候補' : 'frontier candidate encountered during search'}`);
      return;
    }
    if (!graphData) return;
    const hit = scene.nodes.find(node => node.level === 1 && Math.hypot(world.x - node.x, world.y - node.y) < 24);
    if (hit) { scene.selected = graphData.edges.findIndex(edge => edge.target.id === hit.id); updateSelection(); expandSecond(scene.selected); const edge = graphData.edges[scene.selected]; notice(lang === 'ja' ? `${edge.relation.label} → ${edge.target.label} を選択しました。二段目を展開します。` : `Selected ${edge.relation.label} → ${edge.target.label}. Expanding its second hop.`); }
  };
  canvas.addEventListener('pointerup', finish);
  canvas.addEventListener('pointercancel', finish);
  canvas.addEventListener('wheel', event => {
    event.preventDefault();
    const p = point(event);
    const before = { x: (p.x - scene.x) / scene.zoom, y: (p.y - scene.y) / scene.zoom };
    scene.zoom = Math.max(.55, Math.min(2.4, scene.zoom * (event.deltaY < 0 ? 1.1 : .9)));
    scene.x = p.x - before.x * scene.zoom;
    scene.y = p.y - before.y * scene.zoom;
    clampView(); viewStat();
  }, { passive: false });
  $('#graph-reset').onclick = () => { resetView(); scene.selected = 0; updateSelection(); notice(lang === 'ja' ? 'グラフ視点を初期位置へ戻しました。' : 'Graph view reset.'); };
}

async function runPolicy() {
  const out = $('#policy-result');
  out.innerHTML = `<p class="result-arrive">${lang === 'ja' ? '不変チェックポイントをCPUで実行中…' : 'Running immutable checkpoint on CPU…'}</p>`;
  try {
    const data = await api(`/api/policy?task=0&lang=${lang}`); const outcome = data.outcome;
    out.innerHTML = `<p class="${outcome.valid_proof ? 'result-valid' : 'result-invalid'} result-arrive">${outcome.valid_proof ? 'VALID PROOF' : 'NO VALID PROOF'}</p><p class="result-arrive" style="--delay:90ms">${lang === 'ja' ? '回答' : 'Answer'}: ${outcome.answer ? `${outcome.answer.label} [${outcome.answer.identifier}]` : 'none'}</p><p class="result-arrive" style="--delay:160ms">${lang === 'ja' ? '実行時間' : 'Elapsed'}: ${outcome.elapsed_ms} ms · ${lang === 'ja' ? 'クレジット' : 'credits'}: ${outcome.credits} · ${lang === 'ja' ? 'エッジ' : 'edges'}: ${outcome.edges_examined}</p><ol class="trace">${data.steps.map((step, index) => `<li class="reveal-item" style="--delay:${240 + index * 58}ms"><b>${String(step.index).padStart(2, '0')} ${step.operator}</b> ${step.trace}</li>`).join('')}</ol>`;
  } catch (error) { out.innerHTML = `<p class="result-invalid">${error.message}</p>`; }
}

document.querySelectorAll('.nav-item').forEach(button => button.onclick = () => { document.querySelectorAll('.nav-item,.panel').forEach(node => node.classList.remove('active')); button.classList.add('active'); $('#' + button.dataset.panel).classList.add('active'); });
$('#run-search').onclick = search;
$('#query').addEventListener('keydown', event => { if (event.key === 'Enter') search(); });
$('#apply-filter').onclick = loadGraph;
$('#relation').addEventListener('keydown', event => { if (event.key === 'Enter') loadGraph(); });
$('#run-policy').onclick = runPolicy;
$('#run-policy-path').onclick = showPolicyPath;
$('#language').onclick = () => { lang = lang === 'ja' ? 'en' : 'ja'; applyLanguage(); };
api('/api/health').then(() => { notice(lang === 'ja' ? '準備完了。単語またはQ-IDを入力してください。' : 'Ready. Enter a word or Q-ID.'); $('#graph-meta').textContent = 'LOCAL GRAPH / READY'; }).catch(error => notice(error.message));
graphSetup();
applyLanguage();
