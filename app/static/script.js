// ============================================================
// NAVIGATION
// ============================================================
const navBtns  = document.querySelectorAll('.nav-btn');
const sections = document.querySelectorAll('.section');
const topTitle = document.getElementById('topTitle');
const titles   = { dashboard:'Dashboard', predict:'Prédiction', analytics:'Analytics', about:'À propos' };

navBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const sec = btn.dataset.sec;
    navBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    sections.forEach(s => s.classList.remove('active'));
    document.getElementById('sec-' + sec).classList.add('active');
    topTitle.textContent = titles[sec] || '';
    document.getElementById('sidebar').classList.remove('open');
    document.getElementById('overlay').classList.remove('open');
  });
});

// ============================================================
// MOBILE
// ============================================================
document.getElementById('mobToggle').addEventListener('click', () => {
  document.getElementById('sidebar').classList.toggle('open');
  document.getElementById('overlay').classList.toggle('open');
});
document.getElementById('overlay').addEventListener('click', () => {
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('overlay').classList.remove('open');
});

// ============================================================
// THEME
// ============================================================
document.getElementById('themeBtn').addEventListener('click', () => {
  const html = document.documentElement;
  const dark = html.getAttribute('data-theme') === 'dark';
  html.setAttribute('data-theme', dark ? 'light' : 'dark');
  document.getElementById('themeIco').className = dark ? 'fa-solid fa-sun' : 'fa-solid fa-moon';
  document.getElementById('themeLbl').textContent = dark ? 'Mode clair' : 'Mode sombre';
});

// ============================================================
// CHARTS
// ============================================================
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.color = '#7b8799';

new Chart(document.getElementById('riskChart'), {
  type: 'doughnut',
  data: {
    labels: ['Faible', 'Moyen', 'Élevé'],
    datasets: [{ data:[58,24,18], backgroundColor:['#10b981','#f59e0b','#ef4444'], borderWidth:0, hoverOffset:6 }]
  },
  options: { cutout:'68%', plugins:{ legend:{ display:false } }, responsive:true, maintainAspectRatio:false }
});

new Chart(document.getElementById('churnChart'), {
  type: 'doughnut',
  data: {
    labels: ['Fidèles','Churners'],
    datasets: [{ data:[66.7,33.3], backgroundColor:['#10b981','#ef4444'], borderWidth:0, hoverOffset:6 }]
  },
  options: { cutout:'68%', plugins:{ legend:{ display:false } }, responsive:true, maintainAspectRatio:false }
});

new Chart(document.getElementById('featChart'), {
  type: 'bar',
  data: {
    labels: ['AvgBasketValue','MonetaryPerDay','Frequency','MonetaryTotal','UniqueProducts',
             'AvgProductsPerTrans','MonetaryAvg','PreferredMonth','RegMonth','IsPrivateIP'],
    datasets: [{
      label: 'Importance',
      data: [0.18, 0.14, 0.11, 0.09, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02],
      backgroundColor: 'rgba(99,102,241,0.7)',
      borderRadius: 4
    }]
  },
  options: {
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend:{ display:false } },
    scales: {
      x: { grid:{ color:'rgba(255,255,255,0.04)' } },
      y: { grid:{ display:false } }
    }
  }
});

// ============================================================
// MÉTRIQUES — chargées depuis Flask
// ============================================================
async function loadMetrics() {
  try {
    const res  = await fetch('/metrics');
    const data = await res.json();
    if (data.error) return;

    const acc = (data.accuracy * 100).toFixed(1) + '%';
    const f1  = (data.f1 * 100).toFixed(1) + '%';
    const rec = (data.recall * 100).toFixed(1) + '%';
    const auc = data.roc_auc.toFixed(3);

    // Dashboard KPI
    document.getElementById('kpi-acc').textContent  = acc;
    document.getElementById('kpi-auc').textContent  = auc;
    document.getElementById('kpi-churn').textContent = data.churn_rate + '%';

    // Analytics
    document.getElementById('a-acc').textContent = acc;
    document.getElementById('a-f1').textContent  = f1;
    document.getElementById('a-rec').textContent = rec;
    document.getElementById('a-auc').textContent = auc;

    // About
    document.getElementById('ab-acc').textContent = acc;
    document.getElementById('ab-auc').textContent = auc;
    document.getElementById('ab-f1').textContent  = f1;
    document.getElementById('ab-acc-bar').style.width = (data.accuracy * 100) + '%';
    document.getElementById('ab-auc-bar').style.width = (data.roc_auc  * 100) + '%';
    document.getElementById('ab-f1-bar').style.width  = (data.f1       * 100) + '%';

  } catch(e) { console.error('Metrics error:', e); }
}

document.addEventListener('DOMContentLoaded', loadMetrics);

// ============================================================
// PRÉDICTION
// ============================================================
document.getElementById('runBtn').addEventListener('click', async () => {
  const btn  = document.getElementById('runBtn');
  const txt  = document.getElementById('btnTxt');
  const spin = document.getElementById('btnSpin');

  btn.disabled = true;
  txt.style.display  = 'none';
  spin.style.display = 'block';

  const payload = {
    frequency:         parseFloat(document.getElementById('inp-freq').value),
    monetary:          parseFloat(document.getElementById('inp-monetary').value),
    favorite_season:   document.getElementById('inp-season').value,
    product_diversity: document.getElementById('inp-diversity').value
  };

  try {
    const res    = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const result = await res.json();
    if (result.error) throw new Error(result.error);

    const prob  = result.probability;
    const level = prob < 35 ? 'low' : prob < 65 ? 'med' : 'hi';
    const labels  = { low:'Risque faible', med:'Risque modéré', hi:'Risque élevé' };
    const verdicts = {
      low: 'Client fidèle. Maintenez l\'engagement avec des offres de fidélité personnalisées.',
      med: 'Risque modéré. Envisagez une campagne de réactivation ciblée dans les 30 jours.',
      hi:  'Risque élevé ! Contactez ce client immédiatement avec une offre exclusive.'
    };

    document.getElementById('resIdle').style.display   = 'none';
    document.getElementById('resCont').style.display   = 'block';

    const badge = document.getElementById('rBadge');
    badge.className = 'r-badge ' + level;
    document.getElementById('rLabel').textContent = labels[level];

    const probEl = document.getElementById('rProb');
    probEl.className   = 'r-prob ' + level;
    probEl.textContent = prob + '%';

    const fill = document.getElementById('rFill');
    fill.className  = 'r-fill ' + level;
    fill.style.width = '0%';
    setTimeout(() => { fill.style.width = prob + '%'; }, 50);

    const verdict = document.getElementById('rVerdict');
    verdict.className   = 'r-verdict ' + level;
    verdict.textContent = verdicts[level];

    // 3 modèles
    document.getElementById('rfVal').textContent  = result.rf.churn  + '%';
    document.getElementById('xgbVal').textContent = result.xgb.churn + '%';
    document.getElementById('stkVal').textContent = result.stacking.churn + '%';

    // Couleurs dynamiques
    const colLevel = (v) => v < 35 ? 'var(--success)' : v < 65 ? 'var(--warning)' : 'var(--danger)';
    document.getElementById('rfVal').style.color  = colLevel(result.rf.churn);
    document.getElementById('xgbVal').style.color = colLevel(result.xgb.churn);
    document.getElementById('stkVal').style.color = colLevel(result.stacking.churn);

    // Segment
    document.getElementById('segLbl').textContent = '📊 ' + result.segment;

    // Régression
    document.getElementById('monLbl').textContent = '💷 ' + result.monetary_pred + ' £ prédit';

  } catch(e) {
    alert('Erreur : ' + e.message);
  } finally {
    btn.disabled       = false;
    txt.style.display  = 'flex';
    spin.style.display = 'none';
  }
});