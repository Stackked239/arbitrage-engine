(function () {
  'use strict';

  // ===== HELPERS =====
  function $(sel) { return document.querySelector(sel); }
  function $$(sel) { return document.querySelectorAll(sel); }

  function fmtUSD(val, showSign) {
    const n = Number(val);
    const abs = Math.abs(n);
    const str = abs.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    if (showSign) {
      return (n >= 0 ? '+$' : '-$') + str;
    }
    return '$' + str;
  }

  function fmtPct(val, showSign) {
    const n = Number(val);
    const str = Math.abs(n).toFixed(2) + '%';
    if (showSign) return (n >= 0 ? '+' : '-') + str;
    return str;
  }

  function fmtNum(val, decimals) {
    const n = Number(val);
    if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + 'M';
    if (Math.abs(n) >= 1e3) return n.toLocaleString('en-US', { maximumFractionDigits: decimals || 2 });
    if (Math.abs(n) < 0.001 && n !== 0) return n.toExponential(2);
    return n.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: decimals || 4 });
  }

  function fmtPrice(val) {
    const n = Number(val);
    if (n === 0) return '--';
    if (n >= 10000) return '$' + n.toLocaleString('en-US', { maximumFractionDigits: 0 });
    if (n >= 1) return '$' + n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    if (n >= 0.01) return '$' + n.toFixed(4);
    return '$' + n.toExponential(2);
  }

  function pnlClass(val) {
    const n = Number(val);
    if (n > 0.001) return 'positive';
    if (n < -0.001) return 'negative';
    return 'neutral';
  }

  function fmtTime(ts) {
    try {
      const d = new Date(ts);
      return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
    } catch { return '--'; }
  }

  function fmtDateTime(ts) {
    try {
      const d = new Date(ts);
      return d.toLocaleString('en-US', {
        month: 'short', day: 'numeric', year: 'numeric',
        hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false
      });
    } catch { return '--'; }
  }

  const ENGINE_LABELS = {
    engine1_funding_rate: { num: 'ENGINE 01', short: 'E1-FUND' },
    engine2_polymarket: { num: 'ENGINE 02', short: 'E2-POLY' },
    engine3_flash_loan: { num: 'ENGINE 03', short: 'E3-FLASH' },
    engine4_triangular: { num: 'ENGINE 04', short: 'E4-TRI' },
    engine5_cross_exchange: { num: 'ENGINE 05', short: 'E5-CROSS' },
  };

  // ===== LOAD DATA =====
  async function loadData() {
    try {
      const res = await fetch('/api/data');
      if (!res.ok) throw new Error('Failed to load data');
      return await res.json();
    } catch (e) {
      console.error('Data load error:', e);
      return null;
    }
  }

  // ===== RENDER HEADER =====
  function renderHeader(data) {
    const p = data.portfolio;
    $('#header-timestamp').textContent = fmtDateTime(data.timestamp);
    $('#total-capital').textContent = fmtUSD(p.total_capital_deployed);
    const pnlEl = $('#total-pnl');
    pnlEl.textContent = fmtUSD(p.total_net_pnl, true);
    pnlEl.classList.add(pnlClass(p.total_net_pnl));
    const retEl = $('#total-return');
    retEl.textContent = fmtPct(p.total_return_pct, true);
    retEl.classList.add(pnlClass(p.total_return_pct));
    $('#total-trades').textContent = p.total_trades;
    $('#m-total-capital').textContent = fmtUSD(p.total_capital_deployed);
    const mPnl = $('#m-total-pnl');
    mPnl.textContent = fmtUSD(p.total_net_pnl, true);
    mPnl.classList.add(pnlClass(p.total_net_pnl));
    const mRet = $('#m-total-return');
    mRet.textContent = fmtPct(p.total_return_pct, true);
    mRet.classList.add(pnlClass(p.total_return_pct));
    $('#m-total-trades').textContent = p.total_trades;
  }

  // ===== RENDER ENGINE CARDS =====
  function renderEngines(data) {
    const grid = $('#engine-grid');
    grid.innerHTML = '';
    const engineOrder = [
      'engine1_funding_rate', 'engine2_polymarket', 'engine3_flash_loan',
      'engine4_triangular', 'engine5_cross_exchange',
    ];

    engineOrder.forEach((key) => {
      const eng = data.portfolio.engines[key];
      const cycle = data.engine_cycle_results[key];
      const labels = ENGINE_LABELS[key];
      const hasResults = cycle && (cycle.session_pnl !== undefined);
      const card = document.createElement('div');
      card.className = 'engine-card' + (hasResults ? ' active' : '');
      let metricsHTML = '';
      if (cycle) metricsHTML = buildMetrics(key, cycle);

      card.innerHTML = `
        <div class="engine-header">
          <div class="engine-name-group">
            <span class="engine-number">${labels.num}</span>
            <span class="engine-name">${eng.name}</span>
          </div>
          <div class="engine-status">
            <span class="status-dot ${hasResults ? 'active' : ''}"></span>
            <span>${hasResults ? 'ACTIVE' : 'IDLE'}</span>
          </div>
        </div>
        <div class="engine-stats">
          <div class="engine-stat"><span class="engine-stat-label">Allocated</span><span class="engine-stat-value">${fmtUSD(eng.capital_allocated)}</span></div>
          <div class="engine-stat"><span class="engine-stat-label">Current</span><span class="engine-stat-value">${fmtUSD(eng.current_capital)}</span></div>
          <div class="engine-stat"><span class="engine-stat-label">P&L</span><span class="engine-stat-value ${pnlClass(eng.total_pnl)}">${fmtUSD(eng.total_pnl, true)}</span></div>
          <div class="engine-stat"><span class="engine-stat-label">Return</span><span class="engine-stat-value ${pnlClass(eng.return_pct)}">${fmtPct(eng.return_pct, true)}</span></div>
          <div class="engine-stat"><span class="engine-stat-label">Positions</span><span class="engine-stat-value">${eng.open_positions}</span></div>
          <div class="engine-stat"><span class="engine-stat-label">Trades</span><span class="engine-stat-value">${eng.total_trades}</span></div>
        </div>
        ${metricsHTML}
        <span class="engine-scan-pulse">SCANNING...</span>
      `;
      grid.appendChild(card);
    });
  }

  function buildMetrics(key, cycle) {
    let rows = '';
    switch (key) {
      case 'engine1_funding_rate':
        rows = `<div class="engine-metric-row"><span class="metric-key">Top Symbol</span><span class="metric-val highlight">${cycle.top_symbol || 'N/A'}</span></div><div class="engine-metric-row"><span class="metric-key">Rate</span><span class="metric-val">${cycle.top_rate_pct != null ? cycle.top_rate_pct.toFixed(2) + '%' : '--'}</span></div><div class="engine-metric-row"><span class="metric-key">Annualized</span><span class="metric-val highlight">${cycle.top_ann_pct != null ? fmtNum(cycle.top_ann_pct, 0) + '%' : '--'}</span></div><div class="engine-metric-row"><span class="metric-key">Scanned</span><span class="metric-val">${fmtNum(cycle.symbols_scanned)} symbols</span></div>`;
        break;
      case 'engine2_polymarket':
        rows = `<div class="engine-metric-row"><span class="metric-key">Top Market</span><span class="metric-val highlight metric-wrap">${cycle.top_market || 'N/A'}</span></div><div class="engine-metric-row"><span class="metric-key">APR</span><span class="metric-val highlight">${cycle.top_market_apr_pct != null ? fmtNum(cycle.top_market_apr_pct, 0) + '%' : '--'}</span></div><div class="engine-metric-row"><span class="metric-key">Markets Scanned</span><span class="metric-val">${cycle.markets_scanned || 0}</span></div><div class="engine-metric-row"><span class="metric-key">Rewards</span><span class="metric-val">${fmtUSD(cycle.rewards_this_period || 0)}</span></div>`;
        break;
      case 'engine3_flash_loan':
        rows = `<div class="engine-metric-row"><span class="metric-key">Top Token</span><span class="metric-val highlight">${cycle.top_token || 'N/A'}</span></div><div class="engine-metric-row"><span class="metric-key">Spread</span><span class="metric-val">${cycle.top_spread_pct != null ? cycle.top_spread_pct.toFixed(2) + '%' : '--'}</span></div><div class="engine-metric-row"><span class="metric-key">Net Profit</span><span class="metric-val highlight">${fmtUSD(cycle.top_net_profit_usd || 0)}</span></div><div class="engine-metric-row"><span class="metric-key">Tri Paths</span><span class="metric-val">${cycle.triangular_paths_found || 0} found</span></div>`;
        break;
      case 'engine4_triangular':
        rows = `<div class="engine-metric-row"><span class="metric-key">Pairs in Graph</span><span class="metric-val">${fmtNum(cycle.pairs_in_graph || 0)}</span></div><div class="engine-metric-row"><span class="metric-key">Opportunities</span><span class="metric-val">${cycle.opportunities_found || 0}</span></div><div class="engine-metric-row"><span class="metric-key">Top Triangle</span><span class="metric-val">${cycle.top_triangle || 'N/A'}</span></div><div class="engine-metric-row"><span class="metric-key">Executions</span><span class="metric-val">${cycle.total_executions || 0}</span></div>`;
        break;
      case 'engine5_cross_exchange':
        rows = `<div class="engine-metric-row"><span class="metric-key">Top Token</span><span class="metric-val highlight">${cycle.top_token || 'N/A'}</span></div><div class="engine-metric-row"><span class="metric-key">Spread</span><span class="metric-val">${cycle.top_spread_pct != null ? cycle.top_spread_pct.toFixed(2) + '%' : '--'}</span></div><div class="engine-metric-row"><span class="metric-key">Buy</span><span class="metric-val">${cycle.top_buy_exchange || '--'}</span></div><div class="engine-metric-row"><span class="metric-key">Sell</span><span class="metric-val">${cycle.top_sell_exchange || '--'}</span></div>`;
        break;
    }
    return `<div class="engine-metric">${rows}</div>`;
  }

  // ===== RENDER TRADE TABLE =====
  function renderTrades(data) {
    const tbody = $('#trade-tbody');
    tbody.innerHTML = '';
    const trades = (data.trades || []).slice(-15).reverse();

    if (trades.length === 0) {
      const tr = document.createElement('tr');
      tr.innerHTML = '<td colspan="9" style="text-align:center; color:var(--text-muted); padding:24px;">No trades recorded yet</td>';
      tbody.appendChild(tr);
      return;
    }

    trades.forEach((t) => {
      const tr = document.createElement('tr');
      const engineLabel = ENGINE_LABELS[t.engine] ? ENGINE_LABELS[t.engine].short : t.engine;
      let sideClass = 'td-side-neutral';
      if (t.side === 'long') sideClass = 'td-side-long';
      else if (t.side === 'short') sideClass = 'td-side-short';
      let actionClass = 'action-open';
      if (t.action === 'close') actionClass = 'action-close';
      else if (t.action === 'funding_payment') actionClass = 'action-funding';
      else if (t.action === 'reward') actionClass = 'action-reward';
      else if (t.action === 'flash_loan_arb') actionClass = 'action-flash';
      const pnlC = pnlClass(t.net_pnl);
      const pnlTdClass = pnlC === 'positive' ? 'td-positive' : (pnlC === 'negative' ? 'td-negative' : 'td-neutral');
      tr.innerHTML = `
        <td>${fmtTime(t.timestamp)}</td>
        <td class="td-engine">${engineLabel}</td>
        <td>${t.symbol || '--'}</td>
        <td class="${sideClass}">${t.side || '--'}</td>
        <td><span class="td-action ${actionClass}">${(t.action || '--').replace(/_/g, ' ').toUpperCase()}</span></td>
        <td class="num">${t.amount > 0 ? fmtNum(t.amount) : '--'}</td>
        <td class="num">${t.price > 0 ? fmtPrice(t.price) : '--'}</td>
        <td class="num">${t.fee > 0 ? fmtUSD(t.fee) : '--'}</td>
        <td class="num ${pnlTdClass}">${fmtUSD(t.net_pnl, true)}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  function renderFooter(data) {
    $('#footer-scan').textContent = 'Last scan: ' + fmtDateTime(data.timestamp);
  }

  // ===== AUTO-REFRESH =====
  let lastTimestamp = null;

  function startAutoRefresh() {
    setInterval(async () => {
      try {
        const resp = await fetch('/api/data?t=' + Date.now());
        if (!resp.ok) return;
        const data = await resp.json();
        if (data.timestamp !== lastTimestamp) {
          lastTimestamp = data.timestamp;
          renderHeader(data);
          renderEngines(data);
          renderTrades(data);
          renderFooter(data);
          updateRefreshIndicator();
        }
      } catch (e) {}
    }, 30000);
  }

  function updateRefreshIndicator() {
    const el = document.getElementById('refresh-indicator');
    if (el) {
      el.textContent = 'LIVE';
      el.style.color = 'var(--green)';
      el.style.textShadow = '0 0 8px rgba(0,255,136,0.4)';
      setTimeout(() => { el.style.textShadow = 'none'; }, 1500);
    }
  }

  // ===== INIT =====
  async function init() {
    const data = await loadData();
    if (!data) {
      $('#engine-grid').innerHTML = '<div class="loading-overlay"><div class="loading-text">ERROR: FAILED TO LOAD DATA</div></div>';
      return;
    }
    lastTimestamp = data.timestamp;
    renderHeader(data);
    renderEngines(data);
    renderTrades(data);
    renderFooter(data);
    startAutoRefresh();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
