/**
 * WattWatch — Client-Side Application Controller
 * ===============================================
 * Operates ONLY on actual uploaded file data:
 * - Starts with a clean welcome screen & upload area (no dummy/static data).
 * - Uploads the user's CSV file and receives real computed analysis.
 * - Dynamically renders SVG charts, metrics, health ring, and review table.
 * - Displays clear error messages if the file is invalid or missing.
 */

let allRows = [];
let latestData = null;

document.addEventListener("DOMContentLoaded", () => {
  setupNavigation();
  setupUpload();
  setupTableFiltering();
  setupDialog();
  setupPeriodToggle();

  // Initially, do NOT run with sample or dummy data.
  // The page starts clean, awaiting user's file.
});

// 1. Navigation smooth scrolling and active links
function setupNavigation() {
  const links = document.querySelectorAll(".side-nav a");
  links.forEach(link => {
    link.addEventListener("click", () => {
      links.forEach(l => l.classList.remove("active"));
      link.classList.add("active");
    });
  });
}

// 2. Upload handling
function setupUpload() {
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("csv-file");
  const runBtn = document.getElementById("btn-run-analysis");

  dropZone.addEventListener("click", (e) => {
    if (e.target !== fileInput) fileInput.click();
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  runBtn.addEventListener("click", () => {
    const file = fileInput.files[0] || null;
    runAnalysis(file);
  });
}

function handleFileSelected(file) {
  if (!file.name.toLowerCase().endsWith(".csv")) {
    showToast("Invalid file format. Please select a .csv file.", true);
    document.getElementById("uploadFeedback").textContent = "Error: Please upload a .csv file.";
    return;
  }

  const badge = document.getElementById("selectedFile");
  badge.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  badge.classList.add("visible");
  document.getElementById("uploadFeedback").textContent = `Processing "${file.name}"...`;

  // Process the uploaded file
  runAnalysis(file);
}

function setupPeriodToggle() {
  const periodBtn = document.getElementById("periodButton");
  let periodMode = "all";
  periodBtn.addEventListener("click", () => {
    if (!latestData || allRows.length === 0) return;

    if (periodMode === "all") {
      periodMode = "30";
      periodBtn.innerHTML = "Last 30 days <span>⌄</span>";
    } else {
      periodMode = "all";
      periodBtn.innerHTML = "All readings <span>⌄</span>";
    }
    renderSvgChart(allRows, periodMode);
  });
}

// 3. Send file to Flask API and process actual data dynamically
async function runAnalysis(file) {
  setLoading(true, "Processing file...");

  const zLimit = document.getElementById("z_limit").value || 3.0;
  const iqrMult = document.getElementById("iqr_multiplier").value || 1.5;
  const contam = document.getElementById("contamination").value || 0.05;

  const formData = new FormData();
  if (file) {
    formData.append("file", file);
  }
  formData.append("z_limit", zLimit);
  formData.append("iqr_multiplier", iqrMult);
  formData.append("contamination", contam);

  try {
    const res = await fetch("/api/run", {
      method: "POST",
      body: formData
    });
    const data = await res.json();

    if (!res.ok || !data.success) {
      throw new Error(data.error || "Analysis failed.");
    }

    latestData = data;
    allRows = data.rows;

    // Show dashboard results & hide welcome box
    document.getElementById("welcomeState").style.display = "none";
    document.getElementById("dashboardResults").style.display = "block";

    // Update active file label
    document.getElementById("file-status-label").textContent = `${data.filename}`;
    document.getElementById("banner-title").textContent = `File Active: ${data.filename}`;
    document.getElementById("banner-desc").textContent = `Processed ${data.rows.length} readings from your file. Detection checks complete.`;
    document.getElementById("workspace-mode-label").textContent = data.filename;
    document.getElementById("table-badge").textContent = data.filename;

    // Render all real data
    updateMetrics(data);
    updateQuality(data);
    updateMethods(data);
    renderSvgChart(data.rows, "all");
    renderTable(data.rows);

    showToast(`"${data.filename}" processed successfully!`);
    document.getElementById("uploadFeedback").textContent = `Done. Processed ${data.rows.length} rows from "${data.filename}".`;

  } catch (err) {
    showToast(err.message, true);
    document.getElementById("uploadFeedback").textContent = `Error: ${err.message}`;
  } finally {
    setLoading(false);
  }
}

// 4. Update Metric Cards from actual data
function updateMetrics(data) {
  const stats = data.stats;
  const comp = data.compare_info;
  const count = stats.count;

  document.getElementById("metric-total-readings").textContent = count.toLocaleString();
  document.getElementById("metric-avg-consumption").innerHTML = `${stats.mean.toFixed(2)} <small>kWh</small>`;
  
  const flagged = comp.consensus_count;
  const flaggedPct = ((flagged / Math.max(1, count)) * 100).toFixed(1);
  document.getElementById("metric-flagged-count").textContent = flagged;
  document.getElementById("metric-flagged-percent").textContent = `${flaggedPct}%`;

  if (data.rows.length > 0) {
    const firstDate = data.rows[0].date;
    const lastDate = data.rows[data.rows.length - 1].date;
    document.getElementById("metric-date-span").textContent = `${firstDate} to ${lastDate}`;
    document.getElementById("stat-dates").textContent = `${firstDate.substring(5)}–${lastDate.substring(5)}`;
  }

  document.getElementById("summary-avg").innerHTML = `${stats.mean.toFixed(2)} <small>kWh</small>`;
  document.getElementById("summary-max").innerHTML = `${stats.max.toFixed(1)} <small>kWh</small>`;
  document.getElementById("summary-flagged").innerHTML = `${flagged} <small>shown</small>`;

  document.getElementById("stat-median").textContent = `${stats.median.toFixed(2)} kWh`;
  document.getElementById("stat-std").textContent = `${stats.std_dev.toFixed(2)} kWh`;
}

// 5. Update Quality & Circular Progress Ring from actual data
function updateQuality(data) {
  const total = data.rows.length;
  const flagged = data.compare_info.consensus_count;
  const clean = data.clean_info;

  const validCount = clean.final_rows || total;
  const invalidCount = (clean.invalid_dates_dropped || 0) + (clean.negatives_removed || 0);
  const totalRaw = clean.initial_rows || total;
  const validPercent = totalRaw > 0 ? ((validCount / totalRaw) * 100).toFixed(1) : "100.0";

  document.getElementById("quality-percent").textContent = `${validPercent}%`;
  document.getElementById("quality-valid-count").textContent = validCount.toLocaleString();
  document.getElementById("quality-flagged-count").textContent = flagged.toLocaleString();
  document.getElementById("quality-invalid-count").textContent = invalidCount.toLocaleString();

  document.getElementById("bar-valid").style.width = `${validPercent}%`;
  const flagWidth = Math.min(100, Math.max(2, (flagged / Math.max(1, total)) * 100));
  document.getElementById("bar-flagged").style.width = `${flagWidth.toFixed(1)}%`;
  
  const invalidWidth = Math.min(100, (invalidCount / Math.max(1, totalRaw)) * 100);
  document.getElementById("bar-invalid").style.width = `${invalidWidth.toFixed(1)}%`;

  // SVG ring circumference is 194.8
  const ring = document.getElementById("quality-circle");
  const offset = 194.8 - (parseFloat(validPercent) / 100) * 194.8;
  ring.style.strokeDasharray = "194.8";
  ring.style.strokeDashoffset = `${offset}`;
}

// 6. Update Detection Cards from actual data
function updateMethods(data) {
  const det = data.detect_info;
  document.getElementById("card-count-z").textContent = `${det.zscore_count} days flagged`;
  document.getElementById("card-count-iqr").textContent = `${det.iqr_count} days flagged`;
  document.getElementById("card-count-if").textContent = `${det.iforest_count} days flagged`;

  document.getElementById("card-setting-z").textContent = `|z| > ${det.z_limit}`;
  document.getElementById("card-setting-iqr").textContent = `${det.iqr_multiplier} × IQR`;
  document.getElementById("card-setting-if").textContent = `${det.contamination}`;
}

// 7. Dynamic SVG Chart from actual data
function renderSvgChart(rows, period) {
  let displayRows = rows;
  if (period === "30" && rows.length > 30) {
    displayRows = rows.slice(-30);
  }

  const n = displayRows.length;
  if (n === 0) return;

  const consumptions = displayRows.map(r => r.kwh);
  const maxVal = Math.max(...consumptions);
  const minVal = Math.min(...consumptions);
  const range = (maxVal - minVal) || 1;

  const W = 720;
  const H = 200;
  const topPad = 20;

  const points = displayRows.map((r, i) => {
    const x = (i / (n - 1 || 1)) * W;
    const norm = (r.kwh - minVal) / range;
    const y = H - (norm * (H - topPad)) + 10;
    return { x: Number(x.toFixed(1)), y: Number(y.toFixed(1)), row: r };
  });

  // Construct line path and area path
  let pathD = `M${points[0].x} ${points[0].y}`;
  for (let i = 1; i < points.length; i++) {
    pathD += ` L${points[i].x} ${points[i].y}`;
  }

  const areaD = `${pathD} L${W} 230 L0 230 Z`;

  document.getElementById("chartLinePath").setAttribute("d", pathD);
  document.getElementById("chartAreaFill").setAttribute("d", areaD);

  // Plot flagged dots (amber circles)
  const pointsGroup = document.getElementById("chartPoints");
  pointsGroup.innerHTML = "";

  let flaggedCount = 0;
  points.forEach(p => {
    if (p.row.flagged === 1) {
      flaggedCount++;
      const ring = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      ring.setAttribute("class", "flag-ring");
      ring.setAttribute("cx", p.x);
      ring.setAttribute("cy", p.y);
      ring.setAttribute("r", "7");

      const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      dot.setAttribute("class", "flag-dot");
      dot.setAttribute("cx", p.x);
      dot.setAttribute("cy", p.y);
      dot.setAttribute("r", "3");

      const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
      title.textContent = `${p.row.date}: ${p.row.kwh.toFixed(1)} kWh (Flagged: ${p.row.votes}/3 votes)`;
      dot.appendChild(title);

      pointsGroup.appendChild(ring);
      pointsGroup.appendChild(dot);
    }
  });

  document.getElementById("summary-flagged").innerHTML = `${flaggedCount} <small>shown</small>`;

  // Y-Axis labels (5 steps)
  const yAxis = document.getElementById("chartYAxis");
  const step = range / 4;
  yAxis.innerHTML = `
    <span>${Math.round(maxVal)}</span>
    <span>${Math.round(maxVal - step)}</span>
    <span>${Math.round(maxVal - 2 * step)}</span>
    <span>${Math.round(minVal + step)}</span>
    <span>${Math.round(minVal)}</span>
  `;

  // X-Axis labels
  const xAxis = document.getElementById("chartXAxis");
  if (displayRows.length >= 4) {
    const p1 = displayRows[0].date.substring(5);
    const p2 = displayRows[Math.floor(n * 0.25)].date.substring(5);
    const p3 = displayRows[Math.floor(n * 0.50)].date.substring(5);
    const p4 = displayRows[Math.floor(n * 0.75)].date.substring(5);
    const p5 = displayRows[n - 1].date.substring(5);
    xAxis.innerHTML = `<span>${p1}</span><span>${p2}</span><span>${p3}</span><span>${p4}</span><span>${p5}</span>`;
  }
}

// 8. Results Table Search and Filter
function setupTableFiltering() {
  const search = document.getElementById("tableSearch");
  const filter = document.getElementById("resultFilter");

  search.addEventListener("input", () => applyTableFilter());
  filter.addEventListener("change", () => applyTableFilter());
}

function applyTableFilter() {
  const searchVal = (document.getElementById("tableSearch").value || "").trim().toLowerCase();
  const filterVal = document.getElementById("resultFilter").value;

  const filtered = allRows.filter(row => {
    // Flag filter
    if (filterVal === "flagged" && row.flagged !== 1) return false;
    if (filterVal === "normal" && row.flagged === 1) return false;

    // Search query
    if (searchVal) {
      const dateMatch = row.date.toLowerCase().includes(searchVal);
      const kwhMatch = row.kwh.toString().includes(searchVal);
      const statusMatch = row.flagged === 1 ? "flagged review needed" : "normal no flag";
      return dateMatch || kwhMatch || statusMatch.includes(searchVal);
    }
    return true;
  });

  renderTable(filtered);
}

function renderTable(rows) {
  const tbody = document.getElementById("resultsBody");
  const countBadge = document.getElementById("tableCount");
  const empty = document.getElementById("emptyState");

  countBadge.textContent = `${rows.length} rows`;

  if (rows.length === 0) {
    tbody.innerHTML = "";
    empty.hidden = false;
    return;
  }

  empty.hidden = true;
  tbody.innerHTML = rows.map(r => {
    const isFlag = r.flagged === 1;
    const rowClass = isFlag ? ' class="row-flagged"' : "";

    const sign = r.z_score > 0 ? "+" : "";
    const zScoreClass = r.z_flag === 1 ? "score-high" : "score-low";
    const zScoreTag = `<span class="score ${zScoreClass}">${sign}${r.z_score.toFixed(2)}</span>`;

    const iqrTag = r.iqr_flag === 1
      ? `<span class="result-flag">Flagged</span>`
      : `<span class="result-normal">Normal</span>`;

    const ifTag = r.iforest_flag === 1
      ? `<span class="result-flag">Flagged</span>`
      : `<span class="result-normal">Normal</span>`;

    const statusTag = isFlag
      ? `<span class="review-status"><i></i> Review needed</span>`
      : `<span class="review-status normal-status"><i></i> No flag</span>`;

    return `
      <tr${rowClass}>
        <td>${r.date}</td>
        <td><strong>${r.kwh.toFixed(1)} <small>kWh</small></strong></td>
        <td>${zScoreTag}</td>
        <td>${iqrTag}</td>
        <td>${ifTag}</td>
        <td>${r.votes} / 3</td>
        <td>${statusTag}</td>
      </tr>
    `;
  }).join("");
}

// 9. Dialog Modal
function setupDialog() {
  const dialog = document.getElementById("formatDialog");
  const btnShow = document.getElementById("sampleButton");
  const btnHelp = document.getElementById("btn-quick-help");

  btnShow.addEventListener("click", () => dialog.showModal());
  btnHelp.addEventListener("click", () => dialog.showModal());
}

// 10. Toasts & Loading State
let toastTimer = null;
function showToast(msg, isError = false) {
  const toast = document.getElementById("toast");
  if (toastTimer) clearTimeout(toastTimer);

  toast.textContent = msg;
  toast.className = `toast show ${isError ? "toast-error" : ""}`;
  toastTimer = setTimeout(() => {
    toast.className = "toast";
  }, 3500);
}

function setLoading(isLoading, text = "Ready for upload") {
  const status = document.getElementById("connection-status");
  const statusText = document.getElementById("status-text");

  if (isLoading) {
    status.classList.add("status-loading");
    statusText.textContent = text;
  } else {
    status.classList.remove("status-loading");
    statusText.textContent = text;
  }
}
