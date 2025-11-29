const API_URL = '';
let marketChartInstance = null;
let simulation = null; // Store D3 simulation
let currentIndustryData = [];
let currentIndustryFullLabels = [];
let selectedIndustryIndex = null;
let allJobsCache = []; // Cache for autocomplete

// ==========================================
// 1. PREDICTOR LOGIC
// ==========================================
async function analyzeTask() {
    const inputEl = document.getElementById('taskInput');
    if (!inputEl) return; // Exit if not on predictor page
    
    const input = inputEl.value;
    const loader = document.getElementById('loader');
    const resultsArea = document.getElementById('resultsArea');

    if (!input) {
        alert("Please enter a task.");
        return;
    }
    
    if(loader) loader.classList.remove('hidden');
    if(resultsArea) resultsArea.classList.add('hidden');

    try {
        const res = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text: input})
        });

        if (!res.ok) throw new Error(`Server error: ${res.status}`);

        const data = await res.json();

        // Update UI Elements
        const labelEl = document.getElementById('riskLabel');
        const barEl = document.getElementById('confidenceBar');
        const probText = document.getElementById('probabilityText');
        
        if (labelEl) {
            labelEl.innerText = data.label;
            labelEl.style.color = data.label === 'High Risk' ? '#ef4444' : '#10b981';
        }
        
        if (probText) {
            probText.innerText = `${(data.probability*100).toFixed(1)}% Automation Probability`;
        }
        
        if (barEl) {
            barEl.style.width = '0%';
            setTimeout(() => {
                barEl.style.width = `${data.probability*100}%`;
            }, 50);
            barEl.style.backgroundColor = data.label === 'High Risk' ? '#ef4444' : '#10b981';
        }

        // Prefer server-provided SHAP factors, fallback to a simple client-side extractor
        let highFactors = (data.high_risk_factors && data.high_risk_factors.length) ? data.high_risk_factors : [];
        let lowFactors = (data.low_risk_factors && data.low_ris_factors.length) ? data.low_risk_factors : [];

        if (highFactors.length === 0 && lowFactors.length === 0) {
            const fallback = generateFallbackFactors(input);
            highFactors = fallback.high;
            lowFactors = fallback.low;
        }

        fillList('highRiskList', highFactors);
        fillList('lowRiskList', lowFactors);
        
        if(loader) loader.classList.add('hidden');
        if(resultsArea) resultsArea.classList.remove('hidden');

    } catch (e) {
        console.error(e);
        alert("Error analyzing task. Is the Python server running?");
        if(loader) loader.classList.add('hidden');
    }
}

function fillList(id, items) {
    const el = document.getElementById(id);
    if (!el) return;
    if (!items || items.length === 0) {
        el.innerHTML = '<li><em>No factors available</em></li>';
        return;
    }
    el.innerHTML = items.map(i => `
        <li>
            <span>${i.word}</span>
            <span style="opacity:0.7; font-family: monospace; margin-left:8px;">${Math.abs(i.score).toFixed(2)}</span>
        </li>
    `).join('');
}

// Helper: truncate label with ellipsis if too long
function truncateLabel(s, maxLen=30) {
    if (!s || s.length <= maxLen) return s;
    return s.slice(0, maxLen-1).trim() + '…';
}

// Render industry table (simple, no pagination)
function renderIndustryTable(industryData) {
    const tbody = document.getElementById('industryTableBody');
    if (!tbody) return;
    tbody.innerHTML = '';
    industryData.forEach((item, idx) => {
        const tr = document.createElement('tr');
        tr.style.cursor = 'pointer';
        tr.style.borderBottom = '1px solid rgba(0,0,0,0.05)';
        tr.style.transition = 'all 0.2s ease';
        tr.style.padding = '8px';
        tr.style.userSelect = 'none';
        tr.style.WebkitUserSelect = 'none';
        
        // Risk level color based on score
        const riskColor = item.Automation_Score >= 2.05 ? '#ef4444' : '#10b981';
        
        tr.innerHTML = `
            <td style="padding:12px 10px; text-align:left; font-weight:500; color:#1e293b; word-break:break-word;">${item.Industry}</td>
            <td style="padding:12px 10px; text-align:right; font-weight:600; color:${riskColor};">${item.Automation_Score.toFixed(2)}</td>
            <td style="padding:12px 10px; text-align:right; color:#64748b;">${item.count}</td>
        `;
        
        tr.onmouseenter = () => {
            tr.style.background = 'rgba(59,130,246,0.04)';
            tr.style.transform = 'translateX(4px)';
        };
        tr.onmouseleave = () => {
            tr.style.background = '';
            tr.style.transform = 'translateX(0)';
        };
        
        tr.onclick = () => highlightIndustryRow(item.Industry);
        tbody.appendChild(tr);
    });
}

function highlightIndustryRow(industryName) {
    const tbody = document.getElementById('industryTableBody');
    if (!tbody) return;
    // Remove existing highlights
    Array.from(tbody.querySelectorAll('tr')).forEach(r => r.style.background = '');
    // Find and highlight matching row
    const rows = Array.from(tbody.rows);
    for (let r of rows) {
        if ((r.cells[0] && r.cells[0].innerText.trim()) === industryName) {
            r.style.background = 'rgba(59,130,246,0.08)';
            // scroll into view
            r.scrollIntoView({behavior:'smooth', block:'center'});
            break;
        }
    }
    // Also highlight corresponding bar in chart
    highlightChartBar(industryName);
}

function highlightChartBar(industryName) {
    if (!marketChartInstance || !currentIndustryFullLabels || !currentIndustryFullLabels.length) return;
    const idx = currentIndustryFullLabels.indexOf(industryName);
    if (idx === -1) return;

    // Reset colors
    const ds = marketChartInstance.data.datasets[0];
    const base = ds.data.map(() => 'rgba(59,130,246,0.85)');
    base[idx] = 'rgba(244,114,182,0.95)'; // highlight color

    ds.backgroundColor = base;
    selectedIndustryIndex = idx;

    marketChartInstance.update();

    // Show tooltip for selected bar
    try {
        marketChartInstance.setActiveElements([{datasetIndex:0, index: idx}]);
        marketChartInstance.tooltip.setActiveElements([{datasetIndex:0, index: idx}], {x:0,y:0});
    } catch (e) {
        // ignore if Chart version doesn't support setActiveElements
    }
}

// Simple client-side fallback extractor for small inputs when SHAP is not available
function generateFallbackFactors(text) {
    if (!text || typeof text !== 'string') return {high: [], low: []};
    const AUTO_KEYWORDS = ['automate','automation','automated','process','generate','calculate','extract','analyze','report','data','compute','synthesize','predict'];
    const HUMAN_KEYWORDS = ['negotiate','empath','empathy','creative','creativity','design','manage','lead','supervise','coach','counsel','persuade','judge','advise'];

    const stop = new Set(['the','and','for','with','that','this','from','your','you','are','is','to','a','an','in','on','of','be']);
    const words = text.toLowerCase().match(/\b[a-z]{3,}\b/g) || [];
    const counts = {};
    words.forEach(w => { if (!stop.has(w)) counts[w] = (counts[w]||0) + 1; });

    const entries = Object.keys(counts).map(w => ({word: w, count: counts[w]})).sort((a,b)=>b.count-a.count);

    const high = [];
    const low = [];
    for (let e of entries.slice(0,12)) {
        let score = Math.min(1, 0.2 + 0.15 * e.count + Math.min(0.5, e.word.length/10));
        if (AUTO_KEYWORDS.includes(e.word)) score = 0.4 + Math.random()*0.6; // stronger positive
        if (HUMAN_KEYWORDS.includes(e.word)) score = -(0.4 + Math.random()*0.6); // stronger negative

        if (score > 0) high.push({word: e.word, score: score});
        else low.push({word: e.word, score: score});
    }

    // Ensure we return up to 8 each
    return {high: high.slice(0,8), low: low.slice(0,8)};
}

// ==========================================
// 2. MARKET STATS LOGIC (Index & Leaderboard)
// ==========================================

// Reset table highlighting when clicking blank areas
function resetTableHighlight() {
    const tbody = document.getElementById('industryTableBody');
    if (!tbody) return;
    Array.from(tbody.querySelectorAll('tr')).forEach(r => r.style.background = '');
    
    // Reset chart colors to original
    if (marketChartInstance && marketChartInstance.data && marketChartInstance.data.datasets[0]) {
        const ds = marketChartInstance.data.datasets[0];
        if (currentIndustryData && currentIndustryData.length) {
            ds.backgroundColor = currentIndustryData.map(s => s.Automation_Score >= 2.05 ? 'rgba(239,68,68,0.8)' : 'rgba(59,130,246,0.8)');
            marketChartInstance.update();
        }
    }
}

async function loadMarketStats() {
    // Check if elements exist on the current page
    const marketChartEl = document.getElementById('marketChart');
    const leaderboardEl = document.getElementById('leaderboardBody');
    const mainContent = document.querySelector('.main-content');
    
    // We also use this fetch to populate jobs for Pathfinder if on that page
    const isOnPathfinder = !!document.getElementById('currentJob');

    // Fetch if we need chart, leaderboard, OR pathfinder autocomplete data
    if (!marketChartEl && !leaderboardEl && !isOnPathfinder) return;
    
    // Add click listener to main content area to reset highlighting (for stats page)
    if (mainContent && !mainContent.__hasResetListener) {
        mainContent.addEventListener('click', (e) => {
            if (!e.target.closest('table') && !e.target.closest('canvas') && !e.target.closest('.chart-table-row')) {
                resetTableHighlight();
            }
        });
        mainContent.__hasResetListener = true;
    }

    try {
        const res = await fetch(`${API_URL}/api/stats`);
        const data = await res.json();

        // Save for Autocomplete (Pathfinder)
        if(data.top_risky_jobs) {
            allJobsCache = data.top_risky_jobs;
            console.log(`Loaded ${allJobsCache.length} jobs for cache.`);
        }

        // A. Render Chart (Industry-level horizontal bar chart + table)
        if (marketChartEl) {
            if (typeof Chart === 'undefined') return console.error('Chart.js missing');

            // Destroy existing chart instance to avoid duplicates
            if (marketChartInstance) {
                try { marketChartInstance.destroy(); } catch(e) { /* ignore */ }
                marketChartInstance = null;
            }

            const industryData = (data.industry_stats && data.industry_stats.length) ? data.industry_stats : [];

            if (industryData.length) {
                // Sort industries by Automation_Score descending (highest risk first)
                const sortedIndustryData = industryData.slice().sort((a,b) => (b.Automation_Score || 0) - (a.Automation_Score || 0));
                // Prepare labels with truncation for display, keep full names for tooltips
                const fullLabels = sortedIndustryData.map(i => i.Industry);
                const labels = fullLabels.map(l => truncateLabel(l, 30));
                const scores = sortedIndustryData.map(i => i.Automation_Score);
                const counts = sortedIndustryData.map(i => i.count || 0);

                // store globals for interactions
                currentIndustryData = sortedIndustryData;
                currentIndustryFullLabels = fullLabels;
                selectedIndustryIndex = null;

                marketChartInstance = new Chart(marketChartEl, {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Avg Automation Score',
                            data: scores,
                            backgroundColor: scores.map(s => s >= 2.05 ? 'rgba(239,68,68,0.8)' : 'rgba(59,130,246,0.8)'),
                            borderColor: scores.map(s => s >= 2.05 ? '#dc2626' : '#2563eb'),
                            borderWidth: 2,
                            borderRadius: 8,
                            hoverBackgroundColor: scores.map(s => s >= 2.05 ? 'rgba(239,68,68,1)' : 'rgba(59,130,246,1)'),
                            hoverBorderWidth: 3
                        }]
                    },
                    options: {
                        indexAxis: 'y',
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { display: false },
                            title: { 
                                display: true, 
                                text: 'AI Automation Risk by Industry',
                                font: { size: 16, weight: '600' },
                                color: '#1e293b',
                                padding: { top: 10, bottom: 20 }
                            },
                            tooltip: {
                                backgroundColor: 'rgba(0,0,0,0.8)',
                                padding: 12,
                                cornerRadius: 8,
                                titleFont: { size: 13, weight: 'bold' },
                                bodyFont: { size: 12 },
                                callbacks: {
                                    title: (items) => {
                                        if (!items || !items.length) return '';
                                        const idx = items[0].dataIndex;
                                        return fullLabels[idx];
                                    },
                                    label: (ctx) => `Score: ${ctx.parsed.x.toFixed(2)}`,
                                    footer: (items) => {
                                        if (!items || items.length === 0) return '';
                                        const idx = items[0].dataIndex;
                                        return `Jobs: ${counts[idx]}`;
                                    }
                                }
                            }
                        },
                        scales: {
                            x: { 
                                beginAtZero: true, 
                                title: { display: true, text: 'Automation Score', font: { size: 12, weight: '500' } },
                                grid: { display: true, drawBorder: true, color: 'rgba(0,0,0,0.05)' }
                            },
                            y: { 
                                ticks: { autoSkip: false, font: { size: 11 } }, 
                                title: { display: false },
                                grid: { display: false }
                            }
                        },
                        onClick: (evt, elements) => {
                            // highlight corresponding table row when bar clicked
                            if (elements && elements.length) {
                                const idx = elements[0].index;
                                const industryName = fullLabels[idx];
                                highlightIndustryRow(industryName);
                            }
                        }
                    }
                });

                // Populate the right-hand table (all industries) sorted by score
                renderIndustryTable(sortedIndustryData);
            }
        }

        // B. Render Leaderboard (Only if table body exists)
        if(leaderboardEl) {
            leaderboardEl.innerHTML = data.leaderboard.map((j, i) => `
                <tr>
                    <td>#${i+1}</td>
                    <td>${j['O*NET-SOC Code']}</td>
                    <td style="color:#ef4444; font-weight:bold;">${j.Automation_Score.toFixed(2)}</td>
                </tr>
            `).join('');
        }
    } catch (e) {
        console.error('Error loading stats:', e);
    }
}

// ==========================================
// 3. RISK CLOUD LOGIC (D3.js)
// ==========================================
async function loadCloud(type) {
    const container = document.getElementById('forceGraphCanvas');
    if (!container) return; // Exit if not on risk cloud page

    console.log("Loading Cloud:", type);

    // Update Buttons
    const btnHigh = document.getElementById('btnCloudHigh');
    const btnLow = document.getElementById('btnCloudLow');
    if (btnHigh && btnLow) {
        btnHigh.className = type === 'high' ? 'toggle-btn active' : 'toggle-btn';
        btnLow.className = type === 'low' ? 'toggle-btn active' : 'toggle-btn';
    }

    try {
        const res = await fetch(`${API_URL}/api/cloud?type=${type}`);
        const rawData = await res.json(); 

        // Prepare Data for D3
        const nodes = rawData.map(d => ({
            id: d[0],
            r: Math.max(d[1] * 4, 15), // Scale size
            value: d[1]
        }));

        // Setup D3
        container.innerHTML = '';
        const width = container.clientWidth;
        const height = container.clientHeight || 500;
        const color = type === 'high' ? '#ef4444' : '#10b981';

        if (simulation) simulation.stop();
        
        if (typeof d3 === 'undefined') {
            container.innerHTML = '<p style="color:red">D3.js not loaded.</p>';
            return;
        }

        const svg = d3.select("#forceGraphCanvas")
            .append("svg")
            .attr("width", "100%")
            .attr("height", "100%")
            .attr("viewBox", [0, 0, width, height]);

        const node = svg.append("g")
            .selectAll("circle")
            .data(nodes)
            .join("g")
            .attr("class", "node")
            .call(d3.drag()
                .on("start", (e, d) => { if(!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
                .on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; })
                .on("end", (e, d) => { if(!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }));

        node.append("circle")
            .attr("r", d => d.r)
            .attr("fill", color)
            .attr("fill-opacity", 0.7)
            .attr("stroke", "#fff")
            .attr("stroke-width", 1.5);

        node.append("text")
            .text(d => d.id)
            .attr("text-anchor", "middle")
            .attr("dy", ".35em")
            .attr("font-size", d => Math.min(d.r, 14) + "px")
            .attr("fill", "#1e293b")
            .style("pointer-events", "none")
            .style("font-weight", "bold");

        simulation = d3.forceSimulation(nodes)
            .force("charge", d3.forceManyBody().strength(5))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collide", d3.forceCollide().radius(d => d.r + 2).iterations(2))
            .on("tick", () => {
                node.attr("transform", d => {
                    d.x = Math.max(d.r, Math.min(width - d.r, d.x));
                    d.y = Math.max(d.r, Math.min(height - d.r, d.y));
                    return `translate(${d.x},${d.y})`;
                });
            });

    } catch (e) {
        console.error("Cloud error:", e);
        container.innerHTML = `<p style="color:red; text-align:center;">Failed to load data.</p>`;
    }
}

// ==========================================
// 4. PATHFINDER LOGIC (Autocomplete, Search, Speedometer, Cards)
// ==========================================

function setupPathfinderAutocomplete() {
    const input = document.getElementById("currentJob");
    const list = document.getElementById("jobList");
    if (!input || !list) return;

    let currentFocus = -1;

    // Use allJobsCache populated by loadMarketStats
    input.addEventListener('input', function(e) {
        const val = this.value;
        closeAllLists();
        if (!val) return false;
        currentFocus = -1;

        // Filter from cache
        const matches = allJobsCache.filter(job => 
            job.Title.toLowerCase().startsWith(val.toLowerCase())
        );

        if (matches.length === 0) return;

        list.classList.add('active');

        matches.slice(0, 10).forEach(job => {
            const div = document.createElement("div");
            div.className = "suggestion-item";
            // Highlight matching part
            div.innerHTML = `<span class="suggestion-match">${job.Title.substr(0, val.length)}</span>${job.Title.substr(val.length)}`;
            div.innerHTML += `<input type='hidden' value="${job.Title}">`;

            div.addEventListener("click", function() {
                input.value = this.getElementsByTagName("input")[0].value;
                closeAllLists();
            });
            
            list.appendChild(div);
        });
    });

    input.addEventListener("keydown", function(e) {
        let x = list.getElementsByTagName("div");
        if (e.keyCode == 40) { // Down
            currentFocus++;
            addActive(x);
        } else if (e.keyCode == 38) { // Up
            currentFocus--;
            addActive(x);
        } else if (e.keyCode == 13) { // Enter
            e.preventDefault();
            if (currentFocus > -1 && x) x[currentFocus].click();
            else findPaths(); 
        }
    });

    function addActive(x) {
        if (!x) return false;
        removeActive(x);
        if (currentFocus >= x.length) currentFocus = 0;
        if (currentFocus < 0) currentFocus = (x.length - 1);
        x[currentFocus].classList.add("focused");
        x[currentFocus].scrollIntoView({block: 'nearest'});
    }

    function removeActive(x) {
        for (let i = 0; i < x.length; i++) x[i].classList.remove("focused");
    }

    function closeAllLists(elmnt) {
        if (elmnt != list && elmnt != input) {
            list.classList.remove('active');
            list.innerHTML = "";
        }
    }

    document.addEventListener("click", function (e) {
        closeAllLists(e.target);
    });
}

// NOTE: This function is expected to be placed in pathfinder.html for immediate use, 
// but is included here for completeness of the project logic.
async function findPaths() {
    const input = document.getElementById('currentJob').value;
    const threshold = document.getElementById('similarityFilter').value;
    
    // Elements for "WOW" factors
    const heroSection = document.getElementById('currentAnalysis');
    const cardsGrid = document.getElementById('cardsGrid'); 
    const scoreText = document.getElementById('heroScore');

    if(!input) return alert("Please enter a job title.");
    
    // 1. Reset UI
    cardsGrid.innerHTML = '<div class="loader"></div>';
    if(heroSection) heroSection.style.display = 'none'; 
    if(scoreText) scoreText.innerText = '0.00'; // Reset counter

    try {
        // 2. PARALLEL FETCH: Get Stats for current job (Hero) AND Recommendations (Cards)
        const [statsRes, recRes] = await Promise.all([
            fetch(`${API_URL}/api/compare`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ query: input })
            }),
            fetch(`${API_URL}/api/recommend`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ job_title: input, threshold: parseFloat(threshold) })
            })
        ]);

        const statsData = await statsRes.json();
        const recData = await recRes.json();

        // 3. RENDER HERO SECTION (Text Score)
        if (statsData.found && heroSection) {
            heroSection.style.display = 'flex';
            
            // --- DIRECT ASSIGNMENT LOGIC ---
            let risk = parseFloat(statsData.risk);
            if (isNaN(risk)) risk = 0.0;
            
            // Text Updates
            document.getElementById('heroTitle').innerText = statsData.title;
            if (scoreText) scoreText.innerText = risk.toFixed(2);
            
            // Color & Badge Logic
            const MEDIAN_THRESHOLD = 2.05;
            const isHighRisk = risk >= MEDIAN_THRESHOLD;

            let color = isHighRisk ? "#ef4444" : "#10b981"; // Red or Green
            let label = isHighRisk ? "HIGH RISK" : "LOW RISK";
            let bg = isHighRisk ? "rgba(239, 68, 68, 0.2)" : "rgba(16, 185, 129, 0.2)";
            
            const badge = document.getElementById('heroBadge');

            if(badge) {
                badge.innerText = label;
                badge.style.backgroundColor = bg;
                badge.style.color = color;
            }
            if(scoreText) scoreText.style.color = color;
        }

        // 4. RENDER RECOMMENDATION FLIP CARDS
        if(!recData || recData.length === 0) {
            cardsGrid.innerHTML = `
                <div class="glass-card" style="text-align:center; padding:30px;">
                    <p style="color:#ef4444; font-weight:600;">No matches found.</p>
                    <p style="font-size:0.9rem; color:#64748b;">Try a broader title or lower threshold.</p>
                </div>`;
            return;
        }

        let html = '<div class="results-grid">';
        recData.forEach(job => {
            const taskDescription = job.task || "No representative task available.";
            
            html += `
                <div class="path-card-container">
                    <div class="path-card-inner">
                        <div class="path-card-front">
                            <div class="match-badge">${job.match_score.toFixed(0)}% Match</div>
                            <h4 style="margin:0; color:#1e293b; padding:0 10px;">${job.title}</h4>
                            <p style="margin:5px 0 15px 0; color:#64748b; font-size:0.85rem;">Safe Alternative</p>
                            <div class="safe-score">${job.score.toFixed(2)}</div>
                            <div style="font-size:0.8rem; font-weight:600; color:#10b981;">LOW RISK SCORE</div>
                        </div>
                        <div class="path-card-back">
                            <h3 style="margin-top:0; font-size:1.1rem; border-bottom:1px solid rgba(255,255,255,0.3); padding-bottom:8px; width:100%;">
                                Representative Task
                            </h3>
                            <div class="task-scroll">${taskDescription}</div>
                            <button class="view-btn">Full Details</button>
                        </div>
                    </div>
                </div>`;
        });
        html += '</div>';
        cardsGrid.innerHTML = html;

    } catch(e) {
        console.error('Error in findPaths:', e);
        cardsGrid.innerHTML = '<p style="text-align:center; color:red;">System Error: Check Server Console</p>';
    }
}

// ==========================================
// 5. INITIALIZATION
// ==========================================

const loadLeaderboard = loadMarketStats; // Alias

window.addEventListener('DOMContentLoaded', () => {
    // 1. Predictor Page Prefill
    try {
        const params = new URLSearchParams(window.location.search);
        const q = params.get('q') || params.get('task');
        if (q) {
            const inputEl = document.getElementById('taskInput');
            const analyzeBtn = document.getElementById('analyzeBtn');
            if (inputEl) {
                inputEl.value = decodeURIComponent(q);
                if (typeof analyzeTask === 'function') {
                    setTimeout(analyzeTask, 250);
                }
            }
        }
    } catch (e) { console.error('Prefill error:', e); }

    // 2. Pathfinder Page Setup
    const pathfinderInput = document.getElementById('currentJob');
    if (pathfinderInput) {
        // We reuse loadMarketStats to fetch the job list for autocomplete
        loadMarketStats().then(() => {
            setupPathfinderAutocomplete();
            // Attach findPaths function to the button if needed (assuming inline HTML does this via onclick)
            const findBtn = document.querySelector('.find-btn');
            if (findBtn && !findBtn.onclick) {
                findBtn.addEventListener('click', findPaths);
            }
        });
    }

    // 3. Other Pages
    loadMarketStats(); // Checks internally if elements exist

});
