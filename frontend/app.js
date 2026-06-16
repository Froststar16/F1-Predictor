const API_BASE = "http://127.0.0.1:8000";

// No official team logos here on purpose — those are trademarked.
// A color chip tied to each team's known livery color gives the same
// at-a-glance association without redistributing branded assets.
const TEAMS = {
  "Mercedes":        { color: "#00D7B6" },
  "Ferrari":         { color: "#E8002D" },
  "McLaren":         { color: "#FF8000" },
  "Red Bull Racing": { color: "#3671C6" },
  "Red Bull":        { color: "#3671C6" },
  "Alpine":          { color: "#2293D1" },
  "Racing Bulls":    { color: "#6C98FF" },
  "RB":              { color: "#6C98FF" },
  "Haas":            { color: "#B6BABD" },
  "Williams":        { color: "#64C4FF" },
  "Audi":            { color: "#9B0000" },
  "Sauber":          { color: "#9B0000" },
  "Aston Martin":    { color: "#229971" },
  "Cadillac":        { color: "#9C9FA2" },
};

function teamChip(teamId) {
  const color = (TEAMS[teamId] || { color: "#666" }).color;
  return `<span class="team-chip"><span class="team-dot" style="background:${color}"></span>${teamId}</span>`;
}

function loadTrackDiagram(circuitId) {
  const img = document.getElementById("trackDiagram");
  const fallback = document.getElementById("trackDiagramFallback");
  img.style.display = "block";
  fallback.style.display = "none";
  img.onerror = () => {
    img.style.display = "none";
    fallback.style.display = "block";
  };
  img.src = `tracks/${circuitId}.svg`;
}

async function loadCircuits() {
  const res = await fetch(`${API_BASE}/circuits`);
  const circuits = await res.json();
  const select = document.getElementById("circuitSelect");
  select.innerHTML = circuits.map(c => `<option value="${c.id}">${c.name}</option>`).join("");
  if (circuits.length) loadTrackDiagram(circuits[0].id);
}

document.getElementById("circuitSelect").addEventListener("change", (e) => {
  loadTrackDiagram(e.target.value);
});

function probBar(p) {
  const pct = Math.round(p * 100);
  return `<div class="bar-row"><div class="bar-bg"><div class="bar" style="width:${pct}%"></div></div>${pct}%</div>`;
}

async function predict() {
  const circuitId = document.getElementById("circuitSelect").value;
  loadTrackDiagram(circuitId);
  const tbody = document.querySelector("#resultsTable tbody");
  tbody.innerHTML = `<tr><td colspan="6">Loading...</td></tr>`;

  try {
    const res = await fetch(`${API_BASE}/predict/${circuitId}`);
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    const data = await res.json();

    document.getElementById("circuitName").textContent = data.circuit_name;

    tbody.innerHTML = data.predictions.map(p => `
      <tr class="${p.predicted_podium ? "podium" : ""}">
        <td>${p.driver_id}</td>
        <td>${teamChip(p.team_id)}</td>
        <td>${p.predicted_position}</td>
        <td>${probBar(p.podium_probability)}</td>
        <td>${probBar(p.win_probability)}</td>
        <td>${probBar(p.points_probability)}</td>
      </tr>
    `).join("");
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="6" class="error">Couldn't reach the API. Is the backend running? (${err.message})</td></tr>`;
  }
}

document.getElementById("predictBtn").addEventListener("click", predict);
loadCircuits();
