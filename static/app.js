const state = {
  images: [],
  result: null,
  view: "overlay",
};

const $ = (selector) => document.querySelector(selector);
const select = $("#image-select");
const preview = $("#preview");
const loading = $("#loading");
const status = $("#status");
const metrics = $("#metrics");

function settings() {
  return {
    max_side: Number($("#resolution").value),
    sensitivity: Number($("#sensitivity").value),
    contrast_scale: Number($("#contrast").value),
    min_component: Number($("#component").value),
  };
}

function setBusy(busy, message = "") {
  loading.hidden = !busy;
  document.querySelectorAll(".controls button").forEach((button) => {
    button.disabled = busy;
  });
  status.textContent = message || (busy ? "Processing" : "Ready");
}

function updateOutputLabels() {
  $("#resolution-value").value = `${$("#resolution").value} px`;
  $("#sensitivity-value").value = `${$("#sensitivity").value}%`;
  $("#contrast-value").value = `${$("#contrast").value} px`;
  $("#component-value").value = `${$("#component").value} px`;
}

function readFileDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener("load", () => resolve(reader.result));
    reader.addEventListener("error", () => reject(new Error("Could not read the selected PNG")));
    reader.readAsDataURL(file);
  });
}

function populateImages(images, selectedName = "") {
  select.replaceChildren();
  images.forEach((image) => {
    const option = document.createElement("option");
    option.value = image.name;
    const prepared = image.ready ? " · ready" : image.prepared ? " · needs review" : "";
    option.textContent = `${image.name.replace(/\.(png|jpe?g)$/i, "")}${prepared}`;
    select.append(option);
  });
  if (selectedName && images.some((image) => image.name === selectedName)) {
    select.value = selectedName;
  }
}

function showView(name) {
  state.view = name;
  document.querySelectorAll(".tabs button").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === name);
  });
  if (state.result) {
    preview.src = state.result.previews[name];
  }
}

function renderMetrics(data) {
  const items = [
    ["experimental_complexity", "Experimental complexity", ""],
    ["normalized_vein_length", "Normalized vein length", ""],
    ["junction_count", "Junctions", ""],
    ["mesh_count", "Closed meshes", ""],
    ["endpoint_count", "Endpoints", ""],
    ["orientation_entropy", "Orientation entropy", ""],
  ];
  metrics.innerHTML = items.map(([key, label, unit]) => `
    <div class="metric">
      <strong>${data[key]}${unit}</strong>
      <span>${label}</span>
    </div>
  `).join("");
}

async function request(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Request failed");
  return payload;
}

async function analyze() {
  if (!select.value) return;
  setBusy(true, "Extracting leaf veins");
  try {
    state.result = await request("/api/analyze", {
      filename: select.value,
      settings: settings(),
    });
    showView(state.view);
    renderMetrics(state.result.metrics);
    status.textContent = `Analyzed · ${state.result.metrics.analysis_width}×${state.result.metrics.analysis_height}`;
  } catch (error) {
    status.textContent = `Error: ${error.message}`;
  } finally {
    setBusy(false, status.textContent);
  }
}

async function exportCurrent() {
  if (!select.value) return;
  setBusy(true, "Exporting current result");
  try {
    const payload = await request("/api/export", {
      filename: select.value,
      settings: settings(),
    });
    status.textContent = "Export complete";
    $("#folder-note").textContent = `Result location: ${payload.destination}`;
  } catch (error) {
    status.textContent = `Error: ${error.message}`;
  } finally {
    setBusy(false, status.textContent);
  }
}

async function exportAll() {
  setBusy(true, "Analyzing all images");
  try {
    const payload = await request("/api/export-all", { settings: settings() });
    status.textContent = `Batch complete · ${payload.count} images`;
    $("#folder-note").textContent = `Result location: ${payload.destination}`;
  } catch (error) {
    status.textContent = `Error: ${error.message}`;
  } finally {
    setBusy(false, status.textContent);
  }
}

async function uploadSpecimen() {
  const file = $("#upload-input").files[0];
  if (!file) {
    $("#upload-note").textContent = "Choose a PNG specimen first.";
    return;
  }
  if (!/\.png$/i.test(file.name)) {
    $("#upload-note").textContent = "Only PNG specimen files are supported.";
    return;
  }
  setBusy(true, "Uploading and preparing specimen");
  $("#upload-note").textContent = "Detecting the leaf mask, veins, nodes, and primary path…";
  try {
    const data = await readFileDataUrl(file);
    const payload = await request("/api/specimens/upload", {
      filename: file.name,
      data,
      settings: settings(),
      include_previews: true,
    });
    const existing = state.images.findIndex((image) => image.name === payload.image.name);
    if (existing >= 0) state.images[existing] = payload.image;
    else state.images.push(payload.image);
    state.images.sort((a, b) => a.name.localeCompare(b.name));
    populateImages(state.images, payload.image.name);
    state.result = payload.result;
    showView(state.view);
    renderMetrics(state.result.metrics);
    const chosen = payload.selected_settings;
    $("#resolution").value = chosen.max_side;
    $("#sensitivity").value = chosen.sensitivity;
    $("#contrast").value = chosen.contrast_scale;
    $("#component").value = chosen.min_component;
    updateOutputLabels();
    status.textContent = payload.image.ready
      ? `Prepared · ${payload.image.graph_edges} graph edges · ${payload.image.main_edges} primary-vein edges`
      : "Prepared, but the primary-vein graph needs parameter review";
    $("#upload-note").textContent = `Saved and analyzed as ${payload.image.name}`;
    $("#folder-note").textContent = `Result location: ${payload.destination}`;
    $("#upload-input").value = "";
  } catch (error) {
    status.textContent = `Error: ${error.message}`;
    $("#upload-note").textContent = error.message;
  } finally {
    setBusy(false, status.textContent);
  }
}

async function initialize() {
  updateOutputLabels();
  const response = await fetch("/api/images");
  const payload = await response.json();
  state.images = payload.images;
  $("#folder-note").textContent = `Output folder: ${payload.output_dir}`;
  populateImages(state.images);
  if (state.images.length) {
    await analyze();
  } else {
    status.textContent = "No PNG or JPG images found";
  }
}

document.querySelectorAll("input[type=range]").forEach((input) => {
  input.addEventListener("input", updateOutputLabels);
});
document.querySelectorAll(".tabs button").forEach((button) => {
  button.addEventListener("click", () => showView(button.dataset.view));
});
select.addEventListener("change", analyze);
$("#analyze").addEventListener("click", analyze);
$("#export").addEventListener("click", exportCurrent);
$("#export-all").addEventListener("click", exportAll);
$("#upload").addEventListener("click", uploadSpecimen);

initialize().catch((error) => {
  status.textContent = `Startup failed: ${error.message}`;
});
