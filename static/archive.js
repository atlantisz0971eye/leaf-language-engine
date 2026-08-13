const state = { entries: [], index: 0, view: "disturbed" };
const viewLabels = {
  original: "Original leaf rubbing",
  clean: "Clean encoding",
  disturbed: "Audio-disturbed hard decode",
  difference: "Pixel difference",
  confidence: "Decode confidence",
};

const byId = (id) => document.getElementById(id);
const image = byId("record-image");
const audio = byId("archive-audio");

function formatText(value) {
  return String(value || "").trim();
}

function setRecord(index, updateHash = true) {
  if (!state.entries.length) return;
  state.index = (index + state.entries.length) % state.entries.length;
  const entry = state.entries[state.index];
  const preferred = entry.images[state.view] ? state.view : "original";
  state.view = preferred;

  image.classList.add("switching");
  const nextSource = entry.images[preferred];
  const preload = new Image();
  preload.onload = () => {
    image.src = nextSource;
    image.alt = `${entry.label} — ${viewLabels[preferred]}`;
    requestAnimationFrame(() => image.classList.remove("switching"));
  };
  preload.src = nextSource;

  byId("record-number").textContent = `ARCHIVE ${String(entry.index).padStart(2, "0")} / ${String(state.entries.length).padStart(2, "0")}`;
  byId("record-carrier").textContent = `LEAF ${entry.leaf} · SOUND ${entry.sound}`;
  byId("record-caption").textContent = viewLabels[preferred];
  byId("source-text").textContent = formatText(entry.sourceText);
  byId("decoded-text").textContent = formatText(entry.decodedText);
  byId("waveform-image").src = entry.images.waveform || "";
  byId("sound-name").textContent = entry.sound;
  byId("metric-fidelity").textContent = `${entry.metrics.fidelity}%`;
  byId("metric-words").textContent = entry.metrics.affectedWords.toLocaleString();
  byId("metric-cells").textContent = entry.metrics.changedCells.toLocaleString();
  byId("metric-confidence").textContent = `${entry.metrics.confidence}%`;

  audio.pause();
  audio.src = entry.audio || "";
  byId("sound-toggle").disabled = !entry.audio;
  byId("sound-toggle").querySelector(".sound-icon").textContent = "▶";

  document.querySelectorAll("[data-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === preferred);
  });
  document.querySelectorAll(".leaf-card").forEach((button, buttonIndex) => {
    button.classList.toggle("active", buttonIndex === state.index);
    button.setAttribute("aria-current", buttonIndex === state.index ? "true" : "false");
  });
  if (updateHash) history.replaceState(null, "", `#${entry.id}`);
}

function buildIndex() {
  const fragment = document.createDocumentFragment();
  state.entries.forEach((entry, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "leaf-card";
    button.setAttribute("aria-label", `Open leaf ${String(entry.index).padStart(2, "0")}: ${entry.label}`);
    const thumb = document.createElement("img");
    thumb.src = entry.images.thumb || entry.images.original;
    thumb.alt = "";
    thumb.loading = "lazy";
    const label = document.createElement("span");
    label.textContent = `Leaf ${String(entry.index).padStart(2, "0")}`;
    button.append(thumb, label);
    button.addEventListener("click", () => {
      setRecord(index);
      if (window.innerWidth < 981) window.scrollTo({ top: 0, behavior: "smooth" });
    });
    fragment.append(button);
  });
  byId("leaf-index").append(fragment);
}

document.querySelectorAll("[data-view]").forEach((button) => {
  button.addEventListener("click", () => {
    state.view = button.dataset.view;
    setRecord(state.index, false);
  });
});
byId("previous-record").addEventListener("click", () => setRecord(state.index - 1));
byId("next-record").addEventListener("click", () => setRecord(state.index + 1));
byId("sound-toggle").addEventListener("click", async () => {
  if (!audio.src) return;
  const icon = byId("sound-toggle").querySelector(".sound-icon");
  if (audio.paused) {
    await audio.play();
    icon.textContent = "Ⅱ";
  } else {
    audio.pause();
    icon.textContent = "▶";
  }
});
audio.addEventListener("ended", () => {
  byId("sound-toggle").querySelector(".sound-icon").textContent = "▶";
});
document.addEventListener("keydown", (event) => {
  if (event.key === "ArrowLeft") setRecord(state.index - 1);
  if (event.key === "ArrowRight") setRecord(state.index + 1);
});

fetch("/static/archive/archive.json")
  .then((response) => {
    if (!response.ok) throw new Error("Archive manifest unavailable");
    return response.json();
  })
  .then((manifest) => {
    state.entries = manifest.entries || [];
    byId("index-count").textContent = `${state.entries.length} records`;
    buildIndex();
    const requested = location.hash.replace("#", "");
    const requestedIndex = Math.max(0, state.entries.findIndex((entry) => entry.id === requested));
    setRecord(requestedIndex, false);
  })
  .catch((error) => {
    byId("record-caption").textContent = error.message;
  });
