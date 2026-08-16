// --- Split-flap hero title -------------------------------------------------
(function renderFlapTitle() {
  const title = "SATISFACTION FORECAST";
  const el = document.getElementById("flapTitle");
  if (!el) return;
  title.split("").forEach((ch, i) => {
    const tile = document.createElement("span");
    if (ch === " ") {
      tile.className = "flap-tile space";
    } else {
      tile.className = "flap-tile";
      tile.textContent = ch;
      tile.style.animationDelay = `${i * 0.035}s`;
    }
    el.appendChild(tile);
  });
})();

// --- Live slider value readouts --------------------------------------------
document.querySelectorAll('input[type="range"]').forEach((input) => {
  const out = document.querySelector(`.slider-val[data-for="${input.id}"]`);
  if (!out) return;
  const sync = () => { out.textContent = input.value; };
  input.addEventListener("input", sync);
  sync();
});

// --- Submit + predict --------------------------------------------------------
const form = document.getElementById("forecastForm");
const errorEl = document.getElementById("formError");
const stub = document.getElementById("resultStub");
const stampBox = document.getElementById("stampBox");
const stampText = document.getElementById("stampText");
const confidenceValue = document.getElementById("confidenceValue");
const probValue = document.getElementById("probValue");
const resultModel = document.getElementById("resultModel");
const runBtn = form.querySelector(".run-btn");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorEl.textContent = "";
  runBtn.disabled = true;
  const originalLabel = runBtn.innerHTML;
  runBtn.innerHTML = "<span>RUNNING…</span>";

  const payload = Object.fromEntries(new FormData(form).entries());

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      errorEl.textContent = data.error || "Something went wrong. Check the form and try again.";
      return;
    }

    const satisfied = data.is_satisfied;
    stampText.textContent = satisfied ? "SATISFIED" : "AT RISK";
    stampBox.classList.toggle("at-risk", !satisfied);
    confidenceValue.textContent = `${data.confidence}%`;
    probValue.textContent = `${data.satisfied_probability}%`;
    resultModel.textContent = data.model_name.toUpperCase();

    stub.hidden = false;
    stub.scrollIntoView({ behavior: "smooth", block: "center" });
  } catch (err) {
    errorEl.textContent = "Couldn't reach the prediction server. Is the Flask app running?";
  } finally {
    runBtn.disabled = false;
    runBtn.innerHTML = originalLabel;
  }
});
