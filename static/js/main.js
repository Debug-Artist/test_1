/**
 * ============================================================================
 * AGRONOMIC MOMENTUM OPTIMIZATION PLATFORM
 * Client-Side Application Controller: Interactive Telemetry, Attribution & Math
 * ============================================================================
 */

document.addEventListener("DOMContentLoaded", () => {
    initTopLevelTabs();
    initDualControlBindings();
    initPredictionHandler();
    initRetrainHandler();
    initMathTabs();
    initCalculusSandbox();
    initDatasetManager();

    // Trigger initial math rendering across document with slight deferral for CDN scripts
    setTimeout(() => {
        triggerMathTypesetting(document.body);
    }, 150);
});

window.addEventListener("load", () => {
    triggerMathTypesetting(document.body);
});

/**
 * Robust KaTeX typesetting function that:
 * 1. Enables inline $...$ and display $$...$$ delimiters.
 * 2. Unblocks math inside <code> and <pre> tags.
 * 3. Handles errors without throwing exceptions.
 */
function triggerMathTypesetting(container = document.body) {
    if (window.renderMathInElement && container) {
        try {
            window.renderMathInElement(container, {
                delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '$', right: '$', display: false },
                    { left: '\\(', right: '\\)', display: false },
                    { left: '\\[', right: '\\]', display: true }
                ],
                // Allow math to be formatted even inside code / pre / tables
                ignoredTags: ["script", "noscript", "style", "textarea"],
                throwOnError: false
            });
        } catch (e) {
            console.warn("KaTeX rendering notice:", e);
        }
    }
}

/**
 * Top-Level View Switcher:
 * Switches cleanly between the main "Console & Prediction" view and the "Calculus & Math Explainer" view.
 */
function initTopLevelTabs() {
    const navButtons = document.querySelectorAll(".nav-tab-btn");
    const views = document.querySelectorAll(".app-view");

    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetViewId = btn.getAttribute("data-target");

            navButtons.forEach(b => b.classList.remove("active"));
            views.forEach(v => v.classList.remove("active"));

            btn.classList.add("active");
            const targetView = document.getElementById(targetViewId);
            if (targetView) {
                targetView.classList.add("active");
                // Immediately typeset math in the newly visible view
                setTimeout(() => {
                    triggerMathTypesetting(targetView);
                }, 50);
            }
        });
    });
}

function initDualControlBindings() {
    const controls = [
        { slider: "rainfall-range", number: "rainfall-input" },
        { slider: "temp-range", number: "temp-input" },
        { slider: "ph-range", number: "ph-input" },
        { slider: "humidity-range", number: "humidity-input" },
        { slider: "fertilizer-range", number: "fertilizer-input" }
    ];

    controls.forEach(ctrl => {
        const sliderEl = document.getElementById(ctrl.slider);
        const numberEl = document.getElementById(ctrl.number);

        if (sliderEl && numberEl) {
            sliderEl.addEventListener("input", (e) => {
                numberEl.value = e.target.value;
            });
            numberEl.addEventListener("input", (e) => {
                sliderEl.value = e.target.value;
            });
        }
    });

    bindTunerSlider("tuner-epochs", "lbl-epochs", "");
    bindTunerSlider("tuner-lr", "lbl-lr", "");
    bindTunerSlider("tuner-beta", "lbl-beta", "");
}

function bindTunerSlider(sliderId, labelId, prefix) {
    const slider = document.getElementById(sliderId);
    const label = document.getElementById(labelId);
    if (slider && label) {
        slider.addEventListener("input", (e) => {
            label.textContent = prefix + e.target.value;
        });
    }
}

function initPredictionHandler() {
    const form = document.getElementById("prediction-form");
    const predMomEl = document.getElementById("pred-momentum");
    const predGdEl = document.getElementById("pred-gd");
    const btn = document.getElementById("predict-btn");

    if (!form) return;

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        btn.disabled = true;
        btn.innerHTML = `<span class="mono">Computing...</span>`;

        const payload = {
            Rainfall_mm: parseFloat(document.getElementById("rainfall-input").value),
            Temperature_C: parseFloat(document.getElementById("temp-input").value),
            Soil_pH: parseFloat(document.getElementById("ph-input").value),
            Humidity_pct: parseFloat(document.getElementById("humidity-input").value),
            Fertilizer_Used_kg: parseFloat(document.getElementById("fertilizer-input").value)
        };

        try {
            const resp = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!resp.ok) throw new Error("Inference failure");
            const data = await resp.json();

            // 1. Update primary predictions
            if (predMomEl) predMomEl.textContent = data.prediction_momentum.toFixed(2);
            if (predGdEl) predGdEl.textContent = data.prediction_vanilla_gd.toFixed(2);

            // 2. Update baseline yield in breakdown
            const baselineEl = document.getElementById("contrib-baseline-val");
            if (baselineEl && data.baseline_yield !== undefined) {
                baselineEl.textContent = data.baseline_yield.toFixed(2);
            }

            // 3. Update standardized feature vector cells
            if (data.raw_scaled_features && data.raw_scaled_features.length >= 5) {
                data.raw_scaled_features.forEach((val, idx) => {
                    const zCell = document.getElementById(`z-val-${idx}`);
                    if (zCell) {
                        zCell.textContent = (val >= 0 ? `+${val.toFixed(2)}` : val.toFixed(2));
                    }
                });
            }

            // 4. Update Feature Contribution waterfall / bars
            if (data.feature_contributions && data.feature_contributions.length > 0) {
                data.feature_contributions.forEach((feat, idx) => {
                    const barFill = document.getElementById(`bar-fill-${idx}`);
                    const valEl = document.getElementById(`contrib-val-${idx}`);
                    const pctEl = document.getElementById(`contrib-pct-${idx}`);

                    if (valEl) {
                        const sign = feat.contribution_tons >= 0 ? "+" : "";
                        valEl.textContent = `${sign}${feat.contribution_tons.toFixed(2)} ton/ha`;
                        valEl.className = `contrib-val ${feat.direction === 'positive' ? 'pos' : 'neg'}`;
                    }

                    if (barFill) {
                        barFill.style.width = `${Math.min(100, Math.max(8, feat.percentage * 1.5))}%`;
                        barFill.className = `contrib-bar-fill ${feat.direction === 'positive' ? 'fill-pos' : 'fill-neg'}`;
                    }

                    if (pctEl) {
                        pctEl.textContent = `${feat.percentage}%`;
                    }
                });
            }

        } catch (err) {
            console.error("Prediction Error:", err);
            alert("Error computing prediction. Check server connection.");
        } finally {
            btn.disabled = false;
            btn.innerHTML = `
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M5 12h14M12 5l7 7-7 7"/>
                </svg>
                <span>Compute Real-Time Prediction</span>
            `;
        }
    });
}

function initRetrainHandler() {
    const form = document.getElementById("retrain-form");
    const btn = document.getElementById("retrain-btn");
    const plotImg = document.getElementById("convergence-plot");

    if (!form) return;

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        btn.disabled = true;
        btn.innerHTML = `<span>Optimizing across ${document.getElementById("tuner-epochs").value} epochs...</span>`;

        const payload = {
            epochs: parseInt(document.getElementById("tuner-epochs").value),
            learning_rate: parseFloat(document.getElementById("tuner-lr").value),
            momentum_factor: parseFloat(document.getElementById("tuner-beta").value)
        };

        try {
            const resp = await fetch("/api/retrain", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!resp.ok) throw new Error("Optimization routine failed");
            const data = await resp.json();

            // Update loss plot
            if (plotImg && data.plot_b64) {
                plotImg.src = `data:image/png;base64,${data.plot_b64}`;
            }

            // Update loss trajectory summary stats
            document.getElementById("stat-loss-gd").textContent = data.train_mse_gd.toFixed(4);
            document.getElementById("stat-loss-mom").textContent = data.train_mse_mom.toFixed(4);
            document.getElementById("stat-ep-gd").textContent = data.epochs_to_threshold_gd || "--";
            document.getElementById("stat-ep-mom").textContent = data.epochs_to_threshold_mom || "--";

            // Update Hero Badges
            const heroSpeedup = document.getElementById("hero-speedup");
            if (heroSpeedup) heroSpeedup.textContent = `${data.speedup_factor}x Faster`;

            // Update All 4 Evaluation Metrics in Stage 3 Matrix
            if (data.test_mse_mom !== undefined) document.getElementById("metric-mom-mse").textContent = data.test_mse_mom.toFixed(4);
            if (data.test_mse_gd !== undefined) document.getElementById("metric-gd-mse").textContent = data.test_mse_gd.toFixed(4);
            if (data.test_rmse_mom !== undefined) document.getElementById("metric-mom-rmse").textContent = data.test_rmse_mom.toFixed(4);
            if (data.test_rmse_gd !== undefined) document.getElementById("metric-gd-rmse").textContent = data.test_rmse_gd.toFixed(4);
            if (data.test_mae_mom !== undefined) document.getElementById("metric-mom-mae").textContent = data.test_mae_mom.toFixed(4);
            if (data.test_mae_gd !== undefined) document.getElementById("metric-gd-mae").textContent = data.test_mae_gd.toFixed(4);
            if (data.test_r2_mom !== undefined) document.getElementById("metric-mom-r2").textContent = data.test_r2_mom.toFixed(4);
            if (data.test_r2_gd !== undefined) document.getElementById("metric-gd-r2").textContent = data.test_r2_gd.toFixed(4);

        } catch (err) {
            console.error("Retrain error:", err);
            alert("Failed to retrain models.");
        } finally {
            btn.disabled = false;
            btn.innerHTML = `
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M23 4v6h-6M1 20v-6h6"/>
                    <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
                </svg>
                <span>Retrain Models & Benchmark</span>
            `;
        }
    });
}

function initMathTabs() {
    const tabs = document.querySelectorAll(".math-tab");
    const panes = document.querySelectorAll(".tab-pane");

    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            const targetId = tab.getAttribute("data-tab");

            tabs.forEach(t => t.classList.remove("active"));
            panes.forEach(p => p.classList.remove("active"));

            tab.classList.add("active");
            const targetPane = document.getElementById(targetId);
            if (targetPane) {
                targetPane.classList.add("active");
                triggerMathTypesetting(targetPane);
            }
        });
    });
}

function initCalculusSandbox() {
    const calcBtn = document.getElementById("calc-math-btn");
    if (!calcBtn) return;

    calcBtn.addEventListener("click", async () => {
        calcBtn.disabled = true;
        calcBtn.textContent = "Differentiating...";

        const payload = {
            rainfall: parseFloat(document.getElementById("sb-rain").value),
            temperature: parseFloat(document.getElementById("sb-temp").value),
            soil_ph: parseFloat(document.getElementById("sb-ph").value),
            humidity: parseFloat(document.getElementById("sb-hum").value),
            fertilizer: parseFloat(document.getElementById("sb-fert").value),
            actual_yield: parseFloat(document.getElementById("sb-actual").value)
        };

        try {
            const resp = await fetch("/api/calculate_math", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!resp.ok) throw new Error("Calculation failure");
            const data = await resp.json();

            document.getElementById("sb-res-yhat").textContent = data.predicted_yield.toFixed(3);
            document.getElementById("sb-res-e").textContent = (data.residual_error >= 0 ? `+${data.residual_error.toFixed(3)}` : data.residual_error.toFixed(3));
            document.getElementById("sb-res-e2").textContent = data.squared_error.toFixed(4);

            const gradsContainer = document.getElementById("sb-res-grads");
            gradsContainer.innerHTML = `
                <span class="chip">$\\frac{\\partial L}{\\partial w_1}$: <strong class="mono">${data.partial_derivatives_w[0].toFixed(4)}</strong></span>
                <span class="chip">$\\frac{\\partial L}{\\partial w_2}$: <strong class="mono">${data.partial_derivatives_w[1].toFixed(4)}</strong></span>
                <span class="chip">$\\frac{\\partial L}{\\partial w_3}$: <strong class="mono">${data.partial_derivatives_w[2].toFixed(4)}</strong></span>
                <span class="chip">$\\frac{\\partial L}{\\partial w_4}$: <strong class="mono">${data.partial_derivatives_w[3].toFixed(4)}</strong></span>
                <span class="chip">$\\frac{\\partial L}{\\partial w_5}$: <strong class="mono">${data.partial_derivatives_w[4].toFixed(4)}</strong></span>
                <span class="chip chip-bias">$\\frac{\\partial L}{\\partial b}$: <strong class="mono">${data.partial_derivative_b.toFixed(4)}</strong></span>
            `;

            const updateContainer = document.getElementById("sb-res-updates");
            updateContainer.innerHTML = `
                <span class="chip chip-vel">$V_1^{\\text{new}}$: <strong class="mono">${data.simulated_velocity_w[0].toFixed(4)}</strong></span>
                <span class="chip chip-vel">$V_5^{\\text{new}}$: <strong class="mono">${data.simulated_velocity_w[4].toFixed(4)}</strong></span>
                <span class="chip chip-weight">$w_1^{\\text{next}}$: <strong class="mono">${data.simulated_weights_next[0].toFixed(4)}</strong></span>
                <span class="chip chip-weight">$w_5^{\\text{next}}$: <strong class="mono">${data.simulated_weights_next[4].toFixed(4)}</strong></span>
            `;

            // Retype dynamically updated sandbox chips
            triggerMathTypesetting(document.getElementById("sb-results"));

        } catch (err) {
            console.error("Calculus error:", err);
            alert("Calculation failed.");
        } finally {
            calcBtn.disabled = false;
            calcBtn.textContent = "Evaluate Analytical Calculus";
        }
    });
}

function initDatasetManager() {
    const modal = document.getElementById("dataset-modal");
    const openBtn = document.getElementById("open-dataset-modal-btn");
    const closeBtn = document.getElementById("close-dataset-modal-btn");
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("csv-file-input");
    const uploadBtn = document.getElementById("upload-csv-btn");
    const dropzoneText = document.getElementById("dropzone-text");
    const form = document.getElementById("upload-csv-form");

    if (!modal || !openBtn) return;

    openBtn.addEventListener("click", () => {
        modal.classList.add("active");
    });

    closeBtn.addEventListener("click", () => {
        modal.classList.remove("active");
    });

    modal.addEventListener("click", (e) => {
        if (e.target === modal) modal.classList.remove("active");
    });

    if (dropzone && fileInput) {
        dropzone.addEventListener("click", () => fileInput.click());

        dropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropzone.classList.add("dragover");
        });

        dropzone.addEventListener("dragleave", () => {
            dropzone.classList.remove("dragover");
        });

        dropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropzone.classList.remove("dragover");
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                handleFileSelected(fileInput.files[0]);
            }
        });

        fileInput.addEventListener("change", () => {
            if (fileInput.files && fileInput.files.length > 0) {
                handleFileSelected(fileInput.files[0]);
            }
        });
    }

    function handleFileSelected(file) {
        if (!file.name.endsWith(".csv")) {
            alert("Please select a valid CSV (.csv) file.");
            return;
        }
        dropzoneText.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        uploadBtn.disabled = false;
    }

    if (form) {
        form.addEventListener("submit", async (e) => {
            e.preventDefault();
            if (!fileInput.files || fileInput.files.length === 0) return;

            const file = fileInput.files[0];
            const formData = new FormData();
            formData.append("file", file);

            uploadBtn.disabled = true;
            uploadBtn.textContent = "Processing and Training...";

            try {
                const resp = await fetch("/api/dataset/upload", {
                    method: "POST",
                    body: formData
                });

                if (!resp.ok) {
                    const errData = await resp.json();
                    throw new Error(errData.detail || "Upload failed");
                }

                const res = await resp.json();
                alert(`Dataset '${res.filename}' loaded successfully with ${res.rows} rows!\nRetraining complete.`);
                window.location.reload();

            } catch (err) {
                console.error("Dataset upload error:", err);
                alert(`Error uploading CSV: ${err.message}`);
                uploadBtn.disabled = false;
                uploadBtn.textContent = "Upload CSV & Train Models";
            }
        });
    }
}
