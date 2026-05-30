document.addEventListener("DOMContentLoaded", () => {
    // Initialize Lucide Icons
    lucide.createIcons();

    // ==========================================================================
    // Theme Management (Light / Dark Mode Toggle)
    // ==========================================================================
    const htmlElement = document.documentElement;
    const themeToggleBtn = document.getElementById("theme-toggle");

    // Load persisted theme or default to dark
    const savedTheme = localStorage.getItem("theme") || "dark";
    htmlElement.setAttribute("data-theme", savedTheme);

    themeToggleBtn.addEventListener("click", () => {
        const currentTheme = htmlElement.getAttribute("data-theme");
        const newTheme = currentTheme === "dark" ? "light" : "dark";
        
        htmlElement.setAttribute("data-theme", newTheme);
        localStorage.setItem("theme", newTheme);
    });

    // ==========================================================================
    // Sync Range Sliders and Numeric Inputs
    // ==========================================================================
    const features = ["sepal_length", "sepal_width", "petal_length", "petal_width"];
    
    features.forEach(feature => {
        const slider = document.getElementById(`${feature}_slider`);
        const numInput = document.getElementById(feature);
        const valBadge = document.getElementById(`${feature}_val`);

        const updateValues = (value) => {
            const formattedVal = parseFloat(value).toFixed(1);
            slider.value = formattedVal;
            numInput.value = formattedVal;
            valBadge.textContent = `${formattedVal} cm`;
        };

        // Sync slider changes
        slider.addEventListener("input", (e) => {
            updateValues(e.target.value);
        });

        // Sync numeric input changes
        numInput.addEventListener("change", (e) => {
            let val = parseFloat(e.target.value);
            const min = parseFloat(numInput.min);
            const max = parseFloat(numInput.max);

            if (isNaN(val)) {
                val = parseFloat(slider.value);
            } else if (val < min) {
                val = min;
            } else if (val > max) {
                val = max;
            }
            
            updateValues(val);
        });
    });

    // ==========================================================================
    // Quick Sample / Preset Click Handlers
    // ==========================================================================
    const presetButtons = document.querySelectorAll(".preset-btn");
    
    presetButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            let presetData = "";
            if (btn.hasAttribute("data-setosa")) presetData = btn.getAttribute("data-setosa");
            if (btn.hasAttribute("data-versicolor")) presetData = btn.getAttribute("data-versicolor");
            if (btn.hasAttribute("data-virginica")) presetData = btn.getAttribute("data-virginica");
            
            if (presetData) {
                const values = presetData.split(",").map(Number);
                features.forEach((feature, idx) => {
                    const slider = document.getElementById(`${feature}_slider`);
                    const numInput = document.getElementById(feature);
                    const valBadge = document.getElementById(`${feature}_val`);
                    
                    const val = values[idx];
                    slider.value = val;
                    numInput.value = val;
                    valBadge.textContent = `${val.toFixed(1)} cm`;
                });
                
                // Add preset feedback animation
                btn.style.transform = "scale(0.95)";
                setTimeout(() => btn.style.transform = "", 150);
                
                // Trigger form submission instantly
                document.getElementById("predict-btn").click();
            }
        });
    });

    // ==========================================================================
    // Tab Navigation for Metrics/Diagnostics Panel
    // ==========================================================================
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabPanels = document.querySelectorAll(".tab-panel");

    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            // Remove active states
            tabBtns.forEach(b => b.classList.remove("active"));
            tabPanels.forEach(p => p.classList.remove("active"));

            // Add active state to clicked button and target tab panel
            btn.classList.add("active");
            const targetId = btn.getAttribute("data-tab");
            document.getElementById(targetId).classList.add("active");
        });
    });

    // ==========================================================================
    // Fetch and Populate Model Metadata on Page Load
    // ==========================================================================
    let importanceChart = null;

    const fetchModelMetadata = async () => {
        try {
            const response = await fetch("/api/model-info");
            if (!response.ok) {
                throw new Error("Could not retrieve model metrics.");
            }
            const data = await response.json();
            if (data.success) {
                populateMetrics(data.metadata);
            }
        } catch (error) {
            console.error("Error loading model metadata:", error);
            // Show placeholder error state in diagnostics
            const matrixContainer = document.getElementById("confusion-matrix-grid");
            if (matrixContainer) {
                matrixContainer.innerHTML = `<div class="matrix-loading" style="color: var(--accent-rose)">Failed to load model diagnostics. Make sure the ML model is trained.</div>`;
            }
        }
    };

    const populateMetrics = (metadata) => {
        // 1. Set Accuracy badges and fields
        const accPct = (metadata.accuracy * 100).toFixed(1) + "%";
        document.getElementById("model-accuracy-badge").textContent = accPct;
        document.getElementById("header-accuracy-badge").textContent = `Accuracy: ${accPct}`;
        document.getElementById("overview-accuracy-value").textContent = accPct;
        document.getElementById("metric-train-acc").textContent = (metadata.train_accuracy * 100).toFixed(1) + "%";
        
        // 2. Set algorithm name
        document.getElementById("metric-algorithm").textContent = metadata.algorithm || "Random Forest";
        
        // 3. Set split info and sample size
        document.getElementById("metric-split").textContent = metadata.split_ratio;
        document.getElementById("metric-total-samples").textContent = metadata.sample_sizes.total;

        // 4. Render Chart.js Feature Importances
        renderImportanceChart(metadata.feature_importances);

        // 5. Render Heatmapped Confusion Matrix
        renderConfusionMatrix(metadata.confusion_matrix, metadata.target_names);

        // 6. Populate Model Comparison Table
        if (metadata.model_comparison) {
            renderModelComparison(metadata.model_comparison, metadata.algorithm);
        }

        // 7. Populate Classification Report Table
        renderClassificationReport(metadata.classification_report);

        // 8. Populate Dataset Statistics Table
        renderDatasetStats(metadata.dataset_stats);
    };

    const renderModelComparison = (comparisonList, bestModelName) => {
        const tbody = document.querySelector("#model-comparison-table tbody");
        if (!tbody) return;

        tbody.innerHTML = "";

        comparisonList.forEach(item => {
            const row = document.createElement("tr");
            const isBest = item.model === bestModelName;
            
            const rowStyle = isBest ? ' style="font-weight: 700; background: rgba(16, 185, 129, 0.05);"' : "";
            
            const statusTag = isBest 
                ? `<span class="badge" style="background: linear-gradient(135deg, var(--accent-green) 0%, #059669 100%); font-size: 0.65rem; padding: 0.15rem 0.5rem; text-transform: uppercase;">Active (Best)</span>`
                : `<span style="font-size: 0.75rem; color: var(--text-muted);">Standard</span>`;

            row.innerHTML = `
                <td${rowStyle}>${item.model}</td>
                <td${rowStyle}>${(item.accuracy * 100).toFixed(1)}%</td>
                <td${rowStyle}>${statusTag}</td>
            `;
            tbody.appendChild(row);
        });
    };

    const renderImportanceChart = (importances) => {
        const ctx = document.getElementById("importanceChart").getContext("2d");
        const labels = Object.keys(importances);
        const values = Object.values(importances);

        if (importanceChart) {
            importanceChart.destroy();
        }

        // Get theme-specific colors
        const isDark = document.documentElement.getAttribute("data-theme") === "dark";
        const textClr = isDark ? "#94a3b8" : "#475569";
        const gridClr = isDark ? "rgba(255, 255, 255, 0.05)" : "rgba(0, 0, 0, 0.05)";

        importanceChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    label: "Importance Weight",
                    data: values,
                    backgroundColor: [
                        "rgba(59, 130, 246, 0.75)",  // Sepal Length
                        "rgba(6, 182, 212, 0.75)",   // Sepal Width
                        "rgba(139, 92, 246, 0.75)",  // Petal Length
                        "rgba(244, 63, 94, 0.75)"    // Petal Width
                    ],
                    borderColor: [
                        "rgba(59, 130, 246, 1)",
                        "rgba(6, 182, 212, 1)",
                        "rgba(139, 92, 246, 1)",
                        "rgba(244, 63, 94, 1)"
                    ],
                    borderWidth: 1.5,
                    borderRadius: 6
                }]
            },
            options: {
                indexAxis: "y",
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: isDark ? "rgba(18, 16, 38, 0.95)" : "rgba(255, 255, 255, 0.95)",
                        titleColor: isDark ? "#ffffff" : "#0f172a",
                        bodyColor: isDark ? "#f8fafc" : "#0f172a",
                        borderColor: "rgba(139, 92, 246, 0.2)",
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        min: 0,
                        max: 1.0,
                        grid: { color: gridClr },
                        ticks: { color: textClr, font: { family: "Inter" } }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: textClr, font: { family: "Inter", weight: 600 } }
                    }
                }
            }
        });

        // Watch theme toggles to dynamically recolor chart axes
        const observer = new MutationObserver(() => {
            const dark = htmlElement.getAttribute("data-theme") === "dark";
            const updatedTextClr = dark ? "#94a3b8" : "#475569";
            const updatedGridClr = dark ? "rgba(255, 255, 255, 0.05)" : "rgba(0, 0, 0, 0.05)";
            
            importanceChart.options.scales.x.grid.color = updatedGridClr;
            importanceChart.options.scales.x.ticks.color = updatedTextClr;
            importanceChart.options.scales.y.ticks.color = updatedTextClr;
            importanceChart.update();
        });
        observer.observe(htmlElement, { attributes: true, attributeFilter: ["data-theme"] });
    };

    const renderConfusionMatrix = (matrix, targetNames) => {
        const container = document.getElementById("confusion-matrix-grid");
        if (!container) return;
        
        container.innerHTML = "";
        
        // 1. Top-Left Blank Header Cell
        const cornerCell = document.createElement("div");
        cornerCell.className = "matrix-cell matrix-header";
        cornerCell.innerHTML = "Act \\ Pred";
        container.appendChild(cornerCell);

        // 2. Col Headers (Predicted)
        targetNames.forEach(name => {
            const hCell = document.createElement("div");
            hCell.className = "matrix-cell matrix-header";
            hCell.textContent = name.charAt(0).toUpperCase() + name.slice(1);
            container.appendChild(hCell);
        });

        // 3. Grid Rows
        matrix.forEach((row, rIdx) => {
            // Row Header (Actual)
            const rHeader = document.createElement("div");
            rHeader.className = "matrix-cell matrix-header";
            rHeader.textContent = targetNames[rIdx].charAt(0).toUpperCase() + targetNames[rIdx].slice(1);
            container.appendChild(rHeader);

            // Row Cells
            row.forEach((value, cIdx) => {
                const cell = document.createElement("div");
                
                // Assign visual heatmap intensity
                let intensityClass = "intensity-0";
                if (value > 0) {
                    if (rIdx === cIdx) {
                        // True positives (diagonals) usually higher counts
                        if (value > 8) intensityClass = "intensity-3";
                        else if (value > 4) intensityClass = "intensity-2";
                        else intensityClass = "intensity-1";
                    } else {
                        // Off-diagonals (errors)
                        intensityClass = "intensity-1";
                    }
                }
                
                cell.className = `matrix-cell ${intensityClass}`;
                cell.textContent = value;
                cell.title = `Actual: ${targetNames[rIdx].toUpperCase()}, Predicted: ${targetNames[cIdx].toUpperCase()} (${value} samples)`;
                container.appendChild(cell);
            });
        });
    };

    const renderClassificationReport = (report) => {
        const tbody = document.querySelector("#classification-report-table tbody");
        if (!tbody) return;

        tbody.innerHTML = "";

        const classesToDisplay = ["setosa", "versicolor", "virginica", "macro avg", "weighted avg"];

        classesToDisplay.forEach(classKey => {
            const classMetrics = report[classKey];
            if (!classMetrics) return;

            const row = document.createElement("tr");

            // Format class labels for readability
            let displayName = classKey;
            if (classKey === "setosa" || classKey === "versicolor" || classKey === "virginica") {
                displayName = classKey.charAt(0).toUpperCase() + classKey.slice(1);
            } else if (classKey === "macro avg") {
                displayName = "Macro Average";
            } else if (classKey === "weighted avg") {
                displayName = "Weighted Average";
            }

            // Bold averages for design separation
            const isAvg = classKey.includes("avg");
            const boldStyle = isAvg ? ' style="font-weight: 700; background: rgba(255,255,255,0.015);"' : "";

            row.innerHTML = `
                <td${boldStyle}>${displayName}</td>
                <td${boldStyle}>${(classMetrics.precision * 100).toFixed(1)}%</td>
                <td${boldStyle}>${(classMetrics.recall * 100).toFixed(1)}%</td>
                <td${boldStyle}>${(classMetrics["f1-score"] * 100).toFixed(1)}%</td>
                <td${boldStyle}>${classMetrics.support}</td>
            `;
            tbody.appendChild(row);
        });
    };

    const renderDatasetStats = (stats) => {
        const tbody = document.querySelector("#dataset-stats-table tbody");
        if (!tbody) return;

        tbody.innerHTML = "";

        Object.keys(stats).forEach(featureKey => {
            const featStats = stats[featureKey];
            const row = document.createElement("tr");
            
            row.innerHTML = `
                <td>${featureKey}</td>
                <td>${featStats.min} cm</td>
                <td>${featStats.max} cm</td>
                <td>${featStats.mean} cm</td>
                <td>${featStats.std} cm</td>
            `;
            tbody.appendChild(row);
        });
    };

    // Load metrics initially
    fetchModelMetadata();

    // ==========================================================================
    // Handle Prediction Form Submission
    // ==========================================================================
    const form = document.getElementById("prediction-form");
    const predictBtn = document.getElementById("predict-btn");
    const resultPlaceholder = document.getElementById("result-placeholder");
    const resultDisplay = document.getElementById("result-display");
    
    // Circular Ring Constants
    const confidenceRing = document.getElementById("confidence-ring");
    const radius = confidenceRing.r.baseVal.value;
    const circumference = 2 * Math.PI * radius;
    
    // Set circle dash attributes
    confidenceRing.style.strokeDasharray = `${circumference} ${circumference}`;
    confidenceRing.style.strokeDashoffset = circumference;

    const setRingProgress = (pct) => {
        const offset = circumference - (pct / 100) * circumference;
        confidenceRing.style.strokeDashoffset = offset;
    };

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        // 1. UI Loading State
        predictBtn.disabled = true;
        const originalBtnHTML = predictBtn.innerHTML;
        predictBtn.innerHTML = `
            <span>Processing...</span>
            <div class="loader-spinner" style="width:16px; height:16px; border:2px solid #ffffff; border-top-color:transparent; border-radius:50%; animation:spinSlow 0.8s infinite linear;"></div>
        `;

        // 2. Extract input values
        const payload = {
            sepal_length: parseFloat(document.getElementById("sepal_length").value),
            sepal_width: parseFloat(document.getElementById("sepal_width").value),
            petal_length: parseFloat(document.getElementById("petal_length").value),
            petal_width: parseFloat(document.getElementById("petal_width").value)
        };

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(data.error || "An unknown classification error occurred.");
            }

            // 3. Render Prediction Outcome
            renderPredictionResults(data);

        } catch (error) {
            console.error("Prediction failed:", error);
            alert(`Classification Error: ${error.message}`);
        } finally {
            // Restore button state
            predictBtn.disabled = false;
            predictBtn.innerHTML = originalBtnHTML;
        }
    });

    const renderPredictionResults = (data) => {
        // Toggle view containers
        resultPlaceholder.classList.add("hidden");
        resultDisplay.classList.remove("hidden");

        // 1. Set Species Title
        const speciesTitle = document.getElementById("predicted-species-title");
        speciesTitle.textContent = "Iris " + data.species;
        
        // Remove prior colors
        speciesTitle.style.color = `var(--${data.species.toLowerCase()}-color)`;

        // Update confidence label
        const confidenceTextElement = document.getElementById("predicted-species-confidence");
        const confidenceVal = (data.confidence * 100).toFixed(1);
        confidenceTextElement.textContent = `Confidence: ${confidenceVal}%`;

        // 2. Set Confidence ring progress
        const confidencePctVal = Math.round(data.confidence * 100);
        document.getElementById("confidence-percentage").textContent = `${confidencePctVal}%`;
        setRingProgress(confidencePctVal);
        
        // Set Ring color dynamically matching predicted species
        confidenceRing.style.stroke = `var(--${data.species.toLowerCase()}-color)`;

        // 3. Render Probability Breakdown Bars
        const probabilityObj = data.probabilities;
        Object.keys(probabilityObj).forEach(species => {
            const probVal = Math.round(probabilityObj[species] * 100);
            const probRow = document.getElementById(`prob-${species}`);
            
            if (probRow) {
                // Update text percentage
                probRow.querySelector(".class-percentage").textContent = `${probVal}%`;
                
                // Update fill progress width
                const progressBar = probRow.querySelector(".progress-bar");
                progressBar.style.width = `${probVal}%`;
                
                // Highlight row if it is the predicted one
                if (species.toLowerCase() === data.species.toLowerCase()) {
                    probRow.style.opacity = "1";
                    probRow.querySelector(".class-label").style.color = `var(--${species}-color)`;
                    probRow.querySelector(".class-percentage").style.color = `var(--${species}-color)`;
                } else {
                    probRow.style.opacity = "0.7";
                    probRow.querySelector(".class-label").style.color = "var(--text-secondary)";
                    probRow.querySelector(".class-percentage").style.color = "var(--text-primary)";
                }
            }
        });

        // Trigger subtle animation on the results card container
        const resultsCard = document.getElementById("results-container");
        resultsCard.style.transform = "translateY(-4px)";
        resultsCard.style.boxShadow = `0 16px 48px rgba(139, 92, 246, 0.25)`;
        setTimeout(() => {
            resultsCard.style.transform = "";
            resultsCard.style.boxShadow = "";
        }, 500);

        // Smooth scroll to results on smaller devices
        if (window.innerWidth <= 900) {
            resultsCard.scrollIntoView({ behavior: "smooth" });
        }
    };
});
