chrome.runtime.onMessage.addListener((message) => {

    if (message.type !== "prediction") return;

    if (message.prediction !== "Phishing") return;

    // Create overlay
    let overlay = document.createElement("div");
    overlay.id = "phishing-overlay";

    overlay.innerHTML = `
        <div class="phishing-warning-banner">
            <div class="warning-icon">⚠️</div>
            <div class="warning-content">
                <h1>Deceptive Site Ahead</h1>
                <p>Our phishing detection system believes this website may be attempting
                to steal sensitive information such as passwords or credit card details.</p>
                <p class="confidence"><strong>Confidence:</strong> ${message.confidence}</p>
            </div>
            <button id="closeWarning" class="close-btn">✕</button>
        </div>
        <div class="warning-actions">
            <button id="backSafety" class="action-btn back-btn">Go Back to Safety</button>
            <button id="falseAlarm" class="action-btn report-btn">Report False Alarm</button>
        </div>
    `;

    document.documentElement.appendChild(overlay);

    // Go back
    document.getElementById("backSafety").onclick = () => {
        window.history.back();
    };
    
    // Close warning via X button
    document.getElementById("closeWarning").onclick = () => {
        overlay.remove();
    };

    // Report false alarm
    document.getElementById("falseAlarm").onclick = () => {

        fetch("http://127.0.0.1:8000/report_false_alarm", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                url: message.url,
                model_name: message.model_name || "AdaptiveCNN"
            })
        });

        const btn = document.getElementById("falseAlarm");
        btn.disabled = true;
        btn.textContent = "Reported";
    };

});