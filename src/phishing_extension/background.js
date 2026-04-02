function shouldProcessUrl(urlString) {
    if (!urlString) return false;

    let parsedUrl;
    try {
        parsedUrl = new URL(urlString);
    } catch (error) {
        return false;
    }

    if (parsedUrl.protocol !== "http:" && parsedUrl.protocol !== "https:") {
        return false;
    }

    const isGoogleSearchPage =
        parsedUrl.hostname.includes("google.") &&
        parsedUrl.pathname === "/search";

    return !isGoogleSearchPage;
}

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status !== "complete" || !shouldProcessUrl(tab.url)) {
        return;
    }

    const modelName = "AdaptiveCNN";
    fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            url: tab.url,
            model_name: modelName
        })
    })
    .then(res => res.json())
    .then(data => {
        chrome.tabs.sendMessage(tabId, {
            type: "prediction",
            prediction: data.prediction,
            confidence: data.confidence,
            url: tab.url,
            model_name: modelName
        });
    });
});