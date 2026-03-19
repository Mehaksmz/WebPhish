chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {

    if (changeInfo.status === "complete" && tab.url.startsWith("http")) {

        const modelName = "BaselineCNN"; 
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

    }

});