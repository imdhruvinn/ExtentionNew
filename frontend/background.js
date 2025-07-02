chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (
    changeInfo.status === 'complete' &&
    tab.url &&
    (tab.url.startsWith('http://') || tab.url.startsWith('https://'))
  ) {
    console.log(`Tab updated. Sending URL to backend: ${tab.url}`);

    fetch('http://127.0.0.1:5000/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: tab.url })
    })
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        return response.json();
      })
      .then((data) => {
        const emotions = data.emotions;

        // Only the 6 emotions you care about
        const emotionKeys = ['anger', 'joy', 'sadness', 'fear', 'neutral', 'surprise'];

        chrome.storage.local.get(null, (result) => {
          const updatedStorage = {};
          const currentId = Object.keys(result.id_to_url || {}).length + 1;

          // 1. Save the URL against this ID
          updatedStorage.id_to_url = result.id_to_url || {};
          updatedStorage.id_to_url[currentId] = tab.url;

          // 2. For each emotion, add current ID and value
          for (let key of emotionKeys) {
            updatedStorage[key] = result[key] || {};
            updatedStorage[key][currentId] = emotions[key];
          }

          // 3. Save updated data
          chrome.storage.local.set(updatedStorage, () => {
            console.log(`Saved analysis for ID ${currentId} and URL ${tab.url}`);
          });
        });
      })
      .catch((error) => {
        console.error('Error communicating with backend:', error);
      });
  }
});
