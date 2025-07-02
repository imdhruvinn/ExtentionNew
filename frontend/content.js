// Use a guard to prevent the script from running multiple times on the same page.
// This can happen on complex sites like YouTube where the tab updates frequently.
if (typeof contentScriptInjected === 'undefined') {
  // By declaring this with 'var', it becomes a global variable on the page's 'window' object,
  // allowing us to check for its existence on subsequent injections.
  var contentScriptInjected = true;

  // 1. Scrape the text content from the body of the page.
  const pageText = document.body.innerText;

  // 2. Send the scraped text back to the background script.
  if (pageText) {
    chrome.runtime.sendMessage({ text: pageText });
  }
}
