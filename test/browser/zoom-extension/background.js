// Test-only extension: Playwright calls Chrome's real tab zoom API in this
// worker. It has access only to the loopback fixture, with no content scripts.
chrome.runtime.onInstalled.addListener(() => {});
