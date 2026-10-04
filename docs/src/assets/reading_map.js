// Keep a keyboard-selected card fully visible inside a narrow reading map.
// The diagram, links, and text outline are usable without this enhancement.
document.addEventListener("focusin", (event) => {
    const card = event.target;
    if (!(card instanceof Element) || !card.matches(".reading-map-node:focus-visible")) return;
    const viewport = card.closest(".reading-map-scroll");
    if (!viewport) return;
    const bounds = viewport.getBoundingClientRect();
    const rect = card.getBoundingClientRect();
    const margin = 10;
    if (rect.left < bounds.left + margin) {
        viewport.scrollLeft += rect.left - bounds.left - margin;
    } else if (rect.right > bounds.right - margin) {
        viewport.scrollLeft += rect.right - bounds.right + margin;
    }
});
