/* Progressive enhancement of native catalog links/disclosures and Documenter search. */
(() => {
    "use strict";
    const start = () => {
        const sidebar = document.getElementById("site-navigation");
        const trigger = document.getElementById("documenter-sidebar-button");
        const main = document.querySelector("#documenter > .docs-main");
        const source = document.getElementById("site-catalog");
        if (!sidebar || !trigger || !main || !source) return;
        const catalog = JSON.parse(source.textContent);
        document.body.classList.add("site-js");
        const narrow = window.matchMedia("(max-width: 1055px)");
        const wideOutline = window.matchMedia("(min-width: 1440px)");
        const closeButton = sidebar.querySelector(".site-nav-close");
        const backdrop = document.createElement("button");
        backdrop.type = "button";
        backdrop.className = "site-nav-backdrop";
        backdrop.tabIndex = -1;
        backdrop.setAttribute("aria-label", "Close navigation");
        sidebar.after(backdrop);
        let previousFocus = trigger;

        function setOpen(open, restore = true) {
            open = narrow.matches && open;
            sidebar.classList.toggle("visible", open);
            document.body.classList.toggle("site-nav-open", open);
            trigger.setAttribute("aria-expanded", String(open));
            trigger.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
            sidebar.inert = narrow.matches && !open;
            main.inert = open;
            if (open) {
                previousFocus = document.activeElement;
                closeButton.focus();
            } else if (restore && narrow.matches) {
                (previousFocus && previousFocus.isConnected ? previousFocus : trigger).focus();
            }
        }
        // Capture prevents Documenter's own jQuery handler from toggling twice.
        trigger.addEventListener("click", (event) => {
            event.preventDefault();
            event.stopImmediatePropagation();
            setOpen(!sidebar.classList.contains("visible"));
        }, true);
        closeButton.addEventListener("click", () => setOpen(false));
        backdrop.addEventListener("click", () => setOpen(false));
        narrow.addEventListener("change", () => setOpen(false, false));
        setOpen(false, false);
        document.addEventListener("keydown", (event) => {
            if (!narrow.matches || !sidebar.classList.contains("visible")) return;
            // Documenter's search modal owns its own keyboard/focus while open.
            if (document.getElementById("search-modal")?.classList.contains("is-active")) return;
            if (event.key === "Escape") {
                event.preventDefault();
                setOpen(false);
            } else if (event.key === "Tab") {
                const focusable = [...sidebar.querySelectorAll("a[href], button, summary, select")]
                    .filter(node => !node.disabled && node.getClientRects().length > 0);
                const first = focusable[0], last = focusable[focusable.length - 1];
                if (event.shiftKey && document.activeElement === first) {
                    event.preventDefault(); last?.focus();
                } else if (!event.shiftKey && document.activeElement === last) {
                    event.preventDefault(); first?.focus();
                }
            }
        });
        sidebar.addEventListener("click", (event) => {
            if (event.target.closest("a[href]") && narrow.matches) setOpen(false, false);
        });
        // Opening search from a drawer releases the article before the modal opens.
        const searchButton = document.getElementById("documenter-search-query");
        let searchOrigin = searchButton;
        let searchWasOpen = false;
        searchButton.addEventListener("click", () => {
            searchOrigin = searchButton;
            if (narrow.matches) setOpen(false, false);
        }, true);
        document.addEventListener("keydown", (event) => {
            if ((event.ctrlKey || event.metaKey) && event.key === "/") {
                searchOrigin = document.activeElement;
                if (narrow.matches) setOpen(false, false);
            }
        }, true);
        function syncSearchFocus() {
            const open = Boolean(document.getElementById("search-modal")?.classList.contains("is-active"));
            if (searchWasOpen && !open) {
                const destination = narrow.matches ? trigger : searchOrigin;
                if (destination?.isConnected) destination.focus();
            }
            searchWasOpen = open;
        }

        const outline = document.querySelector(".site-outline-disclosure");
        if (outline) {
            const adjustOutline = () => { outline.open = wideOutline.matches; };
            adjustOutline();
            wideOutline.addEventListener("change", adjustOutline);
            outline.querySelector("summary").addEventListener("click", (event) => {
                if (wideOutline.matches) event.preventDefault();
            });
        }

        const siteRoot = new URL(catalog.root, document.baseURI);
        function annotateSearch() {
            for (const result of document.querySelectorAll(".search-result-link")) {
                if (result.querySelector(".site-result-context")) continue;
                const url = new URL(result.href, document.baseURI);
                if (url.origin !== siteRoot.origin || !url.pathname.startsWith(siteRoot.pathname)) continue;
                const path = decodeURIComponent(url.pathname.slice(siteRoot.pathname.length));
                const record = catalog.pages[path];
                if (!record) continue;
                const context = document.createElement("div");
                context.className = "site-result-context site-search-meta";
                const type = document.createElement("span");
                type.className = "site-result-type";
                type.textContent = record.type_label;
                context.append(type);
                const topics = (record.topics || []).map(id => catalog.topics[id]).filter(Boolean);
                if (topics.length) {
                    const subject = document.createElement("span");
                    subject.className = "site-result-topics";
                    subject.textContent = topics.join(" · ");
                    context.append(subject);
                }
                result.insertBefore(context, result.children[1] || null);
            }
        }
        // Documenter 1.19 replaces .search-modal-card-body after every query/filter.
        // Observe inserted results, including results rendered after lazy search setup.
        let queued = false;
        new MutationObserver((changes) => {
            if (changes.some(change => change.target.id === "search-modal" ||
                [...change.addedNodes].some(node => node.id === "search-modal"))) syncSearchFocus();
            if (queued || !changes.some(change => [...change.addedNodes].some(node =>
                node.nodeType === Node.ELEMENT_NODE &&
                (node.matches(".search-result-link") || node.querySelector(".search-result-link"))))) return;
            queued = true;
            queueMicrotask(() => { queued = false; annotateSearch(); });
        }).observe(document.body, {childList: true, subtree: true, attributes: true, attributeFilter: ["class"]});
        syncSearchFocus();
        annotateSearch();
    };
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
    else start();
})();
