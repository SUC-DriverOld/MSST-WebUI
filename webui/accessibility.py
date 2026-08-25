"""Shared accessibility enhancements for the Gradio WebUI."""

STATUS_ELEM_CLASS = "msst-a11y-status"


def status_textbox(*, label, **kwargs):
	"""Create a read-only textbox whose updates are announced by the WebUI."""
	import gradio as gr

	kwargs.setdefault("interactive", False)
	elem_classes = kwargs.pop("elem_classes", [])
	if isinstance(elem_classes, str):
		elem_classes = [elem_classes]
	kwargs["elem_classes"] = [*elem_classes, STATUS_ELEM_CLASS]
	return gr.Textbox(label=label, **kwargs)


WEBUI_ACCESSIBILITY_CSS = r"""
.msst-a11y-sr-only {
    position: fixed !important;
    width: 1px !important;
    height: 1px !important;
    padding: 0 !important;
    margin: -1px !important;
    overflow: hidden !important;
    clip: rect(0, 0, 0, 0) !important;
    clip-path: inset(50%) !important;
    white-space: nowrap !important;
    border: 0 !important;
}

.gradio-container :is(
    button,
    input,
    textarea,
    select,
    [role="button"],
    [role="tab"],
    [role="checkbox"],
    [role="radio"],
    [role="slider"]
):focus-visible {
    outline: 3px solid var(--color-accent, #005fcc) !important;
    outline-offset: 2px !important;
}

.gradio-container [role="option"].active {
    outline: 2px solid var(--color-accent, #005fcc);
    outline-offset: -2px;
}

.gradio-container :is(button, input, textarea, select, [tabindex]):focus {
    scroll-margin-block: 5rem;
}

@media (forced-colors: active) {
    .gradio-container :is(
        button,
        input,
        textarea,
        select,
        [role="button"],
        [role="tab"],
        [role="checkbox"],
        [role="radio"],
        [role="slider"]
    ):focus-visible,
    .gradio-container [role="option"].active {
        outline-color: CanvasText !important;
    }
}
"""


WEBUI_ACCESSIBILITY_JS = (
	r"""
() => {
    if (window.__msstAccessibilityInitialized) {
        return;
    }
    window.__msstAccessibilityInitialized = true;

    const getRoot = () => {
        const app = document.querySelector("gradio-app");
        return app && app.shadowRoot ? app.shadowRoot : document;
    };

    const liveRegion = document.createElement("div");
    liveRegion.id = "msst-a11y-live-region";
    liveRegion.className = "msst-a11y-sr-only";
	liveRegion.style.cssText = [
		"position:fixed",
		"width:1px",
		"height:1px",
		"padding:0",
		"margin:-1px",
		"overflow:hidden",
		"clip:rect(0, 0, 0, 0)",
		"clip-path:inset(50%)",
		"white-space:nowrap",
		"border:0"
	].join(";");
    liveRegion.setAttribute("role", "status");
    liveRegion.setAttribute("aria-live", "polite");
    liveRegion.setAttribute("aria-atomic", "true");
    document.body.appendChild(liveRegion);

    let lastAnnouncement = "";
    let announcementTimer = null;
    const announce = (message) => {
        const text = String(message || "").replace(/\s+/g, " ").trim();
        if (!text || text === lastAnnouncement) {
            return;
        }
        lastAnnouncement = text;
        window.clearTimeout(announcementTimer);
        liveRegion.textContent = "";
        announcementTimer = window.setTimeout(() => {
            liveRegion.textContent = text;
        }, 20);
    };

    let dropdownCounter = 0;
    const dropdownState = new WeakMap();
    const optionText = (option) =>
        option.getAttribute("aria-label") || option.textContent || "";

    const syncDropdown = (input) => {
        const wrap = input.closest(".wrap");
        if (!wrap) {
            return;
        }

        let state = dropdownState.get(input);
        if (!state) {
            dropdownCounter += 1;
            const component = input.closest("[id]");
            const baseId = component && component.id
                ? `${component.id}-dropdown`
                : `msst-dropdown-${dropdownCounter}`;
            state = { listboxId: `${baseId}-listbox` };
            dropdownState.set(input, state);
        }

        input.setAttribute("role", "combobox");
        input.setAttribute("aria-haspopup", "listbox");
        input.setAttribute("aria-autocomplete", "list");
        input.setAttribute("aria-controls", state.listboxId);
        input.dataset.msstA11yDropdown = "true";

        const listbox = wrap.querySelector("ul[role='listbox']");
        if (!listbox) {
            input.removeAttribute("aria-activedescendant");
            return;
        }

        listbox.id = state.listboxId;
        const label = input.getAttribute("aria-label");
        if (label) {
            listbox.setAttribute("aria-label", label);
        }

        const options = Array.from(listbox.querySelectorAll("[role='option']"));
        options.forEach((option, index) => {
            const optionIndex = option.dataset.index || String(index);
            option.id = `${state.listboxId}-option-${optionIndex}`;
            option.setAttribute("aria-posinset", String(index + 1));
            option.setAttribute("aria-setsize", String(options.length));
        });

        const activeOption = listbox.querySelector("[role='option'].active");
        if (activeOption) {
            input.setAttribute("aria-activedescendant", activeOption.id);
        } else {
            input.removeAttribute("aria-activedescendant");
        }

        if (!listbox.dataset.msstA11yBound) {
            listbox.dataset.msstA11yBound = "true";
            listbox.addEventListener("pointerover", (event) => {
                const option = event.target.closest("[role='option']");
                if (!option || !listbox.contains(option)) {
                    return;
                }
                input.setAttribute("aria-activedescendant", option.id);
                announce(optionText(option));
            });
            listbox.addEventListener("mousedown", (event) => {
                const option = event.target.closest("[role='option']");
                if (option) {
                    announce(optionText(option));
                }
            });
        }

        if (!input.dataset.msstA11yBound) {
            input.dataset.msstA11yBound = "true";
            input.addEventListener("keydown", (event) => {
                if (["ArrowDown", "ArrowUp", "Home", "End", "PageDown", "PageUp"].includes(event.key)) {
                    window.setTimeout(() => syncDropdown(input), 0);
                }
            });
            input.addEventListener("focus", () => {
                window.setTimeout(() => syncDropdown(input), 0);
            });
        }
    };

    const enhanceDropdowns = (root) => {
        root.querySelectorAll(
            "input[role='listbox'], input[role='combobox'][data-msst-a11y-dropdown='true']"
        ).forEach(syncDropdown);
    };
"""
	r"""
    const enhanceTabs = (root) => {
        root.querySelectorAll("[role='tablist']").forEach((tabList) => {
            const getTabs = () => Array.from(
                tabList.querySelectorAll(":scope > [role='tab']")
            );
            getTabs().forEach((tab) => {
                tab.tabIndex = tab.getAttribute("aria-selected") === "true" ? 0 : -1;
            });

            if (tabList.dataset.msstA11yBound) {
                return;
            }
            tabList.dataset.msstA11yBound = "true";
            tabList.addEventListener("keydown", (event) => {
                const tabs = getTabs();
                const current = event.target.closest("[role='tab']");
                const currentIndex = tabs.indexOf(current);
                if (currentIndex < 0) {
                    return;
                }

                let nextIndex = currentIndex;
                if (event.key === "ArrowRight" || event.key === "ArrowDown") {
                    nextIndex = (currentIndex + 1) % tabs.length;
                } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
                    nextIndex = (currentIndex - 1 + tabs.length) % tabs.length;
                } else if (event.key === "Home") {
                    nextIndex = 0;
                } else if (event.key === "End") {
                    nextIndex = tabs.length - 1;
                } else {
                    return;
                }

                event.preventDefault();
				const nextTab = tabs[nextIndex];
				const nextTabId = nextTab.id;
				nextTab.focus();
				nextTab.click();
				window.setTimeout(() => {
					const updatedTabs = getTabs();
					const selectedTab = updatedTabs.find((tab) =>
						(nextTabId && tab.id === nextTabId) ||
						tab.getAttribute("aria-selected") === "true"
					);
					if (selectedTab) {
						selectedTab.focus();
					}
				}, 0);
            });
        });
    };

    const enhanceIconButtons = (root) => {
        root.querySelectorAll(
            "button:not([aria-label]), [role='button']:not([aria-label])"
        ).forEach((button) => {
            if (!button.textContent.trim() && button.title) {
                button.setAttribute("aria-label", button.title);
            }
        });
    };

    const statusValues = new WeakMap();
    const syncStatuses = (root) => {
        root.querySelectorAll(".msst-a11y-status").forEach((status) => {
            status.setAttribute("role", "status");
            status.setAttribute("aria-live", "polite");
            status.setAttribute("aria-atomic", "true");

            const control = status.querySelector("textarea, input");
            const value = control ? control.value : status.textContent;
            if (!statusValues.has(status)) {
                statusValues.set(status, value);
                return;
            }

            const previous = statusValues.get(status);
            if (value !== previous) {
                statusValues.set(status, value);
                if (value) {
                    announce(value);
                }
            }
        });
    };

	let observer = null;
	let observedRoot = null;
	const observeRoot = () => {
		const root = getRoot();
		if (root !== observedRoot) {
			if (observer) {
				observer.disconnect();
			}
			observer = new MutationObserver(scheduleEnhance);
			observer.observe(root, {
				subtree: true,
				childList: true,
				attributes: true,
				attributeFilter: ["class", "aria-expanded", "aria-selected", "value"]
			});
			observedRoot = root;
		}
		return root;
	};

    let scheduled = false;
    const enhance = () => {
        scheduled = false;
		const root = observeRoot();
        enhanceDropdowns(root);
        enhanceTabs(root);
        enhanceIconButtons(root);
        syncStatuses(root);
    };
    const scheduleEnhance = () => {
        if (!scheduled) {
            scheduled = true;
            window.requestAnimationFrame(enhance);
        }
    };

	window.setInterval(() => syncStatuses(observeRoot()), 500);
    scheduleEnhance();
}
"""
)
