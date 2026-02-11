const CLS_CONFIG = {
    BACKEND_URL: window.location.origin,
    CLASSIFY_ENDPOINT: '/api/v1/classification',
};

const INDICATOR_LABELS = {
    likely_regulatory_pathway: 'Likely Regulatory Pathway',
    initial_ground_risk_orientation: 'Initial Ground Risk Orientation',
    initial_air_risk_orientation: 'Initial Air Risk Orientation',
    expected_assessment_depth: 'Expected Assessment Depth',
};

const VALUE_STYLES = {
    very_low: { label: 'Very Low', className: 'level-low' },
    low: { label: 'Low', className: 'level-low' },
    medium: { label: 'Medium', className: 'level-medium' },
    high: { label: 'High', className: 'level-high' },
    very_high: { label: 'Very High', className: 'level-high' },
};

function formatValue(raw) {
    if (!raw || typeof raw !== 'string') {
        return { label: '', className: '' };
    }
    const key = raw.trim().toLowerCase();
    if (VALUE_STYLES[key]) {
        return VALUE_STYLES[key];
    }
    // Fallback: title-case the string
    const label = raw
        .replace(/_/g, ' ')
        .split(' ')
        .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
        .join(' ');
    return { label, className: '' };
}

const clsState = {
    userId: '',
    userName: 'Guest',
    storageKey: 'skysafe-classify-user',
    isWaiting: false,
    abortController: null,
    isStopped: false,
};

const els = {
    userId: document.getElementById('cls-user-id'),
    userName: document.getElementById('cls-user-name'),
    mass: document.getElementById('mass'),
    vlos: document.getElementById('vlos'),
    ground: document.getElementById('ground'),
    airspace: document.getElementById('airspace'),
    altitude: document.getElementById('altitude'),
    runBtn: document.getElementById('cls-run-btn'),
    runBtn2: document.getElementById('cls-run-btn-2'),
    resetBtn: document.getElementById('cls-reset-btn'),
    stopBtn: document.getElementById('cls-stop-btn'),
    loading: document.getElementById('cls-loading'),
    results: document.getElementById('indicator-results'),
    sources: document.getElementById('cls-sources'),
    menuToggle: document.getElementById('menu-toggle'),
    menuPanel: document.getElementById('menu-panel'),
    menuHome: document.getElementById('menu-home'),
    menuChat: document.getElementById('menu-chat'),
    menuClassification: document.getElementById('menu-classification'),
};

function persistUserInfo() {
    setCookie('skysafe_user_id', clsState.userId);
    setCookie('skysafe_user_name', clsState.userName);
    try {
        localStorage.setItem(clsState.storageKey, JSON.stringify({
            userId: clsState.userId,
            userName: clsState.userName,
        }));
    } catch (e) {
        console.debug('LocalStorage unavailable for classification user');
    }
}

function setCookie(name, value, days = 30) {
    const expires = new Date(Date.now() + days * 864e5).toUTCString();
    document.cookie = `${name}=${encodeURIComponent(value)}; expires=${expires}; path=/`;
}

function getCookie(name) {
    return document.cookie
        .split('; ')
        .map((v) => v.split('='))
        .reduce((acc, [k, v]) => (k === name ? decodeURIComponent(v || '') : acc), null);
}

function clearUserCookies() {
    document.cookie = `skysafe_user_id=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/`;
    document.cookie = `skysafe_user_name=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/`;
}

function hydrateUserInfo() {
    let stored = null;
    try {
        stored = JSON.parse(localStorage.getItem(clsState.storageKey) || 'null');
    } catch (e) {
        stored = null;
    }

    const cookieId = getCookie('skysafe_user_id');
    const cookieName = getCookie('skysafe_user_name');

    clsState.userId = cookieId || stored?.userId || `user_${Date.now()}`;
    clsState.userName = cookieName || stored?.userName || 'Guest';

    els.userId.value = clsState.userId;
    els.userName.value = clsState.userName;
}

function syncUser() {
    clsState.userId = (els.userId.value || '').trim() || clsState.userId || `user_${Date.now()}`;
    clsState.userName = (els.userName.value || '').trim() || 'Guest';
    els.userId.value = clsState.userId;
    els.userName.value = clsState.userName;
    persistUserInfo();
}

function showLoading() {
    els.loading.classList.remove('hidden');
    els.runBtn.disabled = true;
    els.runBtn2.disabled = true;
}

function hideLoading() {
    els.loading.classList.add('hidden');
    els.runBtn.disabled = false;
    els.runBtn2.disabled = false;
}

function placeholder(container, text, isError = false) {
    const className = isError ? 'placeholder-text error-text' : 'placeholder-text';
    container.innerHTML = `<p class="${className}">${text}</p>`;
}

/**
 * Extract user-friendly error message from server response
 * Server returns: { error: { code, message, details? } }
 */
async function extractErrorMessage(response, fallbackStatus) {
    try {
        const data = await response.json();
        if (data && data.error) {
            const { code, message, details } = data.error;
            let msg = message || code || 'Unknown error';
            if (details) {
                msg += ` (${typeof details === 'string' ? details : JSON.stringify(details)})`;
            }
            return msg;
        }
        return `Server error (${fallbackStatus})`;
    } catch {
        return `Server error (${fallbackStatus})`;
    }
}

function renderSources(sources) {
    // Server returns { sources: { indicator_name: [...], ... } }
    let list = [];
    if (Array.isArray(sources)) {
        list = sources;
    } else if (sources && typeof sources === 'object') {
        // Flatten all source arrays from all indicators
        Object.values(sources).forEach((arr) => {
            if (Array.isArray(arr)) {
                list.push(...arr);
            }
        });
        // De-duplicate
        list = [...new Set(list)];
    }
    if (!list.length) {
        placeholder(els.sources, 'No sources returned.');
        return;
    }
    els.sources.innerHTML = '';
    list.forEach((src, index) => {
        const sourceItem = document.createElement('div');
        sourceItem.className = 'source-item';
        sourceItem.innerHTML = `
            <div class="source-item-title">Source ${index + 1}</div>
            <div class="source-item-meta">${escapeHtml(src)}</div>
        `;
        els.sources.appendChild(sourceItem);
    });
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return String(text).replace(/[&<>"']/g, m => map[m]);
}

function renderResults(results) {
    if (!results || !results.length) {
        placeholder(els.results, 'No indicators returned yet.');
        return;
    }
    els.results.innerHTML = '';
    results.forEach((res) => {
        const name = res.name || 'Unknown indicator';
        const title = INDICATOR_LABELS[name] || name;
        const { label: valueLabel, className } = formatValue(res.value || '');
        const card = document.createElement('div');
        card.className = 'indicator-card';
        card.innerHTML = `
            <div class="indicator-title">${title}</div>
            <div class="indicator-value ${className}">${valueLabel}</div>
            <div class="indicator-expl">${res.explanation || ''}</div>
        `;
        els.results.appendChild(card);
    });
}

function normalizeResults(payload) {
    // Server returns { indicators: { name: {name, value, explanation}, ... } }
    if (payload && typeof payload === 'object' && payload.indicators && typeof payload.indicators === 'object' && !Array.isArray(payload.indicators)) {
        // Convert object to array
        return Object.entries(payload.indicators).map(([key, val]) => ({
            name: key,
            value: val?.value || '',
            explanation: val?.explanation || '',
        }));
    }
    if (Array.isArray(payload)) return payload;
    if (payload && typeof payload === 'object') {
        if (Array.isArray(payload.results)) return payload.results;
        if (Array.isArray(payload.indicators)) return payload.indicators;
    }
    return [];
}

function buildPayload() {
    return {
        maximum_takeoff_mass_category: els.mass.value,
        vlos_or_bvlos: els.vlos.value,
        ground_environment: els.ground.value,
        airspace_type: els.airspace.value,
        maximum_altitude_category: els.altitude.value,
        indicators: Object.keys(INDICATOR_LABELS),
    };
}

async function runIndicators() {
    syncUser();
    if (!clsState.userId) {
        alert('Please provide a User ID.');
        return;
    }
    
    // Create new AbortController for this request
    clsState.abortController = new AbortController();
    clsState.isStopped = false;
    
    showLoading();
    // Clear previous results
    els.results.innerHTML = '';
    placeholder(els.sources, '');
    
    try {
        const resp = await fetch(`${CLS_CONFIG.BACKEND_URL}${CLS_CONFIG.CLASSIFY_ENDPOINT}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(buildPayload()),
            signal: clsState.abortController.signal,
        });
        
        // Check if stopped during fetch
        if (clsState.isStopped) {
            return;
        }
        
        if (!resp.ok) {
            const errMsg = await extractErrorMessage(resp, resp.status);
            throw new Error(errMsg);
        }
        
        const data = await resp.json();
        
        // Check if stopped before rendering
        if (clsState.isStopped) {
            return;
        }
        
        const results = normalizeResults(data);
        
        if (!results.length) {
            placeholder(els.results, 'No indicators returned. The server response was empty.', true);
        } else {
            renderResults(results);
        }
        
        renderSources(data.sources || {});
    } catch (err) {
        // Don't show error if request was aborted by user
        if (err.name === 'AbortError' || clsState.isStopped) {
            placeholder(els.results, 'Request stopped by user.', false);
            placeholder(els.sources, '');
            return;
        }
        
        console.error('Classification error:', err);
        
        let userMessage;
        if (err.name === 'TypeError' && err.message.includes('fetch')) {
            userMessage = '⚠️ Unable to connect to the server. Please check your connection and try again.';
        } else {
            userMessage = `⚠️ ${err.message}`;
        }
        
        placeholder(els.results, userMessage, true);
        placeholder(els.sources, 'No sources due to error.');
    } finally {
        clsState.abortController = null;
        hideLoading();
    }
}

function stopIndicators() {
    if (clsState.abortController) {
        clsState.isStopped = true;
        clsState.abortController.abort();
        // Don't nullify here - let the finally block handle it
    }
    hideLoading();
}


function resetForm() {
    els.mass.value = 'lt_25kg';
    els.vlos.value = 'VLOS';
    els.ground.value = 'sparsely_populated';
    els.airspace.value = 'uncontrolled';
    els.altitude.value = 'gt_50m_le_120m';
}


function init() {
    hydrateUserInfo();
    syncUser();
    resetForm();

    els.runBtn.addEventListener('click', runIndicators);
    els.runBtn2.addEventListener('click', runIndicators);
    els.resetBtn.addEventListener('click', resetForm);
    els.stopBtn.addEventListener('click', stopIndicators);
    els.userId.addEventListener('change', syncUser);
    els.userName.addEventListener('change', syncUser);

    // Menu interactions
    els.menuToggle.addEventListener('click', () => {
        els.menuPanel.classList.toggle('open');
    });
    els.menuHome.addEventListener('click', () => window.location.href = '/home');
    els.menuChat.addEventListener('click', () => window.location.href = '/chat');
    els.menuClassification.addEventListener('click', () => window.location.href = '/classification');
    document.addEventListener('click', (e) => {
        if (!els.menuPanel.contains(e.target) && !els.menuToggle.contains(e.target)) {
            els.menuPanel.classList.remove('open');
        }
    });
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
