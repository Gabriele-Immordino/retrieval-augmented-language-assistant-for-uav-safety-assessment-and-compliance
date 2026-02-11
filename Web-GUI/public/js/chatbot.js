// Configuration
const CONFIG = {
    BACKEND_URL: window.location.origin,
    CHAT_ENDPOINT: '/api/v1/chat',
    CHAT_STREAM_ENDPOINT: '/api/v1/chat/stream',
    CHAT_HISTORY_ENDPOINT: '/api/v1/history',
    CLEAR_HISTORY_ENDPOINT: '/api/v1/history',
};

// State management
const state = {
    userId: '',
    userName: 'Guest',
    chatHistory: [],
    isWaiting: false,
    storageKey: 'skysafe-chat-user',
    preprocessQuery: false,
    reasoningEffort: 'medium',
    useStreaming: true,
    abortController: null,
    isStopped: false,
};

// DOM elements
const elements = {
    chatHistory: document.getElementById('chat-history'),
    loadingIndicator: document.getElementById('loading-indicator'),
    chatForm: document.getElementById('chat-form'),
    userInput: document.getElementById('user-input'),
    sendBtn: document.getElementById('send-btn'),
    stopBtn: document.getElementById('stop-btn'),
    sourcesContainer: document.getElementById('sources-container'),
    userIdInput: document.getElementById('user-id-input'),
    userNameInput: document.getElementById('user-name-input'),
    loadHistoryBtn: document.getElementById('load-history-btn'),
    clearChatBtn: document.getElementById('clear-chat-btn'),
    settingsBtn: document.getElementById('settings-btn'),
    settingsPanel: document.getElementById('settings-panel'),
    settingsClose: document.getElementById('settings-close'),
    preprocessToggle: document.getElementById('preprocess-toggle'),
    reasoningEffort: document.getElementById('reasoning-effort'),
    menuToggle: document.getElementById('menu-toggle'),
    menuPanel: document.getElementById('menu-panel'),
    menuHome: document.getElementById('menu-home'),
    menuChat: document.getElementById('menu-chat'),
    menuClassification: document.getElementById('menu-classification'),
};

/**
 * Persist user identity in localStorage
 */
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

function persistUserInfo() {
    const payload = {
        userId: state.userId,
        userName: state.userName,
    };
    setCookie('skysafe_user_id', state.userId);
    setCookie('skysafe_user_name', state.userName);
    try {
        localStorage.setItem(state.storageKey, JSON.stringify(payload));
    } catch (e) {
        console.debug('LocalStorage unavailable, skipping persist');
    }
}

/**
 * Load user identity from cookies/localStorage or defaults
 */
function hydrateUserInfo() {
    let stored = null;
    try {
        stored = JSON.parse(localStorage.getItem(state.storageKey) || 'null');
    } catch (e) {
        stored = null;
    }

    const cookieId = getCookie('skysafe_user_id');
    const cookieName = getCookie('skysafe_user_name');

    state.userId = cookieId || stored?.userId || `user_${Date.now()}`;
    state.userName = cookieName || stored?.userName || 'Guest';

    elements.userIdInput.value = state.userId;
    elements.userNameInput.value = state.userName;
}

/**
 * Sync state from inputs and persist
 */
function syncUserStateFromInputs() {
    state.userId = (elements.userIdInput.value || '').trim() || state.userId || `user_${Date.now()}`;
    state.userName = (elements.userNameInput.value || '').trim() || 'Guest';
    elements.userIdInput.value = state.userId;
    elements.userNameInput.value = state.userName;
    persistUserInfo();
}



/**
 * Display user message in chat
 */
function displayUserMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user';
    messageDiv.innerHTML = `<div class="message-content">${escapeHtml(message)}</div>`;
    elements.chatHistory.appendChild(messageDiv);
    scrollToBottom();
}

/**
 * Display assistant message in chat (with markdown rendering)
 * Returns the message element for streaming updates
 */
function displayAssistantMessage(message, reasoning = '', forStreaming = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    
    // Create content container (the balloon)
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    if (forStreaming) {
        // For streaming, start with empty content
        contentDiv.innerHTML = '';
        contentDiv.dataset.streaming = 'true';
    } else {
        let htmlContent = '';
        try {
            htmlContent = DOMPurify.sanitize(marked.parse(message));
        } catch (error) {
            console.error('Error rendering markdown:', error);
            htmlContent = escapeHtml(message);
        }
        
        // Build reasoning section inside the balloon at the bottom
        let reasoningHtml = '';
        if (reasoning && reasoning.trim()) {
            let reasoningContent = '';
            try {
                reasoningContent = DOMPurify.sanitize(marked.parse(reasoning));
            } catch {
                reasoningContent = escapeHtml(reasoning);
            }
            reasoningHtml = `
                <details class="reasoning-details">
                    <summary class="reasoning-summary">Reasoning</summary>
                    <div class="reasoning-content">${reasoningContent}</div>
                </details>
            `;
        }
        
        contentDiv.innerHTML = htmlContent + reasoningHtml;
    }
    
    messageDiv.appendChild(contentDiv);
    elements.chatHistory.appendChild(messageDiv);
    scrollToBottom();
    
    return messageDiv;
}

/**
 * Create reasoning collapsible element
 */
function createReasoningElement(reasoning) {
    const details = document.createElement('details');
    details.className = 'reasoning-details';
    
    const summary = document.createElement('summary');
    summary.className = 'reasoning-summary';
    summary.textContent = 'Reasoning';
    
    const content = document.createElement('div');
    content.className = 'reasoning-content';
    
    try {
        content.innerHTML = DOMPurify.sanitize(marked.parse(reasoning));
    } catch {
        content.innerHTML = escapeHtml(reasoning);
    }
    
    details.appendChild(summary);
    details.appendChild(content);
    
    // Add animation listener
    details.addEventListener('toggle', () => {
        if (details.open) {
            content.style.maxHeight = content.scrollHeight + 'px';
        }
    });
    
    return details;
}

/**
 * Update streaming message content
 */
function updateStreamingMessage(messageDiv, text, reasoning = '') {
    const contentDiv = messageDiv.querySelector('.message-content');
    if (!contentDiv) return;
    
    try {
        const htmlContent = marked.parse(text);
        const cleanContent = DOMPurify.sanitize(htmlContent);
        contentDiv.innerHTML = cleanContent;
    } catch {
        contentDiv.innerHTML = escapeHtml(text);
    }
    
    scrollToBottom();
}

/**
 * Finalize streaming message (remove cursor, add reasoning inside balloon)
 */
function finalizeStreamingMessage(messageDiv, finalText, reasoning = '') {
    const contentDiv = messageDiv.querySelector('.message-content');
    if (!contentDiv) return;
    
    // Remove streaming cursor and render final content
    delete contentDiv.dataset.streaming;
    
    let htmlContent = '';
    try {
        htmlContent = DOMPurify.sanitize(marked.parse(finalText));
    } catch {
        htmlContent = escapeHtml(finalText);
    }
    
    // Add reasoning section inside the balloon if available
    let reasoningHtml = '';
    if (reasoning && reasoning.trim()) {
        let reasoningContent = '';
        try {
            reasoningContent = DOMPurify.sanitize(marked.parse(reasoning));
        } catch {
            reasoningContent = escapeHtml(reasoning);
        }
        reasoningHtml = `
            <details class="reasoning-details">
                <summary class="reasoning-summary">Reasoning</summary>
                <div class="reasoning-content">${reasoningContent}</div>
            </details>
        `;
    }
    
    contentDiv.innerHTML = htmlContent + reasoningHtml;
    scrollToBottom();
}

/**
 * Display loading indicator
 */
function showLoading() {
    elements.loadingIndicator.classList.remove('hidden');
    elements.stopBtn.classList.remove('hidden');
    elements.sendBtn.classList.add('hidden');
    scrollToBottom();
}

function hideLoading() {
    elements.loadingIndicator.classList.add('hidden');
    elements.stopBtn.classList.add('hidden');
    elements.sendBtn.classList.remove('hidden');
}

/**
 * Display sources
 */
function displaySources(sources) {
    elements.sourcesContainer.innerHTML = '';
    
    if (!sources || sources.length === 0) {
        elements.sourcesContainer.innerHTML = '<p class="placeholder-text">No sources available</p>';
        return;
    }

    sources.forEach((source, index) => {
        const sourceItem = document.createElement('div');
        sourceItem.className = 'source-item';
        sourceItem.innerHTML = `
            <div class="source-item-title">Source ${index + 1}</div>
            <div class="source-item-meta">${escapeHtml(source)}</div>
        `;
        elements.sourcesContainer.appendChild(sourceItem);
    });
}

/**
 * Scroll chat to bottom
 */
function scrollToBottom() {
    setTimeout(() => {
        elements.chatHistory.scrollTop = elements.chatHistory.scrollHeight;
    }, 0);
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
    return text.replace(/[&<>"']/g, m => map[m]);
}

function formatSubqueryAnswer(index, total, question, answer) {
    if (!index) {
        return answer || '';
    }
    const safeQuestion = (question || '').trim();
    const heading = safeQuestion
        ? `**Sub-query ${index}/${total}: ${safeQuestion}**`
        : `**Sub-query ${index}/${total}**`;
    const body = (answer || '').trim();
    if (!body) {
        return heading;
    }
    return `${heading}\n\n${body}`;
}

/**
 * Extract user-friendly error message from server response or error object
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

/**
 * Display error message in chat with styling
 */
function displayErrorMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant error';
    messageDiv.innerHTML = `<div class="message-content error-content">⚠️ ${escapeHtml(message)}</div>`;
    elements.chatHistory.appendChild(messageDiv);
    scrollToBottom();
}

/**
 * Remove welcome message if it exists
 */
function removeWelcomeMessage() {
    const welcome = elements.chatHistory.querySelector('.welcome-message');
    if (welcome) {
        welcome.remove();
    }
}

function clearChatUI() {
    elements.chatHistory.innerHTML = '';
}

function renderHistoryFromState() {
    clearChatUI();
    if (!state.chatHistory.length) {
        const empty = document.createElement('div');
        empty.className = 'placeholder-text';
        empty.textContent = 'No messages yet for this user';
        elements.chatHistory.appendChild(empty);
        return;
    }

    state.chatHistory.forEach((msg) => {
        if (msg.role === 'user') {
            displayUserMessage(msg.content || '');
        } else if (msg.role === 'assistant') {
            displayAssistantMessage(msg.content || '');
        }
    });
}
    function normalizeHistoryRecords(records) {
        if (!Array.isArray(records)) return [];
        return records
            .filter((r) => r && typeof r === 'object')
            .map((r) => ({
                role: r.role,
                content: typeof r.content === 'string' ? r.content : '',
            }))
            .filter((r) => r.role === 'user' || r.role === 'assistant');
    }

    async function loadChatHistory() {
        syncUserStateFromInputs();
        const userId = state.userId;
        if (!userId) {
            alert('Please provide a User ID to load history.');
            return;
        }

        showLoading();
        try {
            const url = `${CONFIG.BACKEND_URL}${CONFIG.CHAT_HISTORY_ENDPOINT}/${encodeURIComponent(userId)}`;
            const resp = await fetch(url, { method: 'GET' });
            hideLoading();

            if (!resp.ok) {
                const errMsg = await extractErrorMessage(resp, resp.status);
                throw new Error(errMsg);
            }

            const data = await resp.json();
            const records = data.messages || [];
            state.chatHistory = normalizeHistoryRecords(records);
            renderHistoryFromState();
        } catch (err) {
            hideLoading();
            console.error('History error:', err);
        
            let userMessage;
            if (err.name === 'TypeError' && err.message.includes('fetch')) {
                userMessage = 'Unable to connect to the server. Please check your connection.';
            } else {
                userMessage = `Could not load history: ${err.message}`;
            }
            displayErrorMessage(userMessage);
        }
    }

/**
 * Clear chat history from server and UI
 */
async function clearChatHistory() {
    syncUserStateFromInputs();
    const userId = state.userId;
    if (!userId) {
        alert('Please provide a User ID to clear history.');
        return;
    }

    // Ask for confirmation
    if (!confirm('Are you sure you want to delete all chat history for this user? This action cannot be undone.')) {
        return;
    }

    showLoading();
    try {
        const url = `${CONFIG.BACKEND_URL}${CONFIG.CLEAR_HISTORY_ENDPOINT}/${encodeURIComponent(userId)}`;
        const resp = await fetch(url, { method: 'DELETE' });
        hideLoading();

        if (!resp.ok) {
            const errMsg = await extractErrorMessage(resp, resp.status);
            throw new Error(errMsg);
        }

        const data = await resp.json();
        console.log(`Cleared ${data.deleted_count} messages for user ${userId}`);

        // Clear local state and UI
        state.chatHistory = [];
        clearChatUI();

        // Show success message
        const successDiv = document.createElement('div');
        successDiv.className = 'welcome-message';
        successDiv.innerHTML = `
            <h2>Chat Cleared</h2>
            <p>Your chat history has been deleted. Start a new conversation!</p>
        `;
        elements.chatHistory.appendChild(successDiv);
    } catch (err) {
        hideLoading();
        console.error('Clear history error:', err);
        
        let userMessage;
        if (err.name === 'TypeError' && err.message.includes('fetch')) {
            userMessage = 'Unable to connect to the server. Please check your connection.';
        } else {
            userMessage = `Could not clear history: ${err.message}`;
        }
        displayErrorMessage(userMessage);
    }
}


/**
 * Send message to backend (with streaming support)
 */
async function sendMessage(question) {
    syncUserStateFromInputs();
    if (!state.userId) {
        alert('Please provide a User ID before chatting.');
        return;
    }

    // Remove welcome message on first interaction
    removeWelcomeMessage();

    // Add user message to chat
    displayUserMessage(question);

    // Add to chat history
    state.chatHistory.push({
        role: 'user',
        content: question
    });

    // Create new AbortController for this request
    state.abortController = new AbortController();
    state.isStopped = false;

    // Show loading
    showLoading();
    state.isWaiting = true;
    elements.sendBtn.disabled = true;
    elements.userInput.disabled = true;

    const payload = {
        user_id: state.userId,
        user_name: state.userName,
        preprocess_query: state.preprocessQuery,
        chat_history: state.chatHistory,
        reasoning_effort: state.reasoningEffort,
    };

    console.log('Sending chat request with payload:', JSON.stringify(payload, null, 2));

    // Use streaming if enabled and preprocessing is off
    if (state.useStreaming) {
        await sendMessageStreaming(payload);
    } else {
        await sendMessageNonStreaming(payload);
    }
}

/**
 * Send message with streaming response
 */
async function sendMessageStreaming(payload) {
    try {
        const response = await fetch(`${CONFIG.BACKEND_URL}${CONFIG.CHAT_STREAM_ENDPOINT}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
            signal: state.abortController.signal,
        });

        // Check if stopped during fetch
        if (state.isStopped) {
            return;
        }

        // Hide the spinner/text but keep stop button visible during streaming
        elements.loadingIndicator.classList.add('hidden');

        if (!response.ok) {
            const errMsg = await extractErrorMessage(response, response.status);
            throw new Error(errMsg);
        }

        if (!response.body) {
            throw new Error('Streaming not supported by this browser.');
        }

        displaySources([]);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        const messageMap = new Map();
        const answerBuffers = new Map();
        const reasoningBuffers = new Map();
        const subqueryMeta = new Map();
        const aggregatedSources = [];

        let pendingBuffer = '';
        let multiMode = false;

        const ensureMessage = (idx, meta) => {
            const key = Number.isInteger(idx) ? idx : 0;
            if (!messageMap.has(key)) {
                const messageDiv = displayAssistantMessage('', '', true);
                messageMap.set(key, messageDiv);
            }
            return messageMap.get(key);
        };

        const appendAnswerToken = (idx, token) => {
            const key = Number.isInteger(idx) ? idx : 0;
            const prev = answerBuffers.get(key) || '';
            const next = prev + token;
            answerBuffers.set(key, next);
            return next;
        };

        const updateSources = (sources, meta) => {
            if (multiMode && meta && meta.subquery_index) {
                if (Array.isArray(sources) && sources.length) {
                    const prefix = meta.subquery_text
                        ? `Sub-query ${meta.subquery_index}: ${meta.subquery_text}`
                        : `Sub-query ${meta.subquery_index}`;
                    sources.forEach((src) => {
                        aggregatedSources.push(`${prefix} — ${src}`);
                    });
                    displaySources(aggregatedSources);
                }
            } else {
                displaySources(Array.isArray(sources) ? sources : []);
            }
        };

        const handleEvent = (data) => {
            if (!data || typeof data !== 'object') {
                return;
            }

            if (data.type === 'subquery_start' && typeof data.subquery_index === 'number') {
                multiMode = true;
                subqueryMeta.set(data.subquery_index, data);
                answerBuffers.set(data.subquery_index, '');
                reasoningBuffers.set(data.subquery_index, '');
                ensureMessage(data.subquery_index, data);
                return;
            }

            if (data.type === 'sources') {
                const idx = typeof data.subquery_index === 'number' ? data.subquery_index : 0;
                const meta = idx ? (subqueryMeta.get(idx) || data) : null;
                if (idx && meta && !subqueryMeta.has(idx)) {
                    subqueryMeta.set(idx, meta);
                }
                updateSources(data.sources || [], meta);
                return;
            }

            if (data.type === 'token' && typeof data.token === 'string') {
                const idx = typeof data.subquery_index === 'number' ? data.subquery_index : 0;
                const meta = idx ? (subqueryMeta.get(idx) || data) : null;
                if (idx && meta && !subqueryMeta.has(idx)) {
                    subqueryMeta.set(idx, meta);
                }
                const messageDiv = ensureMessage(idx, meta);
                const current = appendAnswerToken(idx, data.token);
                updateStreamingMessage(messageDiv, current);
                return;
            }

            if (data.type === 'reasoning' && typeof data.token === 'string') {
                const idx = typeof data.subquery_index === 'number' ? data.subquery_index : 0;
                const meta = idx ? (subqueryMeta.get(idx) || data) : null;
                if (idx && meta && !subqueryMeta.has(idx)) {
                    subqueryMeta.set(idx, meta);
                }
                const prev = reasoningBuffers.get(idx) || '';
                reasoningBuffers.set(idx, prev + data.token);
                return;
            }

            if (data.type === 'subquery_done' && typeof data.subquery_index === 'number') {
                const meta = subqueryMeta.get(data.subquery_index) || data;
                if (!subqueryMeta.has(data.subquery_index)) {
                    subqueryMeta.set(data.subquery_index, meta);
                }
                const totalCount = typeof meta.total_subqueries === 'number'
                    ? meta.total_subqueries
                    : (typeof data.total_subqueries === 'number' ? data.total_subqueries : 1);
                let displayText = (data.display_answer || data.answer || '').trim();
                if (!displayText) {
                    displayText = formatSubqueryAnswer(
                        data.subquery_index,
                        totalCount,
                        meta.subquery_text,
                        answerBuffers.get(data.subquery_index) || ''
                    );
                }
                const reasoningText = (data.reasoning || reasoningBuffers.get(data.subquery_index) || '').trim();
                const messageDiv = ensureMessage(data.subquery_index, meta);
                finalizeStreamingMessage(messageDiv, displayText, reasoningText);
                answerBuffers.set(data.subquery_index, displayText);
                reasoningBuffers.set(data.subquery_index, reasoningText);
                state.chatHistory.push({ role: 'assistant', content: displayText });
                return;
            }

            if (data.type === 'done') {
                if (multiMode && !data.answer && !data.display_answer) {
                    return;
                }
                const idx = typeof data.subquery_index === 'number' ? data.subquery_index : 0;
                const meta = idx ? (subqueryMeta.get(idx) || data) : null;
                if (idx && meta && !subqueryMeta.has(idx)) {
                    subqueryMeta.set(idx, meta);
                }
                const totalCount = meta && typeof meta.total_subqueries === 'number'
                    ? meta.total_subqueries
                    : (typeof data.total_subqueries === 'number' ? data.total_subqueries : 1);
                let displayText = (data.display_answer || data.answer || '').trim();
                if (!displayText && meta && meta.subquery_index) {
                    displayText = formatSubqueryAnswer(
                        meta.subquery_index,
                        totalCount,
                        meta.subquery_text,
                        answerBuffers.get(idx) || ''
                    );
                } else if (!displayText) {
                    displayText = answerBuffers.get(0) || '';
                }
                const reasoningText = (data.reasoning || reasoningBuffers.get(idx) || '').trim();
                const messageDiv = ensureMessage(idx, meta);
                finalizeStreamingMessage(messageDiv, displayText, reasoningText);
                answerBuffers.set(idx, displayText);
                reasoningBuffers.set(idx, reasoningText);
                if (!multiMode) {
                    state.chatHistory.push({ role: 'assistant', content: displayText });
                }
                return;
            }

            if (data.type === 'error') {
                throw new Error(data.message || 'Streaming error');
            }
        };

        const processBuffer = (buffer) => {
            let working = buffer;
            let boundary = working.indexOf('\n\n');
            while (boundary !== -1) {
                const rawEvent = working.slice(0, boundary).trim();
                working = working.slice(boundary + 2);
                if (!rawEvent) {
                    boundary = working.indexOf('\n\n');
                    continue;
                }
                const cleaned = rawEvent.replace(/\r/g, '');
                const dataLine = cleaned
                    .split('\n')
                    .map((line) => line.trim())
                    .find((line) => line.startsWith('data:'));
                if (!dataLine) {
                    boundary = working.indexOf('\n\n');
                    continue;
                }
                const chunk = dataLine.slice(5).trim();
                if (!chunk) {
                    boundary = working.indexOf('\n\n');
                    continue;
                }
                let parsed;
                try {
                    parsed = JSON.parse(chunk);
                } catch (err) {
                    console.error('Error parsing SSE data:', err);
                    boundary = working.indexOf('\n\n');
                    continue;
                }
                handleEvent(parsed);
                boundary = working.indexOf('\n\n');
            }
            return working;
        };

        while (true) {
            // Check if stopped by user
            if (state.isStopped) {
                reader.cancel();
                break;
            }
            const { done, value } = await reader.read();
            if (done) {
                break;
            }
            pendingBuffer += decoder.decode(value, { stream: true });
            pendingBuffer = processBuffer(pendingBuffer);
        }

        if (pendingBuffer.trim() && !state.isStopped) {
            processBuffer(`${pendingBuffer}\n\n`);
        }

    } catch (error) {
        // Don't show error if request was aborted by user
        if (error.name === 'AbortError' || state.isStopped) {
            return;
        }
        
        console.error('Chat error:', error);
        
        let userMessage;
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            userMessage = 'Unable to connect to the server. Please check your connection and try again.';
        } else {
            userMessage = `Error: ${error.message}`;
        }
        
        displayErrorMessage(userMessage);
    } finally {
        hideLoading();
        state.abortController = null;
        state.isWaiting = false;
        elements.sendBtn.disabled = false;
        elements.userInput.disabled = false;
        elements.userInput.focus();
    }
}

/**
 * Send message without streaming (fallback)
 */
async function sendMessageNonStreaming(payload) {
    try {
        const response = await fetch(`${CONFIG.BACKEND_URL}${CONFIG.CHAT_ENDPOINT}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
            signal: state.abortController.signal,
        });

        // Check if stopped during fetch
        if (state.isStopped) {
            return;
        }

        console.log('Response received:', response.status, response.statusText);

        if (!response.ok) {
            const errMsg = await extractErrorMessage(response, response.status);
            throw new Error(errMsg);
        }

        const data = await response.json();

        // Check if stopped before rendering
        if (state.isStopped) {
            return;
        }

        // Add assistant message to chat (with reasoning if available)
        displayAssistantMessage(data.answer, data.reasoning || '');

        // Add to chat history
        state.chatHistory.push({
            role: 'assistant',
            content: data.answer
        });

        // Display sources
        if (data.sources && data.sources.length > 0) {
            displaySources(data.sources);
        }

    } catch (error) {
        // Don't show error if request was aborted by user
        if (error.name === 'AbortError' || state.isStopped) {
            return;
        }
        
        console.error('Chat error:', error);
        
        let userMessage;
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            userMessage = 'Unable to connect to the server. Please check your connection and try again.';
        } else {
            userMessage = `Error: ${error.message}`;
        }
        
        displayErrorMessage(userMessage);
    } finally {
        hideLoading();
        state.abortController = null;
        state.isWaiting = false;
        elements.sendBtn.disabled = false;
        elements.userInput.disabled = false;
        elements.userInput.focus();
    }
}

/**
 * Stop the current chat request
 */
function stopChat() {
    if (state.abortController) {
        state.isStopped = true;
        state.abortController.abort();
        // Don't nullify here - let the finally block handle it
    }
    hideLoading();
    state.isWaiting = false;
    elements.sendBtn.disabled = false;
    elements.userInput.disabled = false;
    elements.userInput.focus();
}

/**
 * Handle form submission
 */
function handleFormSubmit(e) {
    e.preventDefault();

    const question = elements.userInput.value.trim();
    if (!question || state.isWaiting) {
        return;
    }

    elements.userInput.value = '';
    sendMessage(question);
}

/**
 * Initialize app
 */
function init() {
    hydrateUserInfo();
    syncUserStateFromInputs();

    // Add event listeners
    elements.chatForm.addEventListener('submit', handleFormSubmit);
    elements.stopBtn.addEventListener('click', stopChat);
    elements.userInput.focus();

    elements.userIdInput.addEventListener('change', syncUserStateFromInputs);
    elements.userNameInput.addEventListener('change', syncUserStateFromInputs);
    elements.loadHistoryBtn.addEventListener('click', loadChatHistory);
    elements.clearChatBtn.addEventListener('click', clearChatHistory);

    // Settings panel interactions
    elements.settingsBtn.addEventListener('click', () => {
        elements.settingsPanel.classList.toggle('hidden');
    });
    elements.settingsClose.addEventListener('click', () => {
        elements.settingsPanel.classList.add('hidden');
    });
    elements.preprocessToggle.addEventListener('change', (e) => {
        state.preprocessQuery = e.target.checked;
        // Disable reasoning effort when preprocessing is on
        elements.reasoningEffort.disabled = e.target.checked;
    });
    elements.reasoningEffort.addEventListener('change', (e) => {
        state.reasoningEffort = e.target.value;
    });
    // Close settings panel when clicking outside
    document.addEventListener('click', (e) => {
        if (!elements.settingsPanel.contains(e.target) && !elements.settingsBtn.contains(e.target)) {
            elements.settingsPanel.classList.add('hidden');
        }
    });

    // Menu interactions
    elements.menuToggle.addEventListener('click', () => {
        elements.menuPanel.classList.toggle('open');
    });
    elements.menuHome.addEventListener('click', () => window.location.href = '/home');
    elements.menuChat.addEventListener('click', () => window.location.href = '/chat');
    elements.menuClassification.addEventListener('click', () => window.location.href = '/classification');
    document.addEventListener('click', (e) => {
        if (!elements.menuPanel.contains(e.target) && !elements.menuToggle.contains(e.target)) {
            elements.menuPanel.classList.remove('open');
        }
    });

    // Remove welcome message on first input
    elements.userInput.addEventListener('focus', () => {
        if (elements.chatHistory.querySelector('.welcome-message')) {
            // Just focus, don't remove yet
        }
    });

    // Attempt to load existing history for stored user
    loadChatHistory();

    console.log('Chatbot initialized');
}

// Start app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
