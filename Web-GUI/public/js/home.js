const HOME_STATE_KEY = 'skysafe-user';

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

function persistUser(id, name) {
    setCookie('skysafe_user_id', id);
    setCookie('skysafe_user_name', name);
    try {
        localStorage.setItem(HOME_STATE_KEY, JSON.stringify({ userId: id, userName: name }));
    } catch (e) {
        console.debug('localStorage unavailable');
    }
}

function hydrate() {
    const idFromCookie = getCookie('skysafe_user_id');
    const nameFromCookie = getCookie('skysafe_user_name');
    let stored = null;
    try {
        stored = JSON.parse(localStorage.getItem(HOME_STATE_KEY) || 'null');
    } catch (e) {
        stored = null;
    }

    const userId = idFromCookie || stored?.userId || `user_${Date.now()}`;
    const userName = nameFromCookie || stored?.userName || 'Guest';

    document.getElementById('home-user-id').value = userId;
    document.getElementById('home-user-name').value = userName;
}

function clearAll() {
    document.getElementById('home-user-id').value = '';
    document.getElementById('home-user-name').value = '';
    clearUserCookies();
    try { localStorage.removeItem(HOME_STATE_KEY); } catch (e) {}
}

function goTo(path) {
    const id = (document.getElementById('home-user-id').value || '').trim() || `user_${Date.now()}`;
    const name = (document.getElementById('home-user-name').value || '').trim() || 'Guest';
    persistUser(id, name);
    window.location.href = path;
}

function init() {
    hydrate();
    document.getElementById('home-chat-btn').addEventListener('click', () => goTo('/chat'));
    document.getElementById('home-class-btn').addEventListener('click', () => goTo('/classification'));
    document.getElementById('home-clear-btn').addEventListener('click', clearAll);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
