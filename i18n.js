(() => {
  'use strict';
  const dictionary = window.WAM_ZH || {};
  const preferenceKey = 'internw0-language';
  const valid = value => value === 'en' || value === 'zh';
  const query = new URL(location.href).searchParams.get('lang');
  let saved;
  try { saved = localStorage.getItem(preferenceKey); } catch { /* Storage may be disabled. */ }
  let language = valid(query) ? query : valid(saved) ? saved : 'en';
  const originals = new WeakMap();
  const excluded = 'script,style,noscript,pre,code,svg,[translate="no"],[data-language-toggle]';
  const attributeNames = ['aria-label', 'alt', 'title'];
  const prefixes = [['Watch filtered example: ', '观看过滤片段：'], ['Preview placeholder for ', '查看占位说明：'], ['Play ', '播放：']];
  const suffixes = [[' — opens in a new tab', '（在新标签页打开）'], [' — forthcoming', '（即将发布）'], [' demonstrations', '演示']];

  function translate(raw) {
    if (language === 'en' || typeof raw !== 'string') return raw;
    const [, before, inner, after] = raw.match(/^(\s*)([\s\S]*?)(\s*)$/);
    const key = inner.replace(/\s+/g, ' ');
    let value = Object.hasOwn(dictionary, key) ? dictionary[key] : undefined;
    if (value === undefined) {
      for (const [prefix, translated] of prefixes) {
        if (key.startsWith(prefix)) { value = translated + translate(key.slice(prefix.length)); break; }
      }
    }
    if (value === undefined) {
      for (const [suffix, translated] of suffixes) {
        if (key.endsWith(suffix)) { value = translate(key.slice(0, -suffix.length)) + translated; break; }
      }
    }
    return value === undefined ? raw : before + value + after;
  }

  // Retain the exact English text per node/attribute. Switching never rebuilds
  // interactive elements, resets a video, or changes asset URLs or numeric data.
  function localizeValue(node, key, current, write) {
    let values = originals.get(node);
    if (!values) { values = new Map(); originals.set(node, values); }
    let entry = values.get(key);
    if (!entry || current !== entry.rendered) entry = { source: current };
    const translated = translate(entry.source);
    entry.rendered = translated;
    values.set(key, entry);
    if (current !== translated) write(translated);
  }

  function apply(root) {
    if (root.nodeType === Node.TEXT_NODE) {
      if (!root.parentElement?.closest(excluded)) localizeValue(root, 'text', root.data, value => { root.data = value; });
      return;
    }
    if (root.nodeType !== Node.ELEMENT_NODE || root.closest(excluded)) return;
    for (const name of attributeNames) {
      if (root.hasAttribute(name)) localizeValue(root, name, root.getAttribute(name), value => root.setAttribute(name, value));
    }
    if (root.matches('meta[name="description"],meta[property="og:title"],meta[property="og:description"]')) {
      localizeValue(root, 'content', root.content, value => { root.content = value; });
    }
    for (const child of root.childNodes) apply(child);
  }

  const toggle = document.querySelector('[data-language-toggle]');
  function render() {
    document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
    apply(document.documentElement);
    if (toggle) {
      toggle.hidden = false;
      toggle.textContent = language === 'en' ? '中文' : 'EN';
      toggle.lang = language === 'en' ? 'zh-CN' : 'en';
      toggle.setAttribute('aria-label', language === 'en' ? 'Switch to Chinese' : '切换到英文');
      toggle.title = language === 'en' ? '切换到中文' : 'Switch to English';
    }
  }

  // Translate only affected subtrees when existing controls add dialogs or cards.
  // Disconnect during writes so our own translations cannot trigger an idle loop.
  const observer = new MutationObserver(records => {
    observer.disconnect();
    for (const record of records) {
      if (record.type === 'childList') for (const node of record.addedNodes) apply(node);
      else apply(record.target);
    }
    observe();
  });
  function observe() {
    observer.observe(document.documentElement, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: [...attributeNames, 'content'] });
  }
  function remember() {
    try { localStorage.setItem(preferenceKey, language); } catch { /* The toggle still works. */ }
  }
  function setLanguage(next, updateUrl = true) {
    if (!valid(next)) return;
    language = next;
    observer.disconnect();
    render();
    observe();
    remember();
    if (updateUrl) {
      const url = new URL(location.href);
      url.searchParams.set('lang', language);
      try { history.replaceState(history.state, '', url); } catch { /* file:// preview */ }
    }
  }
  toggle?.addEventListener('click', () => setLanguage(language === 'en' ? 'zh' : 'en'));
  addEventListener('popstate', () => {
    const selected = new URL(location.href).searchParams.get('lang');
    if (valid(selected)) setLanguage(selected, false);
  });
  render();
  observe();
  if (valid(query)) remember();
  window.WAM_I18N = { setLanguage, get language() { return language; }, translate };
})();
