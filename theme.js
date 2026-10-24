/* System / Light / Dark. "System" removes the override and the CSS follows
 * prefers-color-scheme. Storage can be missing (private mode), so every
 * access is guarded and the page still renders with the system theme. */
(function () {
  const KEY = 'facecam-theme';
  const root = document.documentElement;
  let saved = 'system';
  try { saved = localStorage.getItem(KEY) || 'system'; } catch (e) {}

  function apply(t) {
    if (t === 'light' || t === 'dark') root.dataset.theme = t;
    else delete root.dataset.theme;
  }
  apply(saved);

  document.addEventListener('DOMContentLoaded', () => {
    const sel = document.getElementById('theme');
    if (!sel) return;
    sel.value = saved;
    sel.addEventListener('change', () => {
      apply(sel.value);
      try { localStorage.setItem(KEY, sel.value); } catch (e) {}
    });
  });
})();
