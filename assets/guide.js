/* Quant from Zero · guide behaviour: maths, theme, contents, copy buttons, reading progress */
(function () {
  var root = document.documentElement;
  function ready(f) { if (document.readyState !== 'loading') f(); else document.addEventListener('DOMContentLoaded', f); }
  ready(function () {
    if (window.renderMathInElement) {
      renderMathInElement(document.querySelector('.content') || document.body, {
        delimiters: [{ left: '$$', right: '$$', display: true }, { left: '\\[', right: '\\]', display: true }, { left: '$', right: '$', display: false }],
        throwOnError: false, ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
      });
    }
    var tb = document.querySelector('.themebtn');
    if (tb) tb.addEventListener('click', function () {
      var dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
      root.dataset.theme = dark ? 'light' : 'dark';
      try { localStorage.setItem('qfz-theme', root.dataset.theme); } catch (e) {}
    });
    var toc = document.getElementById('toc'), btn = document.querySelector('.tocbtn');
    if (btn && toc) {
      btn.addEventListener('click', function () { var o = toc.classList.toggle('open'); btn.setAttribute('aria-expanded', o); });
      toc.addEventListener('click', function (e) { if (e.target.tagName === 'A') { toc.classList.remove('open'); btn.setAttribute('aria-expanded', false); } });
    }
    document.querySelectorAll('pre.code').forEach(function (pre) {
      var b = document.createElement('button'); b.className = 'copy'; b.type = 'button'; b.textContent = 'Copy';
      b.addEventListener('click', function () {
        var txt = pre.querySelector('code').innerText;
        (navigator.clipboard ? navigator.clipboard.writeText(txt) : Promise.reject()).then(function () { b.textContent = 'Copied'; setTimeout(function () { b.textContent = 'Copy'; }, 1200); }, function () { b.textContent = 'Select & copy'; });
      });
      pre.appendChild(b);
    });
    var bar = document.querySelector('.progress'), links = Array.prototype.slice.call(document.querySelectorAll('.toc a'));
    var heads = links.map(function (a) { return document.getElementById(a.getAttribute('href').slice(1)); });
    function onScroll() {
      var h = document.documentElement.scrollHeight - innerHeight;
      if (bar) bar.style.width = (h > 0 ? 100 * scrollY / h : 0) + '%';
      var cur = -1; heads.forEach(function (el, i) { if (el && el.getBoundingClientRect().top < 120) cur = i; });
      links.forEach(function (a, i) { a.classList.toggle('on', i === cur); });
    }
    addEventListener('scroll', onScroll, { passive: true }); onScroll();
  });
})();
