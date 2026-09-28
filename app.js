(() => {
  'use strict';
  const content = window.WAM_CONTENT;
  if (!content) return;
  const $ = (s, root = document) => root.querySelector(s);
  const $$ = (s, root = document) => [...root.querySelectorAll(s)];
  const make = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const safeUrl = value => {
    if (typeof value !== 'string' || !value.trim()) return '';
    try {
      const url = new URL(value, location.href);
      // Embedded image/PDF URLs are produced only by the local preview exporter.
      if (url.protocol === 'data:' && /^data:(image\/(png|jpeg|webp|svg\+xml)|application\/pdf);base64,/i.test(value)) return value;
      return ['http:', 'https:', 'file:'].includes(url.protocol) ? url.href : '';
    } catch { return ''; }
  };
  $$('[data-project-name]').forEach(n => n.textContent = content.name);
  $$('[data-project-title]').forEach(n => n.textContent = content.title);
  $$('[data-project-subtitle]').forEach(n => n.textContent = content.subtitle);
  $$('[data-year]').forEach(n => n.textContent = content.year);
  document.title = content.name + ': ' + content.title;
  $('meta[property="og:title"]').content = document.title;
  const figures = new Map(content.figures.map(f => [f.id, f]));
  const pdfCache = new Map();
  function pdfUrl(value) {
    const url = safeUrl(value);
    if (!url.startsWith('data:application/pdf;base64,')) return url;
    if (!pdfCache.has(url)) {
      const bytes = Uint8Array.from(atob(url.split(',')[1]), c => c.charCodeAt(0));
      pdfCache.set(url, URL.createObjectURL(new Blob([bytes], { type: 'application/pdf' })));
    }
    return pdfCache.get(url);
  }
  // Reflect configured figures in both the static page and the full-size viewer.
  content.figures.forEach(f => {
    const root = $('#figure-' + f.id);
    if (!root) return;
    const img = $('img', root);
    img.src = safeUrl(f.image);
    img.alt = f.alt;
    $('a', root).href = pdfUrl(f.pdf);
    img.addEventListener('error', () => {
      const note = make('p', 'figure-load-error', 'Figure preview could not load. Please use the PDF link below.');
      $('.figure-image-button', root).replaceWith(note);
    }, { once: true });
  });
  const dialog = $('#media-dialog');
  const body = $('#dialog-body');
  let trigger = null;
  function resetDialog(title, description, figure = false) {
    trigger = document.activeElement;
    dialog.classList.toggle('figure-dialog', figure);
    $('#figure-viewer-tools').hidden = !figure;
    $('#dialog-title').textContent = title;
    $('#dialog-description').textContent = description;
    body.replaceChildren();
  }
  function showDialog() {
    dialog.showModal();
    document.body.style.overflow = 'hidden';
    $('.dialog-close', dialog).focus();
  }
  function placeholder(title, text) {
    const block = make('div', 'dialog-placeholder');
    block.append(make('strong', '', title), make('p', '', text));
    return block;
  }
  $$('[data-figure]').forEach(button => button.addEventListener('click', () => {
    const f = figures.get(button.dataset.figure);
    if (!f) return;
    resetDialog(f.label, f.alt, true);
    const viewport = make('div', 'figure-viewport');
    viewport.tabIndex = 0;
    viewport.setAttribute('role', 'region');
    viewport.setAttribute('aria-label', 'Scrollable full-resolution figure');
    const image = make('img', 'viewer-image');
    image.src = safeUrl(f.image); image.alt = f.alt;
    image.addEventListener('error', () => viewport.replaceChildren(placeholder('Image unavailable', 'Use Open PDF to view the original figure.')), { once: true });
    viewport.append(image); body.append(viewport);
    $('#figure-zoom').setAttribute('aria-pressed', 'false');
    $('#figure-zoom').textContent = 'Original size';
    $('#figure-pdf').href = pdfUrl(f.pdf);
    showDialog();
  }));
  $('#figure-zoom').addEventListener('click', () => {
    const viewport = $('.figure-viewport', body);
    if (!viewport) return;
    const zoomed = viewport.classList.toggle('is-zoomed');
    $('#figure-zoom').setAttribute('aria-pressed', String(zoomed));
    $('#figure-zoom').textContent = zoomed ? 'Fit to window' : 'Original size';
    viewport.scrollTo(0, 0);
  });
  const inlineMedia = new window.WAMInlineVideoManager({ resolveUrl: safeUrl, dialog });
  const hero = content.hero || { title: 'Project video', subtitle: 'Project video forthcoming.', src: '', poster: '' };
  if ($('#hero-media')) inlineMedia.mount($('#hero-media'), hero);
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  $('.dialog-close', dialog).addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => {
    if (event.target !== dialog) return;
    const r = dialog.getBoundingClientRect();
    if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dialog.close();
  });
  dialog.addEventListener('close', () => {
    document.body.style.overflow = '';
    trigger?.focus({ preventScroll: true });
  });
  const names = { paper: 'Technical report', code: 'Source code', models: 'Model checkpoints' };
  $$('[data-resource]').forEach(button => {
    const key = button.dataset.resource, url = safeUrl(content.links[key]);
    button.setAttribute('aria-label', names[key] + (url ? ' — opens in a new tab' : ' — forthcoming'));
    button.addEventListener('click', () => {
      if (url && !url.startsWith('data:')) window.open(url, '_blank', 'noopener,noreferrer');
      else {
        resetDialog(names[key], 'A release link has not been provided for this preview.');
        body.append(placeholder('Forthcoming', 'This resource will be linked when it becomes available.'));
        showDialog();
      }
    });
  });
  $$('[data-state]').forEach(n => {
    const key = n.dataset.state;
    if (safeUrl(content.links[key])) {
      n.textContent = content.linkLabels?.[key] || 'Available ↗';
      n.classList.add('available');
    }
  });
  const tabsRoot = $('#demo-tabs');
  const demoStrip = $('#demo-grid');
  const demoScrollControls = $('#demo-scroll-controls');
  const demoPrevious = $('#demo-prev');
  const demoNext = $('#demo-next');
  function updateDemoScroll() {
    const limit = Math.max(0, demoStrip.scrollWidth - demoStrip.clientWidth);
    demoScrollControls.hidden = limit <= 2;
    demoPrevious.disabled = demoStrip.scrollLeft <= 2;
    demoNext.disabled = demoStrip.scrollLeft >= limit - 2;
    demoStrip.tabIndex = limit > 2 ? 0 : -1;
  }
  function scrollDemos(direction) {
    const card = $('.demo-card', demoStrip);
    if (!card) return;
    const step = card.getBoundingClientRect().width + (parseFloat(getComputedStyle(demoStrip).columnGap) || 0);
    demoStrip.scrollBy({ left: direction * step, behavior: reducedMotion.matches ? 'instant' : 'smooth' });
  }
  demoPrevious.addEventListener('click', () => scrollDemos(-1));
  demoNext.addEventListener('click', () => scrollDemos(1));
  demoStrip.addEventListener('scroll', updateDemoScroll, { passive: true });
  demoStrip.addEventListener('keydown', event => {
    if (event.target !== demoStrip) return;
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault();
      scrollDemos(event.key === 'ArrowRight' ? 1 : -1);
    } else if (event.key === 'Home' || event.key === 'End') {
      event.preventDefault();
      demoStrip.scrollTo({ left: event.key === 'Home' ? 0 : demoStrip.scrollWidth, behavior: reducedMotion.matches ? 'instant' : 'smooth' });
    }
  });
  if ('ResizeObserver' in window) new ResizeObserver(updateDemoScroll).observe(demoStrip);
  else window.addEventListener('resize', updateDemoScroll);
  const playableDemoCount = content.demos.reduce((total, category) => total + category.items.filter(demo => Boolean(safeUrl(demo.src))).length, 0);
  if (playableDemoCount) {
    $('#demos-media-note').textContent = 'Explore demonstrations with grippers and dexterous hands.';
  }
  const tabs = content.demos.map(category => {
    const button = make('button', '', category.label);
    button.id = 'demo-tab-' + category.id;
    button.setAttribute('role', 'tab');
    button.setAttribute('aria-controls', 'demo-panel');
    tabsRoot.append(button);
    return button;
  });
  function selectDemo(index, focus = false) {
    tabs.forEach((tab, i) => { tab.setAttribute('aria-selected', String(i === index)); tab.tabIndex = i === index ? 0 : -1; });
    if (focus) tabs[index].focus();
    $('#demo-panel').setAttribute('aria-labelledby', tabs[index].id);
    const category = content.demos[index];
    $('#demo-description').textContent = category.description;
    demoStrip.scrollTo({ left: 0, behavior: 'instant' });
    inlineMedia.dispose(demoStrip);
    demoStrip.replaceChildren();
    demoStrip.setAttribute('aria-label', category.label + ' demonstrations');
    const hasPublishedVideos = category.items.some(demo => Boolean(safeUrl(demo.src)));
    $('.demo-disclaimer').textContent = hasPublishedVideos
      ? 'Qualitative demonstrations. Playback-speed annotations, where shown, are part of the supplied footage.'
      : 'These slots are placeholders and do not represent experimental results or measured performance.';
    category.items.forEach((demo, demoIndex) => {
      const card = make('article', 'demo-card');
      const media = make('div', 'demo-media has-media');
      const title = make('h3', 'demo-card-title', demo.title);
      const caption = make('p', 'demo-caption', demo.subtitle);
      title.id = `demo-${category.id}-${demoIndex}-title`;
      caption.id = `demo-${category.id}-${demoIndex}-caption`;
      card.setAttribute('aria-labelledby', title.id);
      const video = inlineMedia.mount(media, demo, { loop: true });
      video?.setAttribute('aria-describedby', caption.id);
      card.append(media, title, caption);
      demoStrip.append(card);
    });
    demoStrip.scrollTo({ left: 0, behavior: 'instant' });
    updateDemoScroll();
    requestAnimationFrame(updateDemoScroll);
  }
  tabs.forEach((tab, i) => tab.addEventListener('click', () => selectDemo(i)));
  tabsRoot.addEventListener('keydown', e => {
    const index = tabs.indexOf(document.activeElement);
    if (index < 0) return;
    const target = e.key === 'ArrowRight' ? (index+1)%tabs.length : e.key === 'ArrowLeft' ? (index-1+tabs.length)%tabs.length : e.key === 'Home' ? 0 : e.key === 'End' ? tabs.length-1 : null;
    if (target !== null) { e.preventDefault(); selectDemo(target, true); }
  });
  selectDemo(0);
  // Filtering clips play in place too; the category explanation remains above them.
  const filterExamples = content.dataFiltering?.examples || [];
  if (filterExamples.length) {
    const root = $('#filter-examples');
    const categories = content.dataFiltering.categories || [];
    const filterTabs = categories.map(category => {
      const button = make('button', '', category.label);
      button.id = 'filter-tab-' + category.id;
      button.setAttribute('role', 'tab');
      button.setAttribute('aria-controls', 'filter-panel');
      $('#filter-tabs').append(button);
      return button;
    });
    function selectFilter(index, focus = false) {
      const category = categories[index];
      filterTabs.forEach((tab, i) => { tab.setAttribute('aria-selected', String(i === index)); tab.tabIndex = i === index ? 0 : -1; });
      if (category) {
        $('#filter-panel').setAttribute('aria-labelledby', filterTabs[index].id);
        $('#filter-description').textContent = category.description;
      }
      if (focus) filterTabs[index].focus({ preventScroll: true });
      const examples = category ? filterExamples.filter(item => item.category === category.id) : filterExamples;
      inlineMedia.dispose(root);
      // Also release the static no-JS fallback players on the first render.
      $$('video[src]', root).forEach(video => { video.pause(); video.removeAttribute('src'); video.load(); });
      root.replaceChildren();
      for (const example of examples) {
        const card = make('article', 'filter-card');
        const media = make('div', 'filter-media');
        if (Number.isFinite(example.width) && example.width > 0 && Number.isFinite(example.height) && example.height > 0) {
          media.style.aspectRatio = `${example.width} / ${example.height}`;
        }
        inlineMedia.mount(media, example, { loop: true });
        card.append(media); root.append(card);
      }
    }
    filterTabs.forEach((tab, index) => tab.addEventListener('click', () => selectFilter(index)));
    $('#filter-tabs').addEventListener('keydown', event => {
      const index = filterTabs.indexOf(document.activeElement);
      if (index < 0) return;
      const next = event.key === 'ArrowRight' ? (index + 1) % filterTabs.length : event.key === 'ArrowLeft' ? (index - 1 + filterTabs.length) % filterTabs.length : event.key === 'Home' ? 0 : event.key === 'End' ? filterTabs.length - 1 : null;
      if (next !== null) { event.preventDefault(); selectFilter(next, true); }
    });
    selectFilter(0);
  }
  // Zero-based animated bars; per-benchmark upper bounds are labeled below.
  // Methods without a supplied value are omitted, not plotted as zero.
  const experiments = content.experiments;
  if (experiments) {
    $('#results-note').textContent = experiments.note;
    const mark = (asset, className) => {
      if (!asset?.src || !safeUrl(asset.src)) return null;
      const image = make('img', className);
      if (asset.wide) image.classList.add(className + '--wide');
      image.src = safeUrl(asset.src); image.alt = ''; image.width = 32; image.height = 32;
      image.loading = 'lazy'; image.decoding = 'async';
      image.addEventListener('error', () => image.remove(), { once: true });
      return image;
    };
    for (const benchmark of experiments.benchmarks) {
      const card = make('article', 'benchmark-card');
      const heading = make('h3', '', benchmark.title);
      heading.id = 'benchmark-' + benchmark.id;
      card.setAttribute('aria-labelledby', heading.id);
      const header = make('div', 'benchmark-heading');
      const benchmarkMark = mark(benchmark.mark, 'benchmark-mark');
      if (benchmarkMark) header.append(benchmarkMark);
      const title = make('div');
      title.append(heading, make('p', 'benchmark-setting', benchmark.setting || 'Benchmark evaluation'));
      header.append(title); card.append(header);
      const results = benchmark.results.filter(result => Number.isFinite(result.value) && result.value >= 0 && result.value <= 100);
      const axisMax = [30, 100].includes(benchmark.axisMax) && results.every(r => r.value <= benchmark.axisMax) ? benchmark.axisMax : 100;
      const metric = make('p', 'chart-metric', 'Success rate (%) ↑');
      const chart = make('div', 'comparison-chart');
      const rows = make('ul', 'comparison-rows');
      for (const [index, result] of results.entries()) {
        const row = make('li', 'comparison-row');
        row.classList.toggle('is-ours', result.method === content.name);
        row.style.setProperty('--bar-delay', index * 90 + 'ms');
        const label = make('div', 'comparison-label');
        const method = make('span', 'comparison-method');
        const methodMark = mark(experiments.methodMarks?.[result.method], 'method-mark');
        if (methodMark) method.append(methodMark);
        method.append(make('span', '', result.method));
        const valueLabel = Number.isInteger(result.value) ? result.value.toFixed(1) : String(result.value);
        label.append(method, make('span', 'comparison-value', valueLabel + '%'));
        row.append(label);
        const track = make('div', 'comparison-track'); track.setAttribute('aria-hidden', 'true');
        const bar = make('div', 'comparison-bar');
        bar.style.setProperty('--result', (result.value / axisMax * 100) + '%');
        track.append(bar); row.append(track);
        rows.append(row);
      }
      const axis = make('div', 'comparison-axis'); axis.setAttribute('aria-hidden', 'true');
      for (const tick of axisMax === 30 ? [0, 10, 20, 30] : [0, 25, 50, 75, 100]) axis.append(make('span', '', String(tick)));
      chart.append(rows, axis); card.append(metric, chart);
      $('#benchmark-grid').append(card);
    }
    const benchmarkStrip = $('#benchmark-grid');
    const benchmarkControls = $('#benchmark-scroll-controls');
    const benchmarkPrevious = $('#benchmark-prev');
    const benchmarkNext = $('#benchmark-next');
    function updateBenchmarkScroll() {
      const limit = Math.max(0, benchmarkStrip.scrollWidth - benchmarkStrip.clientWidth);
      if (benchmarkControls) benchmarkControls.hidden = limit <= 2;
      if (benchmarkPrevious) benchmarkPrevious.disabled = benchmarkStrip.scrollLeft <= 2;
      if (benchmarkNext) benchmarkNext.disabled = benchmarkStrip.scrollLeft >= limit - 2;
    }
    function scrollBenchmarks(direction) {
      const card = benchmarkStrip.firstElementChild;
      if (!card) return;
      const step = card.getBoundingClientRect().width + (parseFloat(getComputedStyle(benchmarkStrip).columnGap) || 0);
      benchmarkStrip.scrollBy({ left: direction * step, behavior: reducedMotion.matches ? 'instant' : 'smooth' });
    }
    benchmarkPrevious?.addEventListener('click', () => scrollBenchmarks(-1));
    benchmarkNext?.addEventListener('click', () => scrollBenchmarks(1));
    benchmarkStrip.addEventListener('scroll', updateBenchmarkScroll, { passive: true });
    if ('ResizeObserver' in window) new ResizeObserver(updateBenchmarkScroll).observe(benchmarkStrip);
    else window.addEventListener('resize', updateBenchmarkScroll);
    benchmarkStrip.addEventListener('keydown', event => {
      if (event.target !== benchmarkStrip || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') scrollBenchmarks(event.key === 'ArrowRight' ? 1 : -1);
      else benchmarkStrip.scrollTo({ left: event.key === 'Home' ? 0 : benchmarkStrip.scrollWidth, behavior: reducedMotion.matches ? 'instant' : 'smooth' });
    });
    updateBenchmarkScroll();
    // A single entrance per card; never animate the background or replay on every scroll.
    if ('IntersectionObserver' in window && !reducedMotion.matches) {
      const cards = $$('.benchmark-card');
      const observer = new IntersectionObserver(entries => {
        for (const entry of entries) if (entry.isIntersecting) {
          entry.target.classList.remove('chart-pending');
          observer.unobserve(entry.target);
        }
      }, { threshold: 0.18 });
      cards.forEach(card => { card.classList.add('chart-pending', 'chart-reveal'); observer.observe(card); });
      reducedMotion.addEventListener('change', event => {
        if (!event.matches) return;
        cards.forEach(card => card.classList.remove('chart-pending', 'chart-reveal'));
        observer.disconnect();
      }, { once: true });
    }
  }
  if (content.team) {
    const target = $('#authors'); target.replaceChildren();
    target.append(make('p', 'author-affiliation-statement', content.team.affiliationStatement));
    const additional = make('p', 'author-additional-notes');
    content.team.additionalAffiliations.forEach((author, index) => {
      if (index) additional.append(document.createTextNode(' '));
      additional.append(make('span', 'author-university-note', `${author.name} is also affiliated with ${author.institution}.`));
    });
    target.append(additional);
  }
  if (content.citation.trim()) {
    $('#citation-block').hidden = false;
    const citation = $('#citation-text'); citation.replaceChildren();
    content.citation.split('\n').forEach((line, index) => {
      if (index) citation.append(document.createTextNode('\n'));
      const node = make('span', 'citation-line', line);
      // Keep soft-wrapped lines aligned too, including narrow mobile viewports.
      const indent = line.match(/^(?:\s*[A-Za-z]+\s*=\s*\{|\s+)/)?.[0].length || 0;
      node.style.setProperty('--citation-indent', `${indent}ch`);
      citation.append(node);
    });
  }
  let timer;
  function toast(text) {
    clearTimeout(timer); $('#toast').textContent = text; $('#toast').classList.add('visible');
    timer = setTimeout(() => $('#toast').classList.remove('visible'), 3500);
  }
  $('#copy-citation').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(content.citation); toast('BibTeX copied.'); }
    catch {
      const range = document.createRange(); range.selectNodeContents($('#citation-text'));
      const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range);
      toast('Citation selected. Press Ctrl+C or ⌘C to copy.');
    }
  });
  const menu = $('.menu-toggle'), mobile = $('#mobile-nav');
  function closeMenu(returnFocus = false) {
    mobile.hidden = true; menu.setAttribute('aria-expanded', 'false'); menu.setAttribute('aria-label', 'Open navigation');
    if (returnFocus) menu.focus();
  }
  menu.addEventListener('click', () => {
    const open = menu.getAttribute('aria-expanded') !== 'true';
    menu.setAttribute('aria-expanded', String(open)); menu.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
    mobile.hidden = !open;
  });
  $$('a', mobile).forEach(link => link.addEventListener('click', () => closeMenu()));
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && !mobile.hidden) closeMenu(true); });
  document.addEventListener('click', e => { if (!$('#header').contains(e.target)) closeMenu(); });
  matchMedia('(min-width: 1101px)').addEventListener('change', e => { if (e.matches) closeMenu(); });
})();
