(() => {
  'use strict';
  // Visibility-driven playback: no background polling and no eager clip downloads.
  class InlineVideoManager {
    constructor({ resolveUrl, dialog } = {}) {
      this.resolveUrl = resolveUrl;
      this.dialog = dialog;
      this.items = new Map();
      this.reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
      this.observer = 'IntersectionObserver' in window ? new IntersectionObserver(entries => {
        for (const entry of entries) {
          const state = this.items.get(entry.target);
          if (!state) continue;
          state.visible = entry.isIntersecting && entry.intersectionRatio >= 0.35;
          this.sync(state);
        }
      }, { threshold: [0, 0.35] }) : null;
      document.addEventListener('visibilitychange', () => this.refresh());
      window.addEventListener('pagehide', () => { for (const s of this.items.values()) this.pause(s); });
      window.addEventListener('pageshow', () => this.refresh());
      const preferenceChanged = () => {
        if (!this.allowAutoplay()) for (const s of this.items.values()) this.pause(s);
        this.refresh();
      };
      this.reducedMotion.addEventListener('change', preferenceChanged);
      navigator.connection?.addEventListener?.('change', preferenceChanged);
      if (dialog) new MutationObserver(() => this.refresh()).observe(dialog, { attributes: true, attributeFilter: ['open'] });
    }
    allowAutoplay() { return Boolean(this.observer) && !this.reducedMotion.matches && !navigator.connection?.saveData; }
    active(s) { return !s.disposed && s.visible && !document.hidden && !this.dialog?.open; }
    pause(s) {
      if (!s.video.paused) {
        if (s.pending) s.interrupted = true;
        s.managedPause = true; s.video.pause();
      }
    }
    load(s) {
      if (s.loaded || s.disposed) return;
      s.loaded = true;
      s.video.preload = navigator.connection?.saveData ? 'none' : 'metadata';
      s.video.src = s.url;
    }
    sync(s) {
      if (!this.active(s)) { this.pause(s); return; }
      this.load(s);
      if (!this.allowAutoplay() || s.userPaused || s.completed || s.blocked || s.failed || s.pending || !s.video.paused) return;
      s.pending = true; s.interrupted = false;
      Promise.resolve(s.video.play()).catch(error => {
        // AbortError is expected when a clip leaves the viewport or its tab is removed.
        if (!s.disposed && (error?.name !== 'AbortError' || !s.interrupted)) s.blocked = true;
      }).finally(() => {
        s.pending = false;
        if (!this.active(s)) this.pause(s);
        else if (s.interrupted) this.sync(s);
      });
    }
    refresh() { for (const s of this.items.values()) this.sync(s); }
    mount(host, media, { loop = false } = {}) {
      const url = this.resolveUrl(media.src);
      if (!url || url.startsWith('data:')) return;
      const video = host.querySelector('video') || document.createElement('video');
      video.className = 'inline-video';
      video.controls = true;
      video.defaultMuted = true;
      video.muted = true;
      video.playsInline = true;
      video.loop = loop;
      video.preload = 'none';
      video.setAttribute('muted', '');
      video.setAttribute('playsinline', '');
      video.setAttribute('aria-label', media.title);
      if (this.resolveUrl(media.poster)) video.poster = this.resolveUrl(media.poster);
      host.classList.add('inline-media');
      host.replaceChildren(video);
      const retry = document.createElement('button');
      retry.className = 'inline-video-retry'; retry.type = 'button';
      retry.textContent = 'Retry video'; retry.hidden = true;
      host.append(retry);
      const s = { video, host, url, visible: !this.observer, loaded: false, disposed: false,
        userPaused: false, completed: false, blocked: false, failed: false, pending: false, interrupted: false, managedPause: false, listeners: [] };
      const on = (type, fn) => { video.addEventListener(type, fn); s.listeners.push([type, fn]); };
      const prepareManual = () => { s.blocked = false; this.load(s); };
      on('pointerdown', prepareManual);
      on('keydown', prepareManual);
      on('play', () => {
        if (s.disposed) return;
        s.userPaused = false; s.completed = false; s.blocked = false;
        if (!this.active(s)) this.pause(s);
      });
      on('playing', () => { if (!this.active(s)) this.pause(s); });
      on('pause', () => {
        if (s.managedPause) { s.managedPause = false; return; }
        if (this.active(s) && !video.ended) s.userPaused = true;
      });
      on('ended', () => { s.completed = true; });
      on('volumechange', () => {
        if (video.muted || video.volume === 0) return;
        for (const other of this.items.values()) if (other !== s) other.video.muted = true;
      });
      on('error', () => { if (!s.disposed) { s.failed = true; retry.hidden = false; } });
      retry.addEventListener('click', () => {
        s.failed = false; s.blocked = false; s.userPaused = false; s.completed = false;
        retry.hidden = true; this.load(s); video.load();
        video.play().catch(() => { /* Native controls remain available. */ });
      });
      this.items.set(video, s);
      if (this.observer) this.observer.observe(video);
      else this.load(s); // No observer: keep native manual controls; do not autoplay all clips.
      return video;
    }
    dispose(root) {
      for (const [video, s] of this.items) {
        if (!root.contains(video)) continue;
        s.disposed = true;
        this.observer?.unobserve(video);
        for (const [type, fn] of s.listeners) video.removeEventListener(type, fn);
        video.pause(); video.removeAttribute('src'); video.load();
        this.items.delete(video);
      }
    }
  }
  window.WAMInlineVideoManager = InlineVideoManager;
})();
